from functools import partial
from typing import Optional

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.typing import Batch, PRNGKey
from jaxrl_m.networks.diffusion_nets import (
    FourierFeatures,
    ScoreActor,
    cosine_beta_schedule,
    vp_beta_schedule,
)
from jaxrl_m.networks.mlp import MLP, MLPResNet
from jaxrl_m.vision.data_augmentations import batched_random_crop


def ddpm_bc_loss(noise_prediction, noise, loss_weights: Optional[jnp.ndarray] = None):
    ddpm_loss = jnp.square(noise_prediction - noise).sum(-1)
    chex.assert_rank(ddpm_loss, 2)  # (batch, action_horizon)
    if loss_weights is not None:
        chex.assert_shape(loss_weights, ddpm_loss[:, 0].shape)
        ddpm_loss = ddpm_loss * loss_weights[:, None]
    return ddpm_loss.mean(), {
        "ddpm_loss": ddpm_loss,
        "ddpm_loss_mean": ddpm_loss.mean(),
        **(
            {
                "ddpm_loss_weights_mean": loss_weights.mean(),
                "ddpm_loss_weights_std": loss_weights.std(),
                "ddpm_loss_weights_min": loss_weights.min(),
                "ddpm_loss_weights_max": loss_weights.max(),
            }
            if loss_weights is not None
            else {}
        ),
    }


class DDPMBCAgent(flax.struct.PyTreeNode):
    """
    Models action distribution with a diffusion model.

    Assumes observation histories as input and action sequences as output.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(
        self,
        batch: Batch,
        pmap_axis: str = None,
        loss_weights: Optional[jnp.ndarray] = None,
        **kwargs,
    ):
        # Optionally apply DRQ augmentation
        if self.config.get("drq_padding", 0) > 0:
            rng, key = jax.random.split(self.state.rng)
            # Use same key for both observations and next_observations
            batch["observations"]["image"] = batched_random_crop(
                batch["observations"]["image"],
                key,
                padding=self.config["drq_padding"],
                num_batch_dims=2,
            )
            batch["next_observations"]["image"] = batched_random_crop(
                batch["next_observations"]["image"],
                key,
                padding=self.config["drq_padding"],
                num_batch_dims=2,
            )

        def actor_loss_fn(params, rng):
            key, rng = jax.random.split(rng)
            time = jax.random.randint(
                key, (batch["actions"].shape[0],), 0, self.config["diffusion_steps"]
            )
            key, rng = jax.random.split(rng)
            noise_sample = jax.random.normal(key, batch["actions"].shape)

            alpha_hats = self.config["alpha_hats"][time]
            time = time[:, None]
            alpha_1 = jnp.sqrt(alpha_hats)[:, None, None]
            alpha_2 = jnp.sqrt(1 - alpha_hats)[:, None, None]

            noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

            rng, key = jax.random.split(rng)
            noise_pred = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                batch["observations"],
                noisy_actions,
                time,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )

            return ddpm_bc_loss(
                noise_pred,
                noise_sample,
                loss_weights=loss_weights,
            )

        loss_fns = {
            "actor": actor_loss_fn,
        }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # log learning rates
        info["actor_lr"] = self.lr_schedules["actor"](self.state.step)

        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: PRNGKey = None,
        temperature: float = 1.0,
        clip_sampler: bool = True,
        **kwargs,
    ) -> jnp.ndarray:
        if isinstance(observations, dict):
            if "encoding" in observations:
                assert (
                    len(observations["encoding"].shape) > 1
                ), "Must use observation histories"
            else:
                assert (
                    len(observations["image"].shape) > 3
                ), "Must use observation histories"
            assert (
                len(observations["proprio"].shape) > 1
            ), "Must use observation histories"
        else:
            if self.config["image_observations"]:
                assert len(observations.shape) > 3, "Must use observation histories"
            else:
                assert len(observations.shape) > 1, "Must use observation histories"

        def fn(input_tuple, time):
            current_x, rng = input_tuple
            input_time = jnp.broadcast_to(time, (current_x.shape[0], 1))

            eps_pred = self.state.apply_fn(
                {"params": self.state.target_params},
                observations,
                current_x,
                input_time,
                name="actor",
            )

            alpha_1 = 1 / jnp.sqrt(self.config["alphas"][time])
            alpha_2 = (1 - self.config["alphas"][time]) / (
                jnp.sqrt(1 - self.config["alpha_hats"][time])
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(
                key,
                shape=current_x.shape,
            )
            z_scaled = temperature * z
            current_x = current_x + (time > 0) * (
                jnp.sqrt(self.config["betas"][time]) * z_scaled
            )

            if clip_sampler:
                current_x = jnp.clip(
                    current_x, self.config["action_min"], self.config["action_max"]
                )

            return (current_x, rng), ()

        key, rng = jax.random.split(seed)

        if (
            self.config["image_observations"]
            and (
                (
                    isinstance(observations, dict)
                    and len(observations["proprio"].shape) == 2
                )
                or (not isinstance(observations, dict) and len(observations.shape) == 4)
            )
            or not self.config["image_observations"]
            and len(observations.shape) == 2
        ):
            # unbatched input from evaluation
            batch_size = 1
            need_to_unbatch = True
            observations = jax.tree_map(lambda x: x[None], observations)
        else:
            if isinstance(observations, dict):
                batch_size = observations["proprio"].shape[0]
            else:
                batch_size = observations.shape[0]
            need_to_unbatch = False

        input_tuple, () = jax.lax.scan(
            fn,
            (jax.random.normal(key, (batch_size, *self.config["action_dim"])), rng),
            jnp.arange(self.config["diffusion_steps"] - 1, -1, -1),
        )

        for _ in range(self.config["repeat_last_step"]):
            input_tuple, () = fn(input_tuple, 0)

        action_0, rng = input_tuple

        return action_0

    @jax.jit
    def get_debug_metrics(self, batch, seed, gripper_close_val=None):
        actions = self.sample_actions(observations=batch["observations"], seed=seed)

        metrics = {
            "mse": ((actions - batch["actions"]) ** 2).sum((-2, -1)).mean(),
        }

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        # example arrays for model init
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        # agent config
        encoder_def: nn.Module,
        # other shared network config
        image_observations: bool = True,
        use_proprio: bool = False,
        proprioceptive_dims: Optional[int] = None,
        enable_stacking: bool = False,
        score_network_kwargs: dict = {
            "time_dim": 32,
            "num_blocks": 3,
            "dropout_rate": 0.1,
            "hidden_dim": 256,
        },
        # optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # DDPM algorithm train + inference config
        beta_schedule: str = "cosine",
        diffusion_steps: int = 25,
        action_samples: int = 1,
        repeat_last_step: int = 0,
        target_update_rate=0.002,
        dropout_target_networks=True,
        action_min: float = -2.0,
        action_max: float = 2.0,
        **kwargs,
    ):
        assert len(actions.shape) > 1, "Must use action chunking"
        if type(observations) == dict:
            assert (
                len(observations["image"].shape) > 3
            ), "Must use observation histories"
            assert (
                len(observations["proprio"].shape) > 1
            ), "Must use observation histories"
        else:
            if image_observations:
                assert len(observations.shape) > 3, "Must use observation histories"
            else:
                assert len(observations.shape) > 1, "Must use observation histories"

        encoder_def = SACAgent._create_encoder_def(
            encoder_def=encoder_def,
            use_proprio=use_proprio,
            proprioceptive_dims=proprioceptive_dims,
            enable_stacking=enable_stacking,
            goal_conditioned=False,
            early_goal_concat=False,
            shared_goal_encoder=False,
        )

        networks = {
            "actor": ScoreActor(
                encoder_def,
                FourierFeatures(score_network_kwargs["time_dim"], learnable=True),
                MLP(
                    (
                        2 * score_network_kwargs["time_dim"],
                        score_network_kwargs["time_dim"],
                    )
                ),
                MLPResNet(
                    score_network_kwargs["num_blocks"],
                    actions.shape[-2] * actions.shape[-1],
                    dropout_rate=score_network_kwargs["dropout_rate"],
                    use_layer_norm=score_network_kwargs["use_layer_norm"],
                ),
            ),
            "encoder": encoder_def,
        }

        model_def = ModuleDict(networks)

        rng, init_rng = jax.random.split(rng)
        if len(actions.shape) == 3:
            example_time = jnp.zeros((actions.shape[0], 1))
        else:
            example_time = jnp.zeros((1,))
        params = jax.jit(model_def.init)(
            init_rng,
            actor=[observations, actions, example_time],
            encoder=[observations, {"train": False}],
        )["params"]

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "actor": lr_schedule,
        }
        if actor_decay_steps is not None:
            lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=actor_decay_steps,
                end_value=0.0,
            )
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(diffusion_steps))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, diffusion_steps)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(diffusion_steps))

        alphas = 1 - betas
        alpha_hat = jnp.array(
            [jnp.prod(alphas[: i + 1]) for i in range(diffusion_steps)]
        )

        config = flax.core.FrozenDict(
            dict(
                target_update_rate=target_update_rate,
                dropout_target_networks=dropout_target_networks,
                action_dim=actions.shape[-2:],
                action_max=action_max,
                action_min=action_min,
                betas=betas,
                alphas=alphas,
                alpha_hats=alpha_hat,
                diffusion_steps=diffusion_steps,
                action_samples=action_samples,
                repeat_last_step=repeat_last_step,
                image_observations=image_observations,
                **kwargs,
            )
        )
        return cls(state, config, lr_schedules)
