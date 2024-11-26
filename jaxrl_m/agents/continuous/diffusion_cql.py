from dataclasses import field
from functools import partial
from typing import Dict, Optional, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import optax
from jaxrl_m.agents.continuous.cql import ContinuousCQLAgent
from jaxrl_m.agents.continuous.ddpm_bc import DDPMBCAgent
from jaxrl_m.agents.continuous.q_diffusion import (
    QDiffusionState,
    q_diffusion_sample_actions,
    q_diffusion_steps,
)
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.vision.data_augmentations import batched_random_crop
from overrides import overrides


class DiffusionCQLAgent(ContinuousCQLAgent):
    ddpm_agent: DDPMBCAgent = None
    num_ddpm_actions: int = None
    q_diffusion_num_steps: int = None
    q_diffusion_step_size: float = None
    action_space_low: jnp.ndarray = None
    action_space_high: jnp.ndarray = None
    forward_policy_additional_info_metrics: Dict[str, jnp.ndarray] = field(
        default_factory=dict
    )
    action_optimizer_state: Dict[str, optax.OptState] = field(default_factory=dict)

    @overrides
    def _compute_next_actions(self, batch, rng, **kwargs):
        if not self.config.get("bellman_target_uses_base_policy", False):
            return super()._compute_next_actions(batch, rng, **kwargs)

        assert not self.config["cql_max_target_backup"]

        rng, key = jax.random.split(rng)
        next_observations = batch["next_observations"][:, None]
        next_actions = self.ddpm_agent.sample_actions(next_observations, seed=key)
        batch_size = batch["rewards"].shape[0]
        action_dim = batch["actions"].shape[-1]
        chex.assert_shape(next_actions, (batch_size, 1, action_dim))
        next_actions = jnp.squeeze(next_actions, axis=1)
        log_pi = jnp.zeros(batch_size)

        return next_actions, log_pi

    def _get_cql_q_diff(self, *args, **kwargs):
        # The agent is a dataclass
        q_diffusion_num_steps_for_cql = self.config["q_diffusion_num_steps_for_cql"]
        if (
            q_diffusion_num_steps_for_cql >= 0
            and q_diffusion_num_steps_for_cql != self.config["q_diffusion_num_steps"]
        ):
            cql_q_diff, info = self.replace(
                config=self.config.copy(
                    {"q_diffusion_num_steps": q_diffusion_num_steps_for_cql}
                )
            )._get_cql_q_diff(*args, **kwargs)
        else:
            cql_q_diff, info = super()._get_cql_q_diff(*args, **kwargs)

        info.update(self.forward_policy_additional_info_metrics)
        self.forward_policy_additional_info_metrics.clear()
        return cql_q_diff, info

    def forward_policy_with_dataset_actions(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        dataset_actions: jnp.ndarray,
        rng: Optional[PRNGKey],
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Like forward_policy, but instead of using the DDPM to get the actions, it uses the dataset
            actions.
        This should only be used for the critic training.
        """
        assert len(dataset_actions.shape) == 2

        # Add dataset actions l1 norm before q_diffusion_steps to the metrics
        self.forward_policy_additional_info_metrics["dataset_actions_l1_norm"] = (
            jnp.mean(jnp.sum(jnp.abs(dataset_actions), axis=1))
        )

        critic_rng, rng = jax.random.split(rng)
        q_value_before = self.forward_critic(
            observations, dataset_actions, rng=critic_rng, train=False
        )
        q_value_before = q_value_before
        self.forward_policy_additional_info_metrics["dataset_actions_q_values"] = (
            jnp.mean(q_value_before)
        )

        # Propagate the actions through the Q function
        q_diffusion_results: QDiffusionState = q_diffusion_steps(
            observations,
            dataset_actions,
            critic=self,
            max_num_steps=self.config["q_diffusion_num_steps"],
            step_size=self.config["q_diffusion_step_size"],
            optimize_critic_ensemble_min=self.config[
                "q_diffusion_optimize_critic_ensemble_min"
            ],
            use_adam=self.config["q_diffusion_use_adam"],
            adam_kwargs=self.config["q_diffusion_adam_kwargs"].to_dict(),
            adam_state=self.action_optimizer_state["adam_state"],
            action_space_low=self.config["action_space_low"],
            action_space_high=self.config["action_space_high"],
            use_target_critic=self.config["use_target_critic_for_q_diffusion_steps"],
            half_step_size_on_overshooting=self.config[
                "q_diffusion_half_step_size_on_overshooting"
            ],
            overshooting_factor=self.config["q_diffusion_overshooting_factor"],
        )
        q_diffusion_actions = q_diffusion_results.actions
        last_gradient_norm = q_diffusion_results.last_gradient_norm
        self.action_optimizer_state["adam_state"] = q_diffusion_results.opt_state
        assert q_diffusion_actions.shape == dataset_actions.shape

        # Add the last gradient norm, q_diffusion_actions l1 norm, value increase to the metrics
        self.forward_policy_additional_info_metrics["last_gradient_norm"] = (
            last_gradient_norm
        )

        q_values = self.forward_critic(
            observations, q_diffusion_actions, rng=critic_rng, train=False
        )

        self.forward_policy_additional_info_metrics[
            "(dataset_actions)q_diffusion_actions_l1_norm"
        ] = jnp.mean(jnp.sum(jnp.abs(q_diffusion_actions), axis=1))
        self.forward_policy_additional_info_metrics["(dataset_actions)q_values"] = (
            jnp.mean(q_values)
        )
        self.forward_policy_additional_info_metrics[
            "(dataset_actions)value_increase"
        ] = jnp.mean(q_values - q_value_before)
        self.forward_policy_additional_info_metrics[
            "(dataset_actions) index of max value action"
        ] = q_diffusion_results.action_with_max_value_index

        return distrax.Independent(
            distrax.Deterministic(q_diffusion_actions), reinterpreted_batch_ndims=1
        )

    def forward_policy(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: Optional[PRNGKey],
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
        force_gaussian_policy: bool = False,
        dataset_actions: Optional[jnp.ndarray] = None,
    ) -> Tuple[distrax.Distribution, optax.OptState]:
        if force_gaussian_policy or (
            self.config["use_gaussian_policy_for_critic_training"]
            and grad_params is not None
        ):
            assert dataset_actions is None
            # If grad_params is not none, it means we are training the policy
            return super().forward_policy(
                observations, rng=rng, grad_params=grad_params, train=train
            )
        if dataset_actions is not None:
            return self.forward_policy_with_dataset_actions(
                observations,
                dataset_actions,
                rng=rng,
                grad_params=grad_params,
                train=train,
            )

        action_distribution, info_metrics, new_adam_state = q_diffusion_sample_actions(
            observations,
            ddpm_agent=self.ddpm_agent,
            critic_agent=self,
            num_ddpm_actions=self.config["num_ddpm_actions"],
            num_actions_to_keep_for_q_diffusion=self.config[
                "num_actions_to_keep_for_q_diffusion"
            ],
            q_diffusion_num_steps=self.config["q_diffusion_num_steps"],
            q_diffusion_step_size=self.config["q_diffusion_step_size"],
            q_diffusion_optimize_critic_ensemble_min=self.config[
                "q_diffusion_optimize_critic_ensemble_min"
            ],
            use_adam=self.config["q_diffusion_use_adam"],
            rng=rng,
            action_space_low=self.config["action_space_low"],
            action_space_high=self.config["action_space_high"],
            q_diffusion_adam_kwargs=self.config["q_diffusion_adam_kwargs"].to_dict(),
            adam_state=self.action_optimizer_state["adam_state"],
            argmax=self.config["always_use_argmax_for_q_diffusion"],
            half_step_size_on_overshooting=self.config[
                "q_diffusion_half_step_size_on_overshooting"
            ],
            overshooting_factor=self.config["q_diffusion_overshooting_factor"],
        )

        self.forward_policy_additional_info_metrics.update(info_metrics)
        self.action_optimizer_state["adam_state"] = new_adam_state
        return action_distribution

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        if self.config["train_gaussian_policy"]:
            return super().policy_loss_fn(batch, params, rng)
        return 0.0, {}

    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        observations: Data,
        goals: Optional[Data] = None,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        action_optimizer_state: Optional[optax.OptState] = None,
        **kwargs,
    ) -> jnp.ndarray:
        if isinstance(observations, dict):
            key = "proprio" if "proprio" in observations else "state"
            if len(observations[key].shape) == 1:
                need_to_unbatch = True
                observations = jax.tree_map(
                    lambda x: jnp.expand_dims(x, 0), observations
                )
            else:
                need_to_unbatch = False
            assert len(observations[key].shape) == 2
        else:
            if len(observations.shape) == 1:
                need_to_unbatch = True
                observations = jnp.expand_dims(observations, 0)
            else:
                need_to_unbatch = False
            assert len(observations.shape) == 2, observations.shape
        self.action_optimizer_state["adam_state"] = action_optimizer_state
        if self.config["goal_conditioned"]:
            assert goals is not None
            obs = (observations, goals)
        else:
            obs = observations
        dist = self.forward_policy(obs, rng=seed, train=False)
        if not argmax:
            action = dist.sample(seed=seed)
        else:
            if self.config["always_use_argmax_for_q_diffusion"]:
                action = dist.mode()
            else:
                # MixtureSameFamily doesn't implement mode, so first get most likely component
                # and then get the mode of that component
                most_likely_component = dist.mixture_distribution.probs.argmax(axis=-1)
                batch_size = most_likely_component.shape[0]
                most_likely_actions = dist.components_distribution.distribution.mode()[
                    jnp.arange(batch_size), most_likely_component
                ]
                action = most_likely_actions

        if need_to_unbatch:
            assert len(action.shape) == 2 and action.shape[0] == 1
            action = action[0]

        return action

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        pmap_axis: str = None,
        networks_to_update: set = frozenset({"critic"}),
        action_optimizer_state: Optional[optax.OptState] = None,
        **kwargs,
    ):
        """Add action_optimizer_state to self, return it at the end.
        Otherwise, it's CQL + SAC update."""
        self.action_optimizer_state["adam_state"] = action_optimizer_state

        # Optionally apply DRQ augmentation
        if self.config.get("drq_padding", 0) > 0:
            assert not self.config["goal_conditioned"]
            rng, key = jax.random.split(self.state.rng)
            # Use same key for both observations and next_observations
            batch["observations"]["image"] = batched_random_crop(
                batch["observations"]["image"],
                key,
                padding=self.config["drq_padding"],
                num_batch_dims=1,
            )
            batch["next_observations"]["image"] = batched_random_crop(
                batch["next_observations"]["image"],
                key,
                padding=self.config["drq_padding"],
                num_batch_dims=1,
            )

        new_self, info = super().update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=networks_to_update,
            **kwargs,
        )

        action_optimizer_state = self.action_optimizer_state["adam_state"]
        return (
            new_self,
            info,
            action_optimizer_state,
        )

    def update_high_utd(
        self, *args, networks_to_update: set = frozenset({"critic"}), **kwargs
    ):
        assert not self.config["q_diffusion_use_adam"]
        action_optimizer_state = kwargs.pop("action_optimizer_state")
        self.action_optimizer_state["adam_state"] = action_optimizer_state
        agent, infos = super().update_high_utd(
            *args, networks_to_update=networks_to_update, **kwargs
        )
        return agent, infos, action_optimizer_state

    @classmethod
    def create(cls, *args, **kwargs):
        if kwargs["use_gaussian_policy_for_critic_training"]:
            assert kwargs["train_gaussian_policy"]
        ddpm_agent = kwargs.pop("ddpm_agent", 42.0)
        agent = super(DiffusionCQLAgent, cls).create(*args, **kwargs)
        agent = agent.replace(
            ddpm_agent=ddpm_agent,
        )
        return agent
