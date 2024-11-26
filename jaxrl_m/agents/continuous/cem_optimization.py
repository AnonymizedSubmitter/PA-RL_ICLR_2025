from collections import namedtuple
from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import orbax
from jaxrl_m.agents.continuous.ddpm_bc import DDPMBCAgent
from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.typing import PRNGKey

CEMState = namedtuple(
    "CEMState",
    [
        "mean",
        "std",
        "winners",
        "scores",
        "random_seed",
    ],
)


class CrossEntropyMethodOptimizationAgent:
    base_policy: DDPMBCAgent = None
    action_space_low: jnp.ndarray = None
    action_space_high: jnp.ndarray = None
    num_iterations: int = 100
    num_samples: int = 100
    num_winners: int = 10

    def sample_actions(self, *args, **kwargs):
        """Use CEM Optimization to get the best actions."""
        actions = self._sample_actions(*args, **kwargs)
        return actions

    @partial(jax.jit, static_argnames=("self"))
    def _sample_actions(
        self,
        observations: jnp.ndarray,
        critic_agent: SACAgent,
        *,
        seed: PRNGKey,
        **kwargs,
    ) -> jnp.ndarray:
        need_to_unbatch = False
        if isinstance(observations, dict):
            observation_key = "state" if "state" in observations else "proprio"
            if observations[observation_key].ndim == 1:
                need_to_unbatch = True
                observations = jax.tree_map(lambda x: x[None], observations)

            batch_size = observations[observation_key].shape[0]
        else:
            if observations.ndim == 1:
                need_to_unbatch = True
                observations = observations[None]
            batch_size = observations.shape[0]
        action_dim = self.action_space_low.shape[0]

        def cem_step(_, state: CEMState, manual_samples=None) -> CEMState:
            if manual_samples is not None:
                samples = manual_samples
                new_seed = state.random_seed
            else:
                random_seed, new_seed = jax.random.split(state.random_seed)
                samples = state.mean + state.std * jax.random.normal(
                    random_seed, (self.num_samples,) + state.mean.shape
                )
                chex.assert_shape(
                    samples,
                    (
                        self.num_samples,
                        batch_size,
                        action_dim,
                    ),
                )
                samples = samples.transpose((1, 0, 2))
                chex.assert_shape(samples, (batch_size, self.num_samples, action_dim))

                # Use tanh and then multiply by the action space to get the actions.
                samples = (
                    jnp.tanh(samples)
                    * (self.action_space_high - self.action_space_low)
                    / 2
                )
            chex.assert_shape(samples, (batch_size, self.num_samples, action_dim))
            critic_key, new_seed = jax.random.split(new_seed)
            scores = critic_agent.forward_critic(
                observations,
                samples,
                rng=critic_key,
                train=False,
            ).mean(axis=0)
            chex.assert_shape(scores, (batch_size, self.num_samples))
            winners_indices = jnp.argsort(scores, axis=1)[:, -self.num_winners :]
            chex.assert_shape(winners_indices, (batch_size, self.num_winners))
            winners = jnp.take_along_axis(samples, winners_indices[:, :, None], axis=1)
            chex.assert_shape(winners, (batch_size, self.num_winners, action_dim))
            winners_score = jnp.take_along_axis(scores, winners_indices, axis=1)
            mean = jnp.mean(winners, axis=1)
            std = jnp.std(winners, axis=1)
            chex.assert_shape(mean, (batch_size, action_dim))

            return CEMState(
                mean=mean,
                std=std,
                winners=winners,
                scores=winners_score,
                random_seed=new_seed,
            )

        if self.base_policy is not None:
            if isinstance(observations, dict) and "proprio" in observations:
                # calculate embeddings before repeating
                observations = {
                    "encoding": self.base_policy.state.apply_fn(
                        {"params": self.base_policy.state.params},
                        observations,
                        train=False,
                        name="encoder",
                    ),
                    "proprio": observations["proprio"],
                }
            repeated_observations = jax.tree_map(
                lambda x: jnp.repeat(x[:, None], self.num_samples, axis=1).reshape(
                    batch_size * self.num_samples, 1, *x.shape[1:]
                ),
                observations,
            )
            sample_seed, seed = jax.random.split(seed)
            manual_samples = self.base_policy.sample_actions(
                observations=repeated_observations, seed=sample_seed
            )
            chex.assert_shape(
                manual_samples, (batch_size * self.num_samples, 1, action_dim)
            )
            manual_samples = manual_samples.squeeze(axis=1).reshape(
                batch_size, self.num_samples, action_dim
            )
            key, seed = jax.random.split(seed)
            initial_state = CEMState(
                mean=jnp.zeros((batch_size, action_dim)),
                std=jnp.ones((batch_size, action_dim)),
                winners=jnp.zeros((batch_size, self.num_winners, action_dim)),
                scores=jnp.zeros((batch_size, self.num_winners)),
                random_seed=key,
            )
            initial_state = cem_step(None, initial_state, manual_samples=manual_samples)
        else:
            initial_state = CEMState(
                mean=jnp.zeros((batch_size, action_dim)),
                std=jnp.ones((batch_size, action_dim)),
                winners=jnp.zeros((batch_size, self.num_winners, action_dim)),
                scores=jnp.zeros((batch_size, self.num_winners)),
                random_seed=seed,
            )

        final_state = jax.lax.fori_loop(0, self.num_iterations, cem_step, initial_state)

        # Return the sample with the highest score.
        best_sample_indices = jnp.argmax(final_state.scores, axis=1)
        chex.assert_shape(best_sample_indices, (batch_size,))
        best_sample = jnp.take_along_axis(
            final_state.winners, best_sample_indices[:, None, None], axis=1
        ).squeeze(axis=1)
        chex.assert_shape(best_sample, (batch_size, action_dim))
        if need_to_unbatch:
            best_sample = best_sample[0]
        return best_sample

    def update(self, *args, **kwargs):
        # The CEM policy does not need to be updated.
        pass

    def __init__(
        self,
        action_space_low: jnp.ndarray,
        action_space_high: jnp.ndarray,
        num_iterations: int = 2,
        num_samples: int = 64,
        num_winners: int = 6,
        ddpm_checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        assert (
            action_space_low.shape == action_space_high.shape
            and len(action_space_low.shape) == 1
        )
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.num_winners = num_winners
        if ddpm_checkpoint_path is not None and ddpm_checkpoint_path != "":
            # Create base policy
            self.base_policy = DDPMBCAgent.create(
                rng=kwargs["rng"],
                observations=kwargs["observations"],
                goals=None,
                actions=kwargs["actions"],
                encoder_def=kwargs["encoder_def"],
                action_min=kwargs["action_min"],
                action_max=kwargs["action_max"],
                **kwargs["ddpm_agent_kwargs"],
            )
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            self.base_policy = orbax_checkpointer.restore(
                ddpm_checkpoint_path, item=self.base_policy
            )
            devices = jax.local_devices()
            sharding = jax.sharding.PositionalSharding(devices)

            self.base_policy = jax.device_put(
                jax.tree_map(jnp.array, self.base_policy), sharding.replicate()
            )
            print(f"[CEM] Loaded DDPMBCAgent from {ddpm_checkpoint_path}")
