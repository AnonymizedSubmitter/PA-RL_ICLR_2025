from collections import namedtuple
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jaxrl_m.common.common import (
    JaxRLTrainState,
    ModuleDict,
    make_dict_kwargs_hashable_decorator,
)
from jaxrl_m.common.optimizers import make_optimizer

QDiffusionState = namedtuple(
    "QDiffusionState",
    [
        "num_gradient_steps_taken",
        "actions",
        "last_gradient_norm",
        "opt_state",
        "action_with_max_value",
        "max_value",
        "action_with_max_value_index",
        "step_sizes",  # for half_step_size_on_overshooting, store individual step sizes
        "number_of_overshoots",
    ],
)


@make_dict_kwargs_hashable_decorator
@partial(
    jax.jit,
    static_argnames=(
        "max_num_steps",
        "optimize_critic_ensemble_min",
        "use_target_critic",
        "use_adam",
        "adam_kwargs",
        "keep_action_with_max_value",
        "half_step_size_on_overshooting",
    ),
)
def q_diffusion_steps(
    observations: Union[
        jnp.ndarray, Dict[str, jnp.ndarray]
    ],  # dict will not be made hashable bc
    # it's not kwarg
    ddpm_actions: jnp.ndarray,
    critic: flax.struct.PyTreeNode,
    max_num_steps: int,
    step_size: float,
    stop_when_update_norm_smaller_than: float = 0,
    optimize_critic_ensemble_min: bool = True,
    action_space_low: Optional[jnp.ndarray] = None,
    action_space_high: Optional[jnp.ndarray] = None,
    use_target_critic: bool = False,
    use_adam: bool = False,
    adam_kwargs: Optional[Dict[str, Any]] = None,
    adam_state: Optional[optax.OptState] = None,
    keep_action_with_max_value: bool = True,
    half_step_size_on_overshooting: bool = False,
    overshooting_factor: float = 0.5,
) -> QDiffusionState:
    """Takes num_steps steps maximizing the critic."""
    if isinstance(observations, tuple) and len(observations) == 2:
        # It's goal conditioend
        observations, goals = observations
    else:
        goals = None
    if type(observations) is dict:
        if set(observations.keys()) == {"image", "proprio"}:
            assert len(observations["image"].shape) == 4
            assert len(observations["proprio"].shape) == 2
        else:
            assert set(observations.keys()) == {"encoding", "proprio"}
            assert len(observations["encoding"].shape) == 2
            assert len(observations["proprio"].shape) == 2
        batch_size = observations["proprio"].shape[0]
    else:
        assert len(observations.shape) == 2
        batch_size = observations.shape[0]
    if goals is not None:
        observations = (observations, goals)
    assert len(ddpm_actions.shape) == 2
    if half_step_size_on_overshooting:
        assert keep_action_with_max_value

    # action is first to differentiate wrt it
    def critic_fn(action, obs):
        critic_params = (
            critic.state.params if not use_target_critic else critic.state.target_params
        )
        critic_values = critic.state.apply_fn(
            {
                "params": critic_params,
            },
            obs,
            action,
            name="critic",
        )
        if type(critic_values) is tuple:
            # It's a distributional critic
            critic_values = critic_values[0]
        chex.assert_shape(
            critic_values, (critic.config["critic_ensemble_size"], action.shape[0])
        )
        if optimize_critic_ensemble_min:
            return critic_values.min(axis=0).sum(), critic_values.min(axis=0)
        return critic_values.mean(axis=0).sum(), critic_values.mean(axis=0)

    def get_critic_gradient_wrt_actions(obs: jnp.ndarray, action: jnp.ndarray):
        return jax.grad(critic_fn, has_aux=True)(action, obs)

    if use_adam:
        assert not half_step_size_on_overshooting
        optimizer = make_optimizer(learning_rate=step_size, **adam_kwargs)
        if adam_state is None:
            opt_state = optimizer.init(ddpm_actions)
        else:
            opt_state = adam_state
            nu = optax.tree_utils.tree_get(opt_state, "nu")
            chex.assert_shape(nu, (ddpm_actions.shape[-1],))
            # nu = nu[(None,) * len(batch_dimensions)]
            # nu = jnp.broadcast_to(nu, ddpm_actions.shape)
            nu = jnp.broadcast_to(nu[None], ddpm_actions.shape)
            opt_state = optax.tree_utils.tree_set(
                opt_state, mu=jnp.zeros_like(nu), nu=nu
            )

    else:
        opt_state = None

    def condition(state: QDiffusionState):
        return jnp.logical_and(
            state.num_gradient_steps_taken < max_num_steps,
            state.last_gradient_norm > stop_when_update_norm_smaller_than,
        )

    def body(state):

        # Update the action with the maximum value, each action in the batch independently
        if keep_action_with_max_value:
            current_values = critic_fn(state.actions, observations)[1]
            max_value = jnp.maximum(state.max_value, current_values)
            action_with_max_value = jnp.where(
                jnp.broadcast_to(
                    current_values[..., None], state.action_with_max_value.shape
                )
                == max_value[..., None],
                state.actions,
                state.action_with_max_value,
            )
            action_with_max_value_index = jnp.where(
                current_values == max_value,
                state.num_gradient_steps_taken,
                state.action_with_max_value_index,
            )
            # assert action_with_max_value_index.shape == batch_dimensions
            assert action_with_max_value_index.shape == (batch_size,)

            # Implement half_step_size_on_overshooting logic
            if half_step_size_on_overshooting:
                overshot = current_values < state.max_value
                chex.assert_shape(overshot, (batch_size,))
                new_number_of_overshoots = state.number_of_overshoots + overshot
                new_step_sizes = jnp.where(
                    overshot, state.step_sizes * overshooting_factor, state.step_sizes
                )
                # Revert the actions that overshot
                actions = jnp.where(
                    overshot[..., None], state.action_with_max_value, state.actions
                )
            else:
                new_step_sizes = state.step_sizes
                actions = state.actions
                new_number_of_overshoots = state.number_of_overshoots
        else:
            new_step_sizes = state.step_sizes
            actions = state.actions

        critic_gradient_wrt_actions, current_values = get_critic_gradient_wrt_actions(
            observations, actions
        )
        chex.assert_shape(current_values, (batch_size,))
        chex.assert_shape(
            critic_gradient_wrt_actions, (batch_size, ddpm_actions.shape[1])
        )
        chex.assert_shape(new_step_sizes, (batch_size,))

        if use_adam:
            updates, opt_state = optimizer.update(
                critic_gradient_wrt_actions, state.opt_state, actions
            )
            actions = optax.apply_updates(actions, updates)
        else:
            actions = actions + new_step_sizes[..., None] * critic_gradient_wrt_actions
            opt_state = state.opt_state
        # Clip actions to the action space
        if action_space_low is not None and action_space_high is not None:
            chex.assert_shape(action_space_low, ddpm_actions.shape[-1:])
            chex.assert_shape(action_space_high, ddpm_actions.shape[-1:])
            actions = jnp.clip(actions, action_space_low, action_space_high)

        gradient_norm = jnp.linalg.norm(critic_gradient_wrt_actions, axis=-1).mean()
        return QDiffusionState(
            num_gradient_steps_taken=state.num_gradient_steps_taken + 1,
            actions=actions,
            last_gradient_norm=gradient_norm,
            opt_state=opt_state,
            action_with_max_value=action_with_max_value,
            max_value=max_value,
            action_with_max_value_index=action_with_max_value_index,
            step_sizes=new_step_sizes,
            number_of_overshoots=new_number_of_overshoots,
        )

    initial_state = QDiffusionState(
        num_gradient_steps_taken=0,
        actions=ddpm_actions,
        last_gradient_norm=jnp.inf,
        opt_state=opt_state,
        action_with_max_value=ddpm_actions,
        max_value=jnp.broadcast_to(-jnp.inf, (batch_size,)),
        action_with_max_value_index=jnp.zeros(batch_size, dtype=jnp.int32),
        step_sizes=jnp.full((batch_size,), step_size),
        number_of_overshoots=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    final_state = jax.lax.while_loop(
        condition,
        body,
        initial_state,
    )

    # Update the action with the maximum value one last time
    if keep_action_with_max_value:
        current_values = critic_fn(final_state.actions, observations)[1]
        max_value = jnp.maximum(final_state.max_value, current_values)
        action_with_max_value = jnp.where(
            jnp.broadcast_to(
                current_values[..., None], final_state.action_with_max_value.shape
            )
            == max_value[..., None],
            final_state.actions,
            final_state.action_with_max_value,
        )
        action_with_max_value_index = jnp.where(
            current_values == max_value,
            final_state.num_gradient_steps_taken,
            final_state.action_with_max_value_index,
        )

    # Only keep the rms prop statistic
    if use_adam:
        new_mu = jnp.zeros_like(ddpm_actions)
        # average rms prop nu across batch
        nu = optax.tree_utils.tree_get(final_state.opt_state, "nu")
        chex.assert_shape(nu, ddpm_actions.shape)
        # new_nu = jnp.mean(nu, axis=jnp.arange(len(batch_dimensions)))
        new_nu = jnp.mean(nu, axis=0)
        opt_state = optax.tree_utils.tree_set(
            final_state.opt_state, mu=new_mu, nu=new_nu
        )

    return final_state._replace(
        actions=(
            action_with_max_value if keep_action_with_max_value else final_state.actions
        ),
        action_with_max_value=action_with_max_value,
        action_with_max_value_index=action_with_max_value_index,
        opt_state=opt_state,
    )


def test_q_diffusion_steps():
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    observations = jax.random.normal(
        key,
        shape=(
            3,
            3,
        ),
    )  # (batch_size, obs_dim)
    ddpm_actions = jax.random.normal(
        key,
        shape=(
            3,
            3,
        ),
    )  # (batch_size, act_dim)

    class TestCriticModule(nn.Module):
        @nn.compact
        def __call__(self, obs, act):
            assert len(obs.shape) == 2
            return (
                jnp.array(
                    [
                        2 * act[:, 0] * obs[:, 0]
                        - act[:, 1] * obs[:, 1]
                        + act[:, 2] * obs[:, 2]
                    ]
                )
                .reshape(1, obs.shape[0])
                .repeat(2, axis=0)
                .squeeze()
            )

    model_def = ModuleDict({"critic": TestCriticModule()})
    state = JaxRLTrainState.create(
        apply_fn=model_def.apply,
        params={},
        txs=None,
        target_params={},
        rng=rng,
    )

    class TestAgent(flax.struct.PyTreeNode):
        state: JaxRLTrainState
        config = {"critic_ensemble_size": 2}

    critic = TestAgent(state)
    num_steps = 10
    step_size = 10

    q_diffusion_results = q_diffusion_steps(
        observations, ddpm_actions, critic, num_steps, step_size
    )
    expected_actions = (
        ddpm_actions
        + num_steps
        * step_size
        * jnp.array([2, -1, 1]).reshape(1, -1).repeat(3, axis=0)
        * observations
    )
    assert jnp.all(jnp.isclose(q_diffusion_results.actions, expected_actions)), (
        q_diffusion_results.actions,
        expected_actions,
    )
    expected_last_gradient_norm = jnp.linalg.norm(
        jnp.array([2, -1, 1]).reshape(1, -1).repeat(3, axis=0) * observations,
        axis=1,
    ).mean()
    assert jnp.isclose(
        q_diffusion_results.last_gradient_norm, expected_last_gradient_norm
    ), (
        q_diffusion_results.last_gradient_norm,
        expected_last_gradient_norm,
    )
    assert q_diffusion_results.num_gradient_steps_taken == num_steps


@make_dict_kwargs_hashable_decorator
@partial(
    jax.jit,
    static_argnames=(
        "num_ddpm_actions",
        "num_actions_to_keep_for_q_diffusion",
        "q_diffusion_num_steps",
        "q_diffusion_optimize_critic_ensemble_min",
        "use_adam",
        "q_diffusion_adam_kwargs",
        "argmax",
        "half_step_size_on_overshooting",
    ),
)
def q_diffusion_sample_actions(
    observations: jnp.ndarray,
    ddpm_agent: flax.struct.PyTreeNode,
    critic_agent: flax.struct.PyTreeNode,
    num_ddpm_actions: int,
    num_actions_to_keep_for_q_diffusion: int,
    q_diffusion_num_steps: int,
    q_diffusion_step_size: float,
    q_diffusion_optimize_critic_ensemble_min: bool,
    use_adam: bool,
    rng: jax.random.PRNGKey,
    action_space_low: jnp.ndarray,
    action_space_high: jnp.ndarray,
    q_diffusion_adam_kwargs: Optional[Dict[str, Any]] = None,
    adam_state: Optional[optax.OptState] = None,
    argmax: bool = False,
    half_step_size_on_overshooting: bool = False,
    overshooting_factor: float = 0.5,
    dataset_actions_to_consider: Optional[jnp.ndarray] = None,
) -> Tuple[distrax.Distribution, Dict[str, jnp.ndarray], optax.OptState]:
    info_metrics = {}
    if type(observations) is tuple and len(observations) == 2:
        # It's goal conditioned
        observations, goals = observations
    else:
        goals = None
    ddpm_actions = None
    if isinstance(observations, dict):
        if "encoding" in observations:
            assert (
                False
            ), "encoding should not be calculated here, because critic and ddpm use different encodings"
        elif "image" in observations and "proprio" in observations:
            if len(observations["image"].shape) == 3:
                observations["image"] = observations["image"][None]
                observations["proprio"] = observations["proprio"][None]
                if goals is not None and len(goals["image"].shape) == 3:
                    goals["image"] = goals["image"][None]
            batch_size = observations["image"].shape[0]
            if "ddpm_actions" in observations:
                ddpm_actions = observations["ddpm_actions"]
                if len(ddpm_actions.shape) == 1:
                    ddpm_actions = ddpm_actions[None]
        else:
            assert "state" in observations and "ddpm_actions" in observations
            if len(observations["state"].shape) == 1:
                observations["state"] = observations["state"][None]
                observations["ddpm_actions"] = observations["ddpm_actions"][None]
            ddpm_actions = observations["ddpm_actions"]
            observations = observations["state"]
            batch_size = observations.shape[0]
    else:
        if len(observations.shape) == 1:
            observations = observations[None]
        batch_size = observations.shape[0]

    if isinstance(observations, dict) and "proprio" in observations:
        proprio = observations["proprio"]
        if goals is not None:
            observations = (observations, goals)

        if ddpm_actions is None:

            # Calculate encodings for ddpm_observations
            ddpm_observations = {
                "encoding": ddpm_agent.state.apply_fn(
                    {"params": ddpm_agent.state.params},
                    observations,
                    train=False,
                    name="encoder",
                ),
                "proprio": proprio,
            }

            ddpm_repeated_observations = jax.tree_map(
                lambda x: x[:, None]
                .repeat(num_ddpm_actions, axis=1)
                .reshape(batch_size * num_ddpm_actions, *x.shape[1:]),
                ddpm_observations,
            )

        critic_observations = {
            "encoding": critic_agent.state.apply_fn(
                {"params": critic_agent.state.params},
                observations,
                train=False,
                name="critic_encoder",
            ),
            "proprio": proprio,
        }

        repeated_critic_observations = jax.tree_map(
            lambda x: x[:, None]
            .repeat(num_ddpm_actions, axis=1)
            .reshape(batch_size * num_ddpm_actions, *x.shape[1:]),
            critic_observations,
        )
    else:
        ddpm_repeated_observations = (
            observations[:, None]
            .repeat(num_ddpm_actions, axis=1)
            .reshape(batch_size * num_ddpm_actions, observations.shape[-1])
        )
        repeated_critic_observations = ddpm_repeated_observations

    rng, ddpm_key = jax.random.split(rng)
    if ddpm_actions is None:
        ddpm_repeated_observations = jax.tree_map(
            lambda x: x[:, None], ddpm_repeated_observations
        )

        ddpm_actions = ddpm_agent.sample_actions(
            ddpm_repeated_observations, goals=goals, seed=ddpm_key
        )
    else:
        ddpm_actions = ddpm_actions.reshape(-1, 1, ddpm_actions.shape[-1])

    chex.assert_shape(
        ddpm_actions,
        (
            batch_size * num_ddpm_actions,
            1,
            ddpm_actions.shape[-1],
        ),
    )
    ddpm_actions = ddpm_actions.reshape(
        batch_size * num_ddpm_actions, ddpm_actions.shape[-1]
    )

    # Add the DDPM actions l1 norm to the metrics
    info_metrics["ddpm_actions_l1_norm"] = jnp.mean(
        jnp.sum(jnp.abs(ddpm_actions), axis=1)
    )

    rng, key = jax.random.split(rng)
    q_values_before = (
        critic_agent.forward_critic(
            repeated_critic_observations,
            ddpm_actions,
            rng=key,
            train=False,
        )
        .mean(axis=0)
        .reshape(batch_size * num_ddpm_actions)
    )

    if num_actions_to_keep_for_q_diffusion < num_ddpm_actions:
        # Keep only a subset of the actions, ranked by the critic

        q_values = q_values_before.reshape(batch_size, num_ddpm_actions)
        top_k_indices = jnp.argsort(q_values, axis=-1)[
            :, -num_actions_to_keep_for_q_diffusion:
        ]
        ddpm_actions = ddpm_actions.reshape(
            batch_size, num_ddpm_actions, ddpm_actions.shape[-1]
        )
        repeated_critic_observations = jax.tree_map(
            lambda x: x.reshape(
                batch_size,
                num_ddpm_actions,
                *x.shape[1:],
            ),
            repeated_critic_observations,
        )
        ddpm_actions = jnp.take_along_axis(
            ddpm_actions, top_k_indices[:, :, None], axis=1
        )
        # Best actions will be last in the list

        # top_k_indices needs to have the same number of dimensions as repeated_observations
        # for every key. Thus, for every key, we maintain the 2 existing dimensions, and add
        # None for every other dimension.
        repeated_critic_observations = jax.tree_map(
            lambda x: jnp.take_along_axis(
                x,
                top_k_indices[
                    (
                        slice(None),  # batch dimension
                        slice(None),  # num_actions_to_keep_for_q_diffusion dimension
                        *(None,)
                        * (
                            len(x.shape) - len(top_k_indices.shape)
                        ),  # x's additional dimensions
                    )
                ],
                axis=1,
            ),
            repeated_critic_observations,
        )
        ddpm_actions = ddpm_actions.reshape(
            batch_size * num_actions_to_keep_for_q_diffusion,
            ddpm_actions.shape[-1],
        )
        repeated_critic_observations = jax.tree_map(
            lambda x: x.reshape(
                batch_size * num_actions_to_keep_for_q_diffusion,
                *x.shape[2:],
            ),
            repeated_critic_observations,
        )
        q_values_before = jnp.take_along_axis(
            q_values_before.reshape(batch_size, num_ddpm_actions),
            top_k_indices,
            axis=1,
        ).reshape(batch_size * num_actions_to_keep_for_q_diffusion)

    chex.assert_shape(
        ddpm_actions,
        (
            batch_size * num_actions_to_keep_for_q_diffusion,
            ddpm_actions.shape[-1],
        ),
    )
    info_metrics["ddpm_actions_q_values"] = q_values_before
    info_metrics["ddpm_actions_q_values_mean"] = q_values_before.mean()

    if dataset_actions_to_consider is not None:
        # add dataset actions to the ddpm actions
        chex.assert_shape(
            dataset_actions_to_consider, (batch_size, ddpm_actions.shape[-1])
        )
        ddpm_actions = jnp.concatenate(
            [
                ddpm_actions.reshape(
                    batch_size,
                    num_actions_to_keep_for_q_diffusion,
                    ddpm_actions.shape[-1],
                )[
                    :, 1:, :
                ],  # skip the worst action
                dataset_actions_to_consider[:, None, :],
            ],
            axis=1,
        ).reshape(
            batch_size * (num_actions_to_keep_for_q_diffusion),
            ddpm_actions.shape[-1],
        )

    # Propagate the actions through the Q function
    q_diffusion_results: QDiffusionState = q_diffusion_steps(
        repeated_critic_observations,
        ddpm_actions,
        critic=critic_agent,
        max_num_steps=q_diffusion_num_steps,
        step_size=q_diffusion_step_size,
        optimize_critic_ensemble_min=q_diffusion_optimize_critic_ensemble_min,
        use_adam=use_adam,
        adam_kwargs=q_diffusion_adam_kwargs,
        adam_state=adam_state,
        action_space_low=action_space_low,
        action_space_high=action_space_high,
        use_target_critic=False,
        half_step_size_on_overshooting=half_step_size_on_overshooting,
        overshooting_factor=overshooting_factor,
    )
    q_diffusion_actions = q_diffusion_results.actions
    last_gradient_norm = q_diffusion_results.last_gradient_norm
    new_adam_state = q_diffusion_results.opt_state

    assert q_diffusion_actions.shape == ddpm_actions.shape, q_diffusion_actions.shape

    # Add the last gradient norm and q_diffusion_actions l1 norm to the metrics
    info_metrics["last_gradient_norm"] = last_gradient_norm
    info_metrics["q_diffusion_actions_l1_norm"] = jnp.mean(
        jnp.sum(jnp.abs(q_diffusion_actions), axis=1)
    )
    if half_step_size_on_overshooting:
        # Log average number of overshoots
        info_metrics["average_num_overshoots"] = jnp.mean(
            q_diffusion_results.number_of_overshoots
        )
        info_metrics["max_num_overshoots"] = jnp.max(
            q_diffusion_results.number_of_overshoots
        )
        info_metrics["min_num_overshoots"] = jnp.min(
            q_diffusion_results.number_of_overshoots
        )

    if use_adam:
        adam_nu = optax.tree_utils.tree_get(new_adam_state, "nu")
        info_metrics["adam_nu"] = {
            f"adam_nu_{action_dim}": nu_dim for action_dim, nu_dim in enumerate(adam_nu)
        }

    # Compute the Q values
    rng, key = jax.random.split(rng)
    q_values = critic_agent.forward_critic(
        repeated_critic_observations, q_diffusion_actions, rng=key, train=False
    ).mean(axis=0)
    info_metrics["q_values_after_steps"] = q_values.mean()
    info_metrics["total_value_increase"] = jnp.mean(q_values - q_values_before)
    info_metrics["l1(q_diff_actions, ddpm_actions)"] = jnp.mean(
        jnp.sum(jnp.abs(q_diffusion_actions - ddpm_actions), axis=-1)
    )
    info_metrics["index of max value action"] = (
        q_diffusion_results.action_with_max_value_index
    )
    logits = q_values.reshape(batch_size, num_actions_to_keep_for_q_diffusion)

    if argmax:
        # Return Deterministic argmax action
        q_diffusion_actions = q_diffusion_actions.reshape(
            batch_size,
            num_actions_to_keep_for_q_diffusion,
            q_diffusion_actions.shape[-1],
        )
        q_diffusion_actions = q_diffusion_actions[
            jnp.arange(batch_size)[:], jnp.argmax(logits, axis=-1)[:]
        ]
        return (
            distrax.Independent(
                distrax.Deterministic(loc=q_diffusion_actions),
                reinterpreted_batch_ndims=1,
            ),
            info_metrics,
            new_adam_state,
        )

    # Make a batch of Mixtures of deterministic values
    mixture_distribution = distrax.Categorical(logits=logits)
    info_metrics["categorical_entropy"] = mixture_distribution.entropy().mean()
    info_metrics["categorical_logits_min"] = jnp.mean(logits.min(axis=1))
    info_metrics["categorical_logits_max"] = jnp.mean(logits.max(axis=1))
    info_metrics["categorical_logits_mean"] = jnp.mean(logits)
    info_metrics["categorical_logits_std"] = jnp.mean(logits.std(axis=1))
    q_diffusion_actions = q_diffusion_actions.reshape(
        batch_size,
        num_actions_to_keep_for_q_diffusion,
        q_diffusion_actions.shape[-1],
    )

    components_distribution = distrax.Independent(
        distrax.Deterministic(q_diffusion_actions), reinterpreted_batch_ndims=1
    )
    return (
        distrax.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            components_distribution=components_distribution,
        ),
        info_metrics,
        new_adam_state,
    )
