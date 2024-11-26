from functools import partial
from typing import Optional

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from jaxrl_m.agents.continuous.ddpm_bc import DDPMBCAgent
from jaxrl_m.agents.continuous.iql import IQLAgent
from jaxrl_m.agents.continuous.q_diffusion import q_diffusion_sample_actions
from jaxrl_m.common.typing import Data, Params, PRNGKey


class DiffusionIQLAgent(IQLAgent):
    ddpm_agent: DDPMBCAgent = None

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:

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
            argmax=True,
            half_step_size_on_overshooting=self.config[
                "q_diffusion_half_step_size_on_overshooting"
            ],
            overshooting_factor=self.config["q_diffusion_overshooting_factor"],
        )

        return action_distribution

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        argmax=False,
        **kwargs,
    ) -> jnp.ndarray:
        dist = self.forward_policy(observations, seed, train=False)
        return dist.mode()

    @classmethod
    def create(cls, *args, **kwargs):
        ddpm_agent = kwargs.pop("ddpm_agent", 42.0)
        agent = super(DiffusionIQLAgent, cls).create(*args, **kwargs)
        agent = agent.replace(
            ddpm_agent=ddpm_agent,
        )
        return agent
