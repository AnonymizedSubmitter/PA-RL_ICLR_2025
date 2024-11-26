from .continuous.bc import BCAgent
from .continuous.calql import CalQLAgent
from .continuous.cql import ContinuousCQLAgent
from .continuous.ddpm_bc import DDPMBCAgent
from .continuous.iql import IQLAgent
from .continuous.diffusion_iql import DiffusionIQLAgent
from .continuous.sac import SACAgent
from .continuous.diffusion_cql import DiffusionCQLAgent
from .continuous.diffusion_q_learning import DiffusionQLearningAgent
from .continuous.auto_regressive_transformer import AutoRegressiveTransformerAgent

agents = {
    "ddpm_bc": DDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "diffusion_iql": DiffusionIQLAgent,
    "cql": ContinuousCQLAgent,
    "calql": CalQLAgent,
    "diffusion_cql": DiffusionCQLAgent,
    "diffusion_q_learning": DiffusionQLearningAgent,
    "sac": SACAgent,
    "auto_regressive_transformer": AutoRegressiveTransformerAgent,
}
