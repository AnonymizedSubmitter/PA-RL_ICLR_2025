from typing import Any, Callable, Dict, Optional
from collections import deque
import itertools
import os

from absl import app, flags, logging
from functools import partial
import gym
from gym.vector import VectorEnv
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm
import flax
from flax.training import checkpoints
import optax
import orbax.checkpoint
from ml_collections import config_flags
import cv2  # For text drawing in videos
from jaxrl_m.envs.d4rl import (
    calc_return_to_go,
    get_d4rl_dataset_with_mc_calculation,
)
from jaxrl_m.agents import agents
from jaxrl_m.agents.continuous.ddpm_bc import DDPMBCAgent
from jaxrl_m.agents.continuous.gc_ddpm_bc import GCDDPMBCAgent

from jaxrl_m.common.common import shard_batch
from jaxrl_m.vision import encoders
from jaxrl_m.vision.data_augmentations import batched_random_crop
from jaxrl_m.common.evaluation import (
    supply_rng,
    evaluate_with_trajectories_vectorized,
)
from jaxrl_m.data.dataset import Dataset
from jaxrl_m.data.replay_buffer import ReplayBuffer
from jaxrl_m.data.image_replay_buffer import ImageReplayBuffer

from jaxrl_m.common.wandb import WandBLogger
import wandb
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.utils.train_utils import (
    concatenate_batches,
    load_recorded_video,
    subsample_batch,
)
from jaxrl_m.agents.continuous.q_diffusion import (
    QDiffusionState,
    q_diffusion_steps,
    q_diffusion_sample_actions,
)
from jaxrl_m.data.bridge_dataset import BridgeDataset, glob_to_path_list

from omegaconf import OmegaConf, open_dict  # For CALVIN config modification

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS


def preprocess_dataset_for_ddpm(dataset):
    for key in ["observations", "next_observations", "actions"]:
        if type(dataset[key]) == dict:
            if "image" in dataset[key]:
                assert "image" in dataset[key] and "proprio" in dataset[key]
                dataset[key]["image"] = dataset[key]["image"][:, None]
                dataset[key]["proprio"] = dataset[key]["proprio"][:, None]
            else:
                assert "state" in dataset[key]
                dataset[key]["state"] = dataset[key]["state"][:, None]
            if "ddpm_actions" in dataset[key]:
                dataset[key]["ddpm_actions"] = dataset[key]["ddpm_actions"][:, None]
        else:
            dataset[key] = dataset[key][:, None]  # Add empty chunking dimension
    return dataset


def unbatch_dataset_from_ddpm(dataset):
    for key in ["observations", "next_observations", "actions"]:
        # dataset[key] = dataset[key][:, 0]
        dataset[key] = jax.tree_map(lambda x: x[:, 0], dataset[key])
    return dataset


def preprocess_batch_with_q_diffusion(
    batch: Dict[str, jnp.ndarray],
    q_diffusion_agent,
    n_steps: int,
    step_size: float,
    optimize_critic_ensemble_min: bool,
    use_adam: bool,
    adam_kwargs: Optional[Dict[str, Any]],
    action_optimizer_state: Optional[optax.OptState],
    action_space_low: gym.Space,
    action_space_high: gym.Space,
    half_step_size_on_overshooting: bool,
    overshooting_factor: float,
    improve_actions_with_global_search: bool,
    ddpm_agent: Optional[flax.struct.PyTreeNode] = None,
    num_ddpm_actions: int = 32,
    num_actions_to_keep_for_q_diffusion: int = 10,
    distill_argmax: bool = True,
    rng: Optional[jax.random.PRNGKey] = None,
) -> Dict[str, jnp.ndarray]:
    assert len(batch["actions"].shape) == 3 and batch["actions"].shape[1] == 1, batch[
        "actions"
    ].shape
    if isinstance(batch["observations"], dict):
        if "image" in batch["observations"]:
            assert (
                len(batch["observations"]["image"].shape) == 5
                and batch["observations"]["image"].shape[1] == 1
            ), batch["observations"]["image"].shape
        else:
            assert (
                batch["observations"]["state"].ndim == 3
                and batch["observations"]["state"].shape[1] == 1
            )
    else:
        assert (
            len(batch["observations"].shape) == 3
            and batch["observations"].shape[1] == 1
        ), batch["observations"].shape

    # Unbatch the dataset
    batch = unbatch_dataset_from_ddpm(batch)
    observations = batch["observations"]
    if "goals" in batch:
        observations = (observations, batch["goals"])

    if improve_actions_with_global_search:
        assert ddpm_agent is not None
        assert rng is not None
        rng, key = jax.random.split(rng)
        action_distribution, info, new_adam_state = q_diffusion_sample_actions(
            observations,
            ddpm_agent,
            q_diffusion_agent,
            num_ddpm_actions=num_ddpm_actions,
            num_actions_to_keep_for_q_diffusion=num_actions_to_keep_for_q_diffusion,
            q_diffusion_num_steps=n_steps,
            q_diffusion_step_size=step_size,
            q_diffusion_optimize_critic_ensemble_min=optimize_critic_ensemble_min,
            use_adam=use_adam,
            rng=key,
            action_space_low=action_space_low,
            action_space_high=action_space_high,
            q_diffusion_adam_kwargs=adam_kwargs,
            adam_state=action_optimizer_state,
            argmax=distill_argmax,
            half_step_size_on_overshooting=half_step_size_on_overshooting,
            overshooting_factor=overshooting_factor,
            dataset_actions_to_consider=batch["actions"],
        )
        batch["actions"] = action_distribution.sample(seed=rng)
    else:
        if isinstance(observations, dict) and "ddpm_actions" in observations:
            observations = observations["state"]
        q_diffusion_results: QDiffusionState = q_diffusion_steps(
            observations,
            batch["actions"],
            critic=q_diffusion_agent,
            max_num_steps=n_steps,
            step_size=step_size,
            optimize_critic_ensemble_min=optimize_critic_ensemble_min,
            use_adam=use_adam,
            adam_kwargs=adam_kwargs,
            adam_state=action_optimizer_state,
            action_space_low=action_space_low,
            action_space_high=action_space_high,
            half_step_size_on_overshooting=half_step_size_on_overshooting,
            overshooting_factor=overshooting_factor,
        )
        batch["actions"] = q_diffusion_results.actions
    batch = preprocess_dataset_for_ddpm(batch)
    return batch


def get_batch_from_dataset(
    environment_name: str,
    dataset,
    dataset_iterator,
    batch_size: int,
    preprocess_for_ddpm: bool = False,
):
    if environment_name in ["real_robot", "calvin"]:
        batch = next(dataset_iterator)
    else:
        batch = subsample_batch(dataset, batch_size)

    if preprocess_for_ddpm:
        batch = preprocess_dataset_for_ddpm(batch)

    return batch


def train_agent_define_flags():
    flags.DEFINE_string("environment_name", "", "Environment name.")
    flags.DEFINE_string("wandb_project_name", "Q-Diffusion", "WandB project name.")
    flags.DEFINE_string("wandb_experiment_name", "", "WandB experiment name.")
    flags.DEFINE_string("wandb_group", "", "WandB group.")
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training hyperparameter configuration.",
        lock_config=False,
    )
    config_flags.DEFINE_config_file(
        "bridgedata_config",
        None,
        "File path to the bridgedata configuration.",
        lock_config=False,
    )
    flags.DEFINE_integer("n_epochs", 1000, "Number of epochs to train for.")
    flags.DEFINE_integer(
        "n_train_steps_per_epoch", 1000, "Number of training steps per epoch."
    )
    flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
    flags.DEFINE_float("reward_bias", 0.0, "Reward bias.")
    flags.DEFINE_float("clip_action", 0.99999, "Clip action.")
    flags.DEFINE_integer("num_parallel_envs", 8, "Number of parallel environments.")
    flags.DEFINE_bool("debug", False, "Debug config")
    flags.DEFINE_string("resume_path", None, "Resume training from checkpoint.")
    flags.DEFINE_integer("max_episode_steps", 1000, "Maximum episode steps.")
    flags.DEFINE_string(
        "replay_buffer_path", "", "Path to replay buffer to load (Optional)."
    )
    flags.DEFINE_float("mixing_ratio", 0.5, "Ratio of offline data to online data.")

    # Q-diffusion pre-processing flags
    flags.DEFINE_bool(
        "preprocess_dataset_with_q_diffusion",
        False,
        "Preprocess dataset with Q-diffusion.",
    )
    flags.DEFINE_string("q_diffusion_model_path", "", "Path to Q-diffusion model.")
    flags.DEFINE_string("q_diffusion_agent", "conservative_iql", "Q-diffusion agent.")
    flags.DEFINE_integer("q_diffusion_n_steps", 0, "Number of gradient steps to take.")
    flags.DEFINE_float("q_diffusion_step_size", 3e-4, "Q-diffusion step size.")
    flags.DEFINE_bool(
        "skip_if_last_checkpoint_exists", False, "Don't run if last checkpoint exists."
    )
    flags.DEFINE_string("d4rl_dataset_path", "", "Path to D4RL dataset.")
    flags.DEFINE_string("openvla_cache_path", "", "Path to OpenVLA cache.")


@jax.jit
def resize_images_to_100x100(images):
    batch_size = images.shape[0]
    return jax.image.resize(images, (batch_size, 100, 100, 3), method="cubic")


def train_agent(
    config,
    bridgedata_config,
    train_dataset: Dataset,
    eval_env: VectorEnv,
    num_epochs: int,
    num_train_steps_per_epoch: int,
    wandb_project_name: Optional[str] = None,
    wandb_experiment_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    validation_dataset: Optional[Dataset] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    image_replay_buffer_iterator: Optional[tf.data.NumpyIterator] = None,
    image_replay_buffer_iterator_for_ddpm: Optional[tf.data.NumpyIterator] = None,
    mixing_ratio: float = 0.5,
    save_model: bool = True,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: float = 0.999,
    preprocess_dataset_with_q_diffusion: bool = False,
    q_diffusion_model_path: Optional[str] = None,
    q_diffusion_agent: Optional[flax.struct.PyTreeNode] = None,
    q_diffusion_n_steps: int = 0,
    q_diffusion_agent_name: str = "conservative_iql",
    q_diffusion_step_size: float = 3e-4,
    debug: bool = False,
    resume_path: Optional[str] = None,
    openvla_cache_path: Optional[str] = None,
):
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_INIT_TIMEOUT"] = "120"
    wandb.require("core")
    devices = jax.local_devices()
    num_devices = len(devices)
    assert config.batch_size % num_devices == 0

    if bridgedata_config is not None:
        environment_name = "real_robot"
    else:
        environment_name = eval_env.env_fns[0]().unwrapped.spec.id

    if wandb_project_name is not None:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update(
            {
                "project": wandb_project_name,
                "exp_descriptor": wandb_experiment_name,
                "tag": None,
                "group": wandb_group,
            }
        )
        wandb_logger = WandBLogger(
            wandb_config=wandb_config,
            variant=config.to_dict(),
            debug=debug,
        )
        save_dir = tf.io.gfile.join(
            (
                os.path.abspath(config.save_dir)
                if "gs://" not in config.save_dir
                else config.save_dir
            ),
            wandb_logger.config.project,
            wandb_logger.config.exp_descriptor,
            f"seed_{config.seed}",
        )
    else:
        wandb_logger = None
        save_dir = tf.io.gfile.join(
            os.path.abspath(config.save_dir),
        )

    rng = jax.random.PRNGKey(config.seed)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)

    if config.image_observations or environment_name == "calvin":
        offline_batch_size = config.batch_size
        if image_replay_buffer_iterator is not None:
            offline_batch_size = int(offline_batch_size * mixing_ratio)
        if offline_batch_size > 0:
            train_data_iter = train_dataset.iterator(batch_size=offline_batch_size)
        else:
            train_data_iter = None
        if config.improve_ddpm_actions_with_global_search:
            offline_batch_size_for_ddpm = config.ddpm_agent_kwargs.batch_size
            if image_replay_buffer_iterator_for_ddpm is not None:
                offline_batch_size_for_ddpm = int(
                    offline_batch_size_for_ddpm * mixing_ratio
                )
            if offline_batch_size_for_ddpm > 0:
                train_data_iter_for_ddpm = train_dataset.iterator(
                    batch_size=offline_batch_size_for_ddpm
                )
            else:
                train_data_iter_for_ddpm = None
        else:
            train_data_iter_for_ddpm = None
    else:
        train_data_iter = None
        train_data_iter_for_ddpm = None

    goal_conditioned = config.get("goal_conditioned", False)

    # Optionally load model for Q-diffusion pre-processing
    if preprocess_dataset_with_q_diffusion:
        assert (
            q_diffusion_model_path is not None or q_diffusion_agent is not None
        ) and q_diffusion_n_steps > 0
        assert config.agent == "ddpm_bc"

        if q_diffusion_agent is None:
            example_batch = get_batch_from_dataset(
                environment_name=environment_name,
                dataset=train_dataset,
                dataset_iterator=train_data_iter,
                batch_size=config.batch_size,
                preprocess_for_ddpm=True,
            )
            example_batch = shard_batch(example_batch, sharding)

            def q_diffusion_encoder_def(x):
                return x

            rng, construct_rng = jax.random.split(rng)
            q_diffusion_agent = agents[q_diffusion_agent_name].create(
                rng=construct_rng,
                observations=example_batch["observations"],
                actions=example_batch["actions"],
                encoder_def=q_diffusion_encoder_def,
                **config.q_diffusion_agent_kwargs,
            )
            del example_batch

    action_space = eval_env.action_space
    if len(action_space.shape) == 2:
        action_space = gym.spaces.Box(
            action_space.low[0], action_space.high[0], dtype=action_space.dtype
        )
    assert len(action_space.shape) == 1, action_space.shape
    if config.agent == "conservative_iql":
        config.agent_kwargs["action_space"] = action_space
    elif config.agent in ["diffusion_cql", "diffusion_iql", "mc_critic", "sarsa"]:
        config.agent_kwargs["action_space_low"] = action_space.low
        config.agent_kwargs["action_space_high"] = action_space.high

    # Optionally create DDPM agent for conservative IQL / Diffusion CQL
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    is_openvla_agent = False
    is_transformer_ddpm_agent = False
    is_cem_agent = False
    if config.get("ddpm_agent_path", "") != "":
        assert config.agent in [
            "conservative_iql",
            "diffusion_cql",
            "diffusion_iql",
            "mc_critic",
            "sarsa",
        ]

        if config.image_observations:
            ddpm_encoder_def = encoders[config.encoder](**config.encoder_kwargs)

        else:

            def ddpm_encoder_def(x, **kwargs):
                return x

        rng, construct_rng = jax.random.split(rng)
        example_ddpm_batch = get_batch_from_dataset(
            environment_name=environment_name,
            dataset=train_dataset,
            dataset_iterator=(
                train_data_iter
                if train_data_iter is not None
                else image_replay_buffer_iterator
            ),
            batch_size=config.batch_size,
            preprocess_for_ddpm=True,
        )
        example_ddpm_batch = shard_batch(example_ddpm_batch, sharding)

        if config["ddpm_agent_path"] == "openvla":
            from jaxrl_m.agents.continuous.openvla import OpenVLAAgent

            ddpm_agent_class = OpenVLAAgent
            config.ddpm_agent_kwargs["action_std"] = (
                bridgedata_config.action_proprio_metadata["action"]["std"]
            )
            is_openvla_agent = True
        elif config["ddpm_agent_path"].startswith("bc:"):
            from jaxrl_m.agents.continuous.bc import BCAgent

            ddpm_agent_class = BCAgent
            config["ddpm_agent_path"] = config["ddpm_agent_path"].replace("bc:", "")
        elif config["ddpm_agent_path"].startswith("transformer:"):
            from jaxrl_m.agents.continuous.auto_regressive_transformer import (
                AutoRegressiveTransformerAgent,
            )

            ddpm_agent_class = AutoRegressiveTransformerAgent
            config["ddpm_agent_path"] = config["ddpm_agent_path"].replace(
                "transformer:", ""
            )
            is_transformer_ddpm_agent = True
        elif config["ddpm_agent_path"] == "cem":
            from jaxrl_m.agents.continuous.cem_optimization import (
                CrossEntropyMethodOptimizationAgent,
            )

            ddpm_agent_class = CrossEntropyMethodOptimizationAgent
            is_cem_agent = True
            config.ddpm_agent_kwargs["action_space_low"] = action_space.low
            config.ddpm_agent_kwargs["action_space_high"] = action_space.high
        else:
            ddpm_agent_class = DDPMBCAgent
        if is_transformer_ddpm_agent or is_cem_agent:
            ddpm_agent = ddpm_agent_class(
                rng=construct_rng,
                observations=example_ddpm_batch["observations"],
                goals=(
                    example_ddpm_batch["goals"]
                    if "goals" in example_ddpm_batch
                    else None
                ),
                actions=example_ddpm_batch["actions"],
                encoder_def=ddpm_encoder_def,
                action_min=action_space.low.min(),
                action_max=action_space.high.max(),
                **config.ddpm_agent_kwargs,
            )
        else:
            ddpm_agent = ddpm_agent_class.create(
                rng=construct_rng,
                observations=example_ddpm_batch["observations"],
                goals=(
                    example_ddpm_batch["goals"]
                    if "goals" in example_ddpm_batch
                    else None
                ),
                actions=example_ddpm_batch["actions"],
                encoder_def=ddpm_encoder_def,
                action_min=action_space.low.min(),
                action_max=action_space.high.max(),
                **config.ddpm_agent_kwargs,
            )

        if is_openvla_agent:
            config.agent_kwargs["ddpm_agent"] = 42.0
        elif is_transformer_ddpm_agent:
            config.agent_kwargs["ddpm_agent"] = 42.0
            ddpm_agent.restore_checkpoint(config["ddpm_agent_path"])
        elif is_cem_agent:
            pass
        else:
            ddpm_agent = orbax_checkpointer.restore(
                config.ddpm_agent_path, item=ddpm_agent
            )
            config.agent_kwargs["ddpm_agent"] = ddpm_agent
        del example_ddpm_batch

    elif config.get("use_metaworld_scripted_policy"):
        assert environment_name.startswith("metaworld-")
        from jaxrl_m.envs.metaworld import get_scripted_policy

        config.agent_kwargs["ddpm_agent"] = get_scripted_policy(
            environment_name.split("metaworld-")[-1]
        )
        ddpm_agent = None
    else:
        ddpm_agent = None
        is_openvla_agent = False
        is_transformer_ddpm_agent = False

    example_batch = get_batch_from_dataset(
        environment_name=environment_name,
        dataset=train_dataset,
        dataset_iterator=(
            train_data_iter
            if train_data_iter is not None
            else image_replay_buffer_iterator
        ),
        batch_size=config.batch_size,
        preprocess_for_ddpm=("ddpm" in config.agent),
    )
    if is_openvla_agent:
        # Shrink numpy observations from batch to 100x100
        example_batch["observations"]["image"] = np.array(
            [
                cv2.resize(
                    obs,
                    (100, 100),
                    interpolation=cv2.INTER_AREA,
                )
                for obs in example_batch["observations"]["image"]
            ]
        )
        example_batch["next_observations"]["image"] = np.array(
            [
                cv2.resize(
                    obs,
                    (100, 100),
                    interpolation=cv2.INTER_AREA,
                )
                for obs in example_batch["next_observations"]["image"]
            ]
        )

    if config.image_observations:
        logging.info(f"Batch size: {example_batch['observations']['proprio'].shape[0]}")
        logging.info(f"Number of devices: {num_devices}")
        logging.info(
            f"Batch size per device: {example_batch['observations']['proprio'].shape[0] // num_devices}"
        )
    else:
        logging.info(f"Batch size: {example_batch['observations'].shape[0]}")
        logging.info(f"Number of devices: {num_devices}")
        logging.info(
            f"Batch size per device: {example_batch['observations'].shape[0] // num_devices}"
        )

    example_batch = shard_batch(example_batch, sharding)
    if is_openvla_agent:
        example_batch["observations"]["image"] = resize_images_to_100x100(
            example_batch["observations"]["image"]
        )
        example_batch["next_observations"]["image"] = resize_images_to_100x100(
            example_batch["next_observations"]["image"]
        )

    # define encoder
    if config.image_observations:
        encoder_def = encoders[config.encoder](**config.encoder_kwargs)

    else:
        if "ddpm" in config.agent:

            def encoder_def(x, **kwargs):
                if x.shape[1] == 1:
                    return x[:, 0]  # remove time dimension
                return x

        elif is_transformer_ddpm_agent or is_cem_agent:

            def encoder_def(x, **kwargs):
                if isinstance(x, dict):
                    return x["state"]
                return x

        else:

            def encoder_def(x, **kwargs):
                return x

    # initialize agent
    rng, construct_rng = jax.random.split(rng)
    if config.get("bound_q_targets", False):
        min_q_target = train_dataset["rewards"].min() / (
            1 - config.agent_kwargs.discount
        )
        max_q_target = train_dataset["rewards"].max() / (
            1 - config.agent_kwargs.discount
        )
        wandb_logger.log(
            {"training": {"min_q_target": min_q_target, "max_q_target": max_q_target}},
            step=0,
        )
    else:
        min_q_target = None
        max_q_target = None

    is_transformer_agent = config.agent in ["auto_regressive_transformer"]
    if is_transformer_agent:
        agent = agents[config.agent](
            rng=construct_rng,
            observations=example_batch["observations"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **(
                {
                    "action_min": action_space.low.min(),
                    "action_max": action_space.high.max(),
                }
                if "ddpm" in config.agent
                else {}
            ),
            **config.agent_kwargs,
        )

    else:
        observations = example_batch["observations"]
        if is_transformer_ddpm_agent:
            observations = {"state": example_batch["observations"]}
        agent = agents[config.agent].create(
            rng=construct_rng,
            observations=observations,
            goals=example_batch["goals"] if "goals" in example_batch else None,
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            min_q_target=min_q_target,
            max_q_target=max_q_target,
            **(
                {
                    "action_min": action_space.low.min(),
                    "action_max": action_space.high.max(),
                }
                if "ddpm" in config.agent
                else {}
            ),
            **config.agent_kwargs,
        )

    del example_batch
    if resume_path is not None:
        agent = orbax_checkpointer.restore(resume_path, item=agent)
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    if not is_transformer_agent:
        agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    action_optimizer_state = None

    timer = Timer()

    for i in tqdm(range(num_epochs + 1)):
        timer.tick("total")

        """update"""
        timer.tick("train")
        for batch_idx in range(num_train_steps_per_epoch):
            timer.tick("critic_get_batch")
            if replay_buffer is not None or image_replay_buffer_iterator is not None:
                if mixing_ratio > 0:
                    offline_batch = get_batch_from_dataset(
                        environment_name=environment_name,
                        dataset=train_dataset,
                        dataset_iterator=train_data_iter,
                        batch_size=int(config.batch_size * mixing_ratio),
                        preprocess_for_ddpm=("ddpm" in config.agent),
                    )
                else:
                    offline_batch = None

                if image_replay_buffer_iterator is not None:
                    online_batch = next(image_replay_buffer_iterator)
                    assert online_batch["rewards"].shape[0] == config.batch_size * (
                        1 - mixing_ratio
                    )
                else:
                    assert replay_buffer is not None
                    if offline_batch is not None:
                        online_batch_size = (
                            config.batch_size - offline_batch["observations"].shape[0]
                        )
                    else:
                        online_batch_size = config.batch_size
                    online_batch = replay_buffer.sample(online_batch_size).unfreeze()

                if "maze" in environment_name or environment_name == "real_robot":
                    online_batch["masks"] = (
                        online_batch["rewards"] != (1 + reward_bias)
                    ).astype(np.float32)
                elif "kitchen" in environment_name:
                    online_batch["masks"] = (
                        online_batch["rewards"] != (4 + reward_bias)
                    ).astype(np.float32)
                if offline_batch is not None:
                    batch = concatenate_batches([offline_batch, online_batch])
                else:
                    batch = online_batch
                assert batch["rewards"].shape[0] == config.batch_size
            else:
                batch = get_batch_from_dataset(
                    environment_name=environment_name,
                    dataset=train_dataset,
                    dataset_iterator=train_data_iter,
                    batch_size=config.batch_size,
                    preprocess_for_ddpm=("ddpm" in config.agent),
                )
            # Masks checks
            if "maze" in environment_name or environment_name == "real_robot":
                np.logical_or(batch["masks"] == 1, batch["rewards"] == 1 + reward_bias)
            elif "kitchen" in environment_name:
                assert np.all(
                    np.logical_or(
                        batch["masks"] == 1, batch["rewards"] == (4 + reward_bias)
                    )
                )

            if environment_name == "real_robot":
                # Clip actions to action space
                batch["actions"] = np.clip(
                    batch["actions"], action_space.low, action_space.high
                )

            if is_transformer_ddpm_agent or is_openvla_agent or is_cem_agent:
                if is_transformer_ddpm_agent or is_cem_agent:
                    batch = shard_batch(batch, sharding)
                timer.tick("separate_ddpm_actions")

                rng, key = jax.random.split(rng)
                ddpm_actions = ddpm_agent.sample_actions(
                    batch["observations"],
                    repeat=config.agent_kwargs.num_ddpm_actions,
                    **(
                        {"timer": timer, "cache_dir": openvla_cache_path}
                        if is_openvla_agent
                        else {}
                    ),
                    **({"critic_agent": agent, "seed": key} if is_cem_agent else {}),
                )
                ddpm_next_actions = ddpm_agent.sample_actions(
                    batch["next_observations"],
                    repeat=config.agent_kwargs.num_ddpm_actions,
                    **(
                        {"timer": timer, "cache_dir": openvla_cache_path}
                        if is_openvla_agent
                        else {}
                    ),
                    **({"critic_agent": agent, "seed": key} if is_cem_agent else {}),
                )
                if not isinstance(batch["observations"], dict):
                    batch["observations"] = {
                        "state": batch["observations"],
                        "ddpm_actions": ddpm_actions,
                    }
                    batch["next_observations"] = {
                        "state": batch["next_observations"],
                        "ddpm_actions": ddpm_next_actions,
                    }
                else:
                    batch["observations"]["ddpm_actions"] = ddpm_actions
                    batch["next_observations"]["ddpm_actions"] = ddpm_next_actions
                timer.tock("separate_ddpm_actions")

            timer.tick("critic_shard_batch")
            batch = shard_batch(batch, sharding)
            timer.tock("critic_shard_batch")

            if is_openvla_agent:
                timer.tick("critic_image_resize")
                batch["observations"]["image"] = resize_images_to_100x100(
                    batch["observations"]["image"]
                )
                batch["next_observations"]["image"] = resize_images_to_100x100(
                    batch["next_observations"]["image"]
                )

                timer.tock("critic_image_resize")

            timer.tock("critic_get_batch")
            if preprocess_dataset_with_q_diffusion:
                timer.tick("preprocess_with_q_diffusion")
                batch = preprocess_batch_with_q_diffusion(
                    batch,
                    q_diffusion_agent,
                    n_steps=q_diffusion_n_steps,
                    step_size=q_diffusion_step_size,
                    optimize_critic_ensemble_min=config.agent_kwargs[
                        "q_diffusion_optimize_critic_ensemble_min"
                    ],
                    use_adam=config.agent_kwargs["use_adam"],
                    adam_kwargs=config.agent_kwargs["adam_kwargs"].to_dict(),
                    action_optimizer_state=action_optimizer_state,
                    action_space_low=action_space.low,
                    action_space_high=action_space.high,
                    half_step_size_on_overshooting=config.agent_kwargs[
                        "q_diffusion_half_step_size_on_overshooting"
                    ],
                    overshooting_factor=config.agent_kwargs[
                        "q_diffusion_overshooting_factor"
                    ],
                )
                timer.tock("preprocess_with_q_diffusion")
            timer.tick("agent.update")
            if config.agent in ["diffusion_cql", "mc_critic", "sarsa"]:
                agent, critic_update_info, action_optimizer_state = agent.update(
                    batch,
                    action_optimizer_state=action_optimizer_state,
                )
            else:
                agent, critic_update_info = agent.update(
                    batch,
                )
            timer.tock("agent.update")

            if config.improve_ddpm_actions_with_global_search:
                timer.tick("ddpm_get_batch")
                if (
                    replay_buffer is not None
                    or image_replay_buffer_iterator_for_ddpm is not None
                ):
                    if mixing_ratio > 0:
                        offline_batch = get_batch_from_dataset(
                            environment_name=environment_name,
                            dataset=train_dataset,
                            dataset_iterator=train_data_iter_for_ddpm,
                            batch_size=int(
                                config.ddpm_agent_kwargs.batch_size * mixing_ratio
                            ),
                            preprocess_for_ddpm=False,  # Will preprocess later
                        )
                    else:
                        offline_batch = None
                    if image_replay_buffer_iterator_for_ddpm is not None:
                        online_batch = next(image_replay_buffer_iterator_for_ddpm)
                        assert online_batch["observations"]["image"].shape[0] == int(
                            config.ddpm_agent_kwargs.batch_size * (1 - mixing_ratio)
                        )
                    else:
                        assert replay_buffer is not None
                        online_batch = replay_buffer.sample(
                            int(
                                config.ddpm_agent_kwargs.batch_size * (1 - mixing_ratio)
                            )
                        ).unfreeze()

                    if offline_batch is not None:
                        batch = concatenate_batches([offline_batch, online_batch])
                    else:
                        batch = online_batch
                    assert (
                        batch["rewards"].shape[0] == config.ddpm_agent_kwargs.batch_size
                    )
                else:
                    batch = get_batch_from_dataset(
                        environment_name=environment_name,
                        dataset=train_dataset,
                        dataset_iterator=train_data_iter_for_ddpm,
                        batch_size=config.ddpm_agent_kwargs.batch_size,
                        preprocess_for_ddpm=False,  # Will preprocess later
                    )

                if environment_name == "real_robot":
                    # Clip actions to action space
                    batch["actions"] = np.clip(
                        batch["actions"], action_space.low, action_space.high
                    )

                if is_transformer_ddpm_agent or is_openvla_agent or is_cem_agent:
                    timer.tick("separate_ddpm_actions")
                    ddpm_actions = ddpm_agent.sample_actions(
                        batch["observations"],
                        repeat=config.agent_kwargs.num_ddpm_actions,
                        **({"timer": timer} if is_openvla_agent else {}),
                        **({"critic_agent": agent} if is_cem_agent else {}),
                    )
                    if not isinstance(batch["observations"], dict):
                        batch["observations"] = {
                            "state": batch["observations"],
                            "ddpm_actions": ddpm_actions,
                        }
                    else:
                        batch["observations"]["ddpm_actions"] = ddpm_actions
                    timer.tick("separate_ddpm_actions")
                    ddpm_agent.prepare_for_finetuning()

                batch = preprocess_dataset_for_ddpm(batch)

                batch = shard_batch(batch, sharding)
                timer.tock("ddpm_get_batch")
                timer.tick("global_search")
                rng, key = jax.random.split(rng)
                batch = preprocess_batch_with_q_diffusion(
                    batch,
                    agent,
                    n_steps=q_diffusion_n_steps,
                    step_size=q_diffusion_step_size,
                    optimize_critic_ensemble_min=config.agent_kwargs[
                        "q_diffusion_optimize_critic_ensemble_min"
                    ],
                    use_adam=config.agent_kwargs["q_diffusion_use_adam"],
                    adam_kwargs=config.agent_kwargs[
                        "q_diffusion_adam_kwargs"
                    ].to_dict(),
                    action_optimizer_state=action_optimizer_state,
                    action_space_low=action_space.low,
                    action_space_high=action_space.high,
                    half_step_size_on_overshooting=config.agent_kwargs[
                        "q_diffusion_half_step_size_on_overshooting"
                    ],
                    overshooting_factor=config.agent_kwargs[
                        "q_diffusion_overshooting_factor"
                    ],
                    improve_actions_with_global_search=True,
                    ddpm_agent=(
                        ddpm_agent
                        if not getattr(ddpm_agent, "pytorch", False)
                        else 42.0
                    ),
                    num_ddpm_actions=config.agent_kwargs.num_ddpm_actions,
                    num_actions_to_keep_for_q_diffusion=config.agent_kwargs.num_actions_to_keep_for_q_diffusion,
                    rng=key,
                )
                timer.tock("global_search")
                timer.tick("ddpm_update")
                ddpm_agent, ddpm_update_info = ddpm_agent.update(batch)
                if getattr(ddpm_agent, "pytorch", False):
                    ddpm_agent.prepare_for_inference()
                else:
                    agent = agent.replace(ddpm_agent=ddpm_agent)

                timer.tock("ddpm_update")
            else:
                ddpm_update_info = {}

            timer.tick("wandb_logging")
            if batch_idx == 0:
                critic_update_info = jax.device_get(critic_update_info)
                batch_info = {
                    "rewards_mean": np.mean(batch["rewards"]),
                    "rewards_std": np.std(batch["rewards"]),
                    "rewards_max": np.max(batch["rewards"]),
                    "rewards_min": np.min(batch["rewards"]),
                    "masks_mean": np.mean(batch["masks"]),
                    "masks_std": np.std(batch["masks"]),
                    "masks_max": np.max(batch["masks"]),
                    "masks_min": np.min(batch["masks"]),
                    "actions_mean": np.mean(batch["actions"]),
                    "actions_std": np.std(batch["actions"]),
                    "actions_max": np.max(batch["actions"]),
                    "actions_min": np.min(batch["actions"]),
                }
                if "mc_returns" in batch:
                    batch_info.update(
                        {
                            "mc_returns_mean": np.mean(batch["mc_returns"]),
                            "mc_returns_std": np.std(batch["mc_returns"]),
                            "mc_returns_max": np.max(batch["mc_returns"]),
                            "mc_returns_min": np.min(batch["mc_returns"]),
                        }
                    )
                if wandb_logger is not None and (i + 1) % config.log_interval == 0:
                    wandb_logger.log(
                        {
                            "training": critic_update_info,
                            "batch_info": batch_info,
                            "ddpm": ddpm_update_info,
                        },
                        step=i,
                    )
            timer.tock("wandb_logging")

        timer.tock("train")

        if (
            i + 1
        ) % config.eval_interval == 0 and eval_env is not None:  # environment_name != "real_robot":
            # if validation_dataset is not None:
            #     # """validation"""
            #     logging.info("Validation...")
            #     timer.tick("val")
            #     metrics = []
            #     for batch_index in range(10):  # validation_dataset.get_iterator():
            #         batch = subsample_batch(validation_dataset, config.batch_size)
            #         batch = shard_batch(batch, sharding)
            #         rng, val_rng = jax.random.split(rng)
            #         metrics.append(agent.get_debug_metrics(batch, seed=val_rng))
            #     metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            #     if wandb_logger is not None:
            #         wandb_logger.log({"validation": metrics}, step=i)
            #     timer.tock("val")

            """eval"""
            logging.info("Evaluating...")
            timer.tick("evaluation")

            rng, policy_key = jax.random.split(rng)
            if "ddpm" in config.agent:

                def sample_actions(observations, *args, **kwargs):
                    if goal_conditioned:
                        if FLAGS.config.image_observations:
                            assert (
                                len(observations["goal"].shape) == 4
                                and len(observations["image"].shape) == 4
                                and len(observations["proprio"].shape) == 2
                            )
                            observations["image"] = observations["image"][:, None]
                            observations["proprio"] = observations["proprio"][:, None]
                            goals = {"image": observations["goal"]}
                        else:
                            assert (
                                len(observations["goal"].shape) == 2
                                and len(observations["ob"].shape) == 2
                                and len(observations["proprio"].shape) == 2
                            )
                            goals = observations["goal"]
                            observations = observations["ob"][:, None]

                        actions = jax.device_get(
                            agent.sample_actions(
                                observations,
                                goals,
                                *args,
                                **kwargs,
                            )
                        )
                        assert len(actions.shape) == 3, actions.shape  # (B, T, A)
                        return actions[:, 0]
                    else:
                        if not isinstance(observations, dict):
                            assert len(observations.shape) == 2
                        else:
                            assert len(observations["proprio"].shape) == 2
                        actions = jax.device_get(
                            agent.sample_actions(
                                jax.tree_map(lambda x: x[:, None], observations),
                                *args,
                                **kwargs,
                            )
                        )
                        assert len(actions.shape) == 3, actions.shape  # (B, T, A)
                        return actions[:, 0]

            else:

                def sample_actions(observations, *args, ddpm_agent=None, **kwargs):
                    if goal_conditioned:
                        if FLAGS.config.image_observations:
                            assert (
                                len(observations["goal"].shape) == 4
                                and len(observations["image"].shape) == 4
                                and len(observations["proprio"].shape) == 2
                            )
                            goals = {"image": observations["goal"]}
                        else:
                            assert (
                                len(observations["goal"].shape) == 2
                                and len(observations["ob"].shape) == 2
                                and len(observations["proprio"].shape) == 2
                            )
                            goals = observations["goal"]
                            observations = observations["ob"]
                    else:
                        goals = None

                    if ddpm_agent is not None:
                        if not isinstance(observations, dict):
                            observations = {"state": observations}

                        observations["ddpm_actions"] = ddpm_agent.sample_actions(
                            observations,
                            repeat=config.agent_kwargs.num_ddpm_actions,
                            **({"timer": timer} if is_openvla_agent else {}),
                            **(
                                {"critic_agent": agent, "seed": kwargs["seed"]}
                                if is_cem_agent
                                else {}
                            ),
                        )

                    actions = jax.device_get(
                        agent.sample_actions(
                            observations,
                            *args,
                            **kwargs,
                            goals=goals,
                            argmax=config.deterministic_eval,
                            # transformer=transformer,
                        )
                    )
                    if is_transformer_agent:
                        assert (
                            actions.ndim == 3 and actions.shape[1] == 1
                        ), actions.shape
                        actions = actions[:, 0]
                    return actions

            if (
                not config.deterministic_eval
                or "ddpm" in config.agent
                or "diffusion" in config.agent
            ):
                policy_fn = supply_rng(
                    sample_actions,
                    rng=policy_key,
                )
            else:
                policy_fn = sample_actions

            if is_openvla_agent or is_transformer_ddpm_agent or is_cem_agent:
                if hasattr(ddpm_agent, "reset"):
                    ddpm_agent.reset()

                policy_fn = partial(
                    policy_fn,
                    ddpm_agent=ddpm_agent,
                )

            if config.save_video:
                try:
                    eval_env.start_recording(
                        config.num_episodes_per_video, config.num_episodes_per_row
                    )
                except Exception as e:
                    pass
            if config.num_eval_episodes > 0:
                trajectories = evaluate_with_trajectories_vectorized(
                    policy_fn,
                    eval_env,
                    config.num_eval_episodes,
                )

                # log Q - MC
                if hasattr(agent, "forward_critic") and environment_name != "calvin":
                    timer.tick("q-mc calculation")
                    initial_states = [t["observation"][0] for t in trajectories]
                    initial_actions = [t["action"][0] for t in trajectories]
                    initial_qs = jax.tree_map(
                        lambda s, a: agent.forward_critic(
                            s, a, rng=None, train=False
                        ).min(),
                        initial_states,
                        initial_actions,
                    )
                    mc_returns = jax.tree_map(
                        lambda t: calc_return_to_go(
                            env_name=environment_name,
                            rewards=np.array(t["reward"]) * reward_scale + reward_bias,
                            masks=1 - np.array(t["done"]),
                            gamma=config.agent_kwargs.discount,
                            reward_scale=reward_scale,
                            reward_bias=reward_bias,
                            is_sparse_reward="antmaze" in environment_name
                            or environment_name == "real_robot",
                        ),
                        trajectories,
                        is_leaf=lambda x: isinstance(
                            x, dict
                        ),  # only map over traj in trajs
                    )
                    initial_mc_returns = jax.tree_map(lambda t: t[0], mc_returns)

                    timer.tock("q-mc calculation")
                else:
                    initial_qs = None
                    initial_mc_returns = None

                if FLAGS.environment_name == "calvin" and config.save_video:
                    trajectories_to_save = trajectories[: config.num_episodes_per_video]
                    frames = []
                    for traj in trajectories_to_save:
                        trajectory_return = 0
                        for transition, reward in zip(
                            traj["observation"], traj["reward"]
                        ):
                            assert transition["image"].shape[-1] == 3
                            if len(transition["image"].shape) == 4:
                                transition["image"] = transition["image"][0]
                            image = transition["image"]
                            if goal_conditioned:
                                goal_image = transition["image_goal"]
                                image = np.concatenate([image, goal_image], axis=1)
                            # Add text for reward and return so far
                            trajectory_return += reward
                            frame = cv2.putText(
                                image,
                                f"reward: {reward}. return: {trajectory_return}",
                                (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (0, 0, 0),
                                1,
                            )
                            frame = frame.transpose(2, 0, 1)
                            frames.append(frame)
                    frames = np.array(frames)
                    wandb.log(
                        {
                            "video": wandb.Video(
                                frames,
                                fps=24,
                                format="mp4",
                            )
                        },
                        step=i,
                    )

                eval_metrics = {
                    "eval/average_return": np.mean(
                        [np.sum(t["reward"]) for t in trajectories]
                    ),
                    "eval/average_episode_length": np.mean(
                        [len(t["reward"]) for t in trajectories]
                    ),
                    **(
                        {
                            "eval/average_normalized_return": np.mean(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                            "eval/min_normalized_return": np.min(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                            "eval/max_normalized_return": np.max(
                                [
                                    eval_env.get_normalized_score(np.sum(t["reward"]))
                                    for t in trajectories
                                ]
                            ),
                        }
                        if hasattr(eval_env, "get_normalized_score")
                        else {}
                    ),
                    "eval/average_max_reward": np.mean(
                        [np.max(t["reward"]) for t in trajectories]
                    ),
                    **(
                        {
                            "eval/initial state Q": wandb.Histogram(initial_qs),
                            "eval/initial state MC": wandb.Histogram(
                                initial_mc_returns
                            ),
                            "eval/Q - MC": wandb.Histogram(
                                np.array(initial_qs) - np.array(initial_mc_returns)
                            ),
                        }
                        if initial_qs is not None
                        else {}
                    ),
                }
                if wandb_logger is not None:
                    wandb_logger.log(eval_metrics, step=i)

            if config.save_video:
                try:
                    eval_video = load_recorded_video(
                        video_path=eval_env.current_save_path
                    )
                    if wandb_logger is not None:
                        wandb_logger.log({"evaluation/video": eval_video}, step=i)
                except Exception as e:
                    pass
            timer.tock("evaluation")

        if i % config.save_interval == 0:
            logging.info("Saving checkpoint...")
            if is_transformer_agent:
                checkpoint_path = tf.io.gfile.join(save_dir, f"checkpoint_{i}")
                agent.save_checkpoint(checkpoint_path)
            else:
                checkpoint_path = checkpoints.save_checkpoint(
                    save_dir, agent, step=i, keep=1e6, overwrite=True
                )

            logging.info("Saved checkpoint to %s", checkpoint_path)
            if config.improve_ddpm_actions_with_global_search:
                # Save ddpm model
                ddpm_save_dir = tf.io.gfile.join(save_dir, "ddpm_checkpoints")
                ddpm_checkpoint_path = checkpoints.save_checkpoint(
                    ddpm_save_dir, ddpm_agent, step=i, keep=1e6, overwrite=True
                )
                logging.info("Saved ddpm checkpoint to %s", ddpm_checkpoint_path)

        timer.tock("total")

        if wandb_logger is not None and (i + 1) % config.log_interval == 0:
            wandb_logger.log(
                {"timer/total_times": timer.get_total_times(reset=False)}, step=i
            )
            wandb_logger.log({"timer/average_times": timer.get_average_times()}, step=i)


def main(_):
    if FLAGS.debug:
        import pdb

        pdb.set_trace()
        # jax.config.update("jax_disable_jit", True)

    if FLAGS.skip_if_last_checkpoint_exists:
        # check if last checkpoint exists
        last_checkpoint_potential_path = tf.io.gfile.join(
            (
                os.path.abspath(FLAGS.config.save_dir)
                if "gs://" not in FLAGS.config.save_dir
                else FLAGS.config.save_dir
            ),
            FLAGS.wandb_project_name,
            FLAGS.wandb_experiment_name,
            f"seed_{FLAGS.config.seed}",
            f"checkpoint_{FLAGS.n_epochs}/checkpoint",
        )
        if tf.io.gfile.exists(last_checkpoint_potential_path):
            logging.info(
                f"Checkpoint already exists at {last_checkpoint_potential_path}. Skipping run."
            )
            return

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    if FLAGS.bridgedata_config is not None:
        # Real robot
        env = None

        assert isinstance(FLAGS.bridgedata_config.include[0], list)
        task_paths = [
            glob_to_path_list(
                path,
                prefix=FLAGS.config.data_path,
                exclude=FLAGS.bridgedata_config.exclude,
            )
            for path in FLAGS.bridgedata_config.include
        ]

        train_paths = [sub_list for sub_list in task_paths]
        val_paths = [sub_list for sub_list in task_paths]

        dataset = BridgeDataset(
            train_paths,
            FLAGS.config.seed,
            train=True,
            action_proprio_metadata=FLAGS.bridgedata_config.action_proprio_metadata,
            **FLAGS.config.dataset_kwargs,
        )
        val_dataset = BridgeDataset(
            val_paths,
            FLAGS.config.seed,
            action_proprio_metadata=FLAGS.bridgedata_config.action_proprio_metadata,
            train=False,
            **FLAGS.config.dataset_kwargs,
        )
    elif FLAGS.environment_name == "calvin":

        from jaxrl_m.envs.calvin import (
            get_calvin_config,
            get_calvin_env,
            get_calvin_tfrecord_dataset,
        )

        dataset = get_calvin_tfrecord_dataset(
            tfrecord_regexp=FLAGS.config.calvin_tfrecord_regexp,
            **FLAGS.config.dataset_kwargs,
        )
        val_dataset = None
        calvin_config = get_calvin_config()
        if not FLAGS.config.image_observations:
            calvin_config["pixel_ob"] = False
            calvin_config["state_ob"] = True

        calvin_config["screen_size"] = [
            calvin_config.screen_size[0],
            calvin_config.screen_size[1],
        ]
        calvin_config["cameras"]["static"]["width"] = calvin_config["screen_size"][0]
        calvin_config["cameras"]["static"]["height"] = calvin_config["screen_size"][1]
        calvin_config["use_egl"] = False  # Use CPU rendering
        if not FLAGS.config.image_observations:
            OmegaConf.set_struct(calvin_config, False)
            with open_dict(calvin_config):
                calvin_config["include_distractors_in_state_obs"] = (
                    FLAGS.config.calvin_include_distractors_in_state_obs
                )
        if FLAGS.num_parallel_envs > 1:
            goal_conditioned = FLAGS.config.goal_conditioned
            num_parallel_envs = FLAGS.num_parallel_envs
            env = gym.vector.AsyncVectorEnv(
                [
                    lambda: get_calvin_env(
                        goal_conditioned=goal_conditioned,
                        cfg=calvin_config,
                    )
                    for _ in range(num_parallel_envs)
                ],
                context="forkserver",  # the default "fork" is incompatible with JAX
            )
        else:
            env = gym.vector.SyncVectorEnv(
                [
                    lambda: get_calvin_env(
                        goal_conditioned=FLAGS.config.goal_conditioned,
                        cfg=calvin_config,
                    )
                ]
            )

    else:
        # Env and dataset
        if FLAGS.num_parallel_envs > 1:
            environment_name = FLAGS.environment_name
            max_episode_steps = FLAGS.max_episode_steps
            print(f"exp name {environment_name}")
            env = gym.vector.AsyncVectorEnv(
                [
                    lambda: gym.wrappers.TimeLimit(
                        gym.make(environment_name),
                        max_episode_steps=max_episode_steps,
                    )
                    for _ in range(FLAGS.num_parallel_envs)
                ],
                context="forkserver",  # the default "fork" is incompatible with JAX
            )
        else:
            env = gym.vector.SyncVectorEnv(
                [
                    lambda: gym.wrappers.TimeLimit(
                        gym.make(FLAGS.environment_name),
                        max_episode_steps=FLAGS.max_episode_steps,
                    )
                ]
            )

        dataset = get_d4rl_dataset_with_mc_calculation(
            FLAGS.environment_name,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            clip_action=FLAGS.clip_action,
            gamma=FLAGS.config.agent_kwargs.discount,
            dataset_path=FLAGS.d4rl_dataset_path,
        )
        val_dataset = None

        if (
            "antmaze" in FLAGS.environment_name
            and FLAGS.config.agent != "diffusion_q_learning"
        ):
            assert FLAGS.reward_bias == -1 and FLAGS.reward_scale == 1

        dataset["actions"] = np.clip(
            dataset["actions"], -FLAGS.clip_action, FLAGS.clip_action
        )
        # val_dataset = jax.tree_map(lambda x: x[: len(x) // 10], dataset)
        # dataset = jax.tree_map(lambda x: x[len(x) // 10 :], dataset)

        if "dones" in dataset:
            del dataset["dones"]

    # Load replay buffer
    if FLAGS.replay_buffer_path:
        if FLAGS.config.image_observations:
            data_paths = glob_to_path_list(
                tf.io.gfile.join(FLAGS.replay_buffer_path, "*.tfrecord")
            )
            replay_buffer = ImageReplayBuffer(
                data_paths=data_paths,
                seed=FLAGS.config.seed,
                train=True,
                **FLAGS.config.image_replay_buffer_kwargs,
            )
            image_replay_buffer_iterator = replay_buffer.iterator(
                batch_size=int(FLAGS.config.batch_size * (1 - FLAGS.mixing_ratio))
            )
            if FLAGS.config.improve_ddpm_actions_with_global_search:
                image_replay_buffer_iterator_for_ddpm = replay_buffer.iterator(
                    batch_size=int(
                        FLAGS.config.ddpm_agent_kwargs.batch_size
                        * (1 - FLAGS.mixing_ratio)
                    )
                )
            else:
                image_replay_buffer_iterator_for_ddpm = None
        else:
            replay_buffer_dataset_dict = dict(np.load(FLAGS.replay_buffer_path))
            replay_buffer = ReplayBuffer(
                env.observation_space,
                env.action_space,
                capacity=len(replay_buffer_dataset_dict["rewards"]),
                store_mc_return=True,
                store_max_trajectory_reward=True,
                seed=FLAGS.config.seed,
            )
            replay_buffer.dataset_dict = replay_buffer_dataset_dict
            replay_buffer.dataset_len = len(replay_buffer_dataset_dict["rewards"])
            replay_buffer._size = len(replay_buffer_dataset_dict["rewards"])
            image_replay_buffer_iterator = None
            image_replay_buffer_iterator_for_ddpm = None
    else:
        replay_buffer = None
        image_replay_buffer_iterator = None
        image_replay_buffer_iterator_for_ddpm = None

    train_agent(
        config=FLAGS.config,
        bridgedata_config=FLAGS.bridgedata_config,
        train_dataset=dataset,
        validation_dataset=val_dataset,
        replay_buffer=replay_buffer,
        image_replay_buffer_iterator=image_replay_buffer_iterator,
        image_replay_buffer_iterator_for_ddpm=image_replay_buffer_iterator_for_ddpm,
        mixing_ratio=FLAGS.mixing_ratio,
        eval_env=env,
        num_epochs=FLAGS.n_epochs,
        num_train_steps_per_epoch=FLAGS.n_train_steps_per_epoch,
        wandb_project_name=FLAGS.wandb_project_name,
        wandb_experiment_name=FLAGS.wandb_experiment_name,
        wandb_group=FLAGS.wandb_group,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        preprocess_dataset_with_q_diffusion=FLAGS.preprocess_dataset_with_q_diffusion,
        q_diffusion_model_path=FLAGS.q_diffusion_model_path,
        q_diffusion_n_steps=FLAGS.q_diffusion_n_steps,
        q_diffusion_agent=FLAGS.q_diffusion_agent,
        q_diffusion_step_size=FLAGS.q_diffusion_step_size,
        resume_path=FLAGS.resume_path,
        debug=FLAGS.debug,
        openvla_cache_path=FLAGS.openvla_cache_path,
    )


if __name__ == "__main__":
    train_agent_define_flags()
    app.run(main)
