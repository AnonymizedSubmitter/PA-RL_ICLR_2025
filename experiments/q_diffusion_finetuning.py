from absl import app, flags, logging
from typing import Any, Dict, List, Tuple, Union
from collections.abc import Iterable
import os
import time
from functools import partial
import concurrent.futures
import gym
import numpy as np
import cv2
import seaborn as sns
import shutil
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm
from flax.training import checkpoints
import orbax.checkpoint
from ml_collections import config_flags
from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.vision import encoders
from jaxrl_m.vision.data_augmentations import batched_random_crop
from jaxrl_m.common.traj import TrajSampler, calc_return_to_go
from jaxrl_m.data.replay_buffer import ReplayBuffer
from jaxrl_m.data.image_replay_buffer import (
    ImageReplayBuffer,
    save_trajectory_as_tfrecord,
)
from jaxrl_m.envs.d4rl import (
    get_d4rl_dataset_with_mc_calculation,
    TruncationWrapper,
)

from jaxrl_m.envs.calvin import (
    get_calvin_config,
    get_calvin_env,
    get_calvin_tfrecord_dataset,
)
from omegaconf import OmegaConf, open_dict  # For CALVIN config modification
import wandb
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.utils.train_utils import concatenate_batches, subsample_batch
from jaxrl_m.agents.continuous.ddpm_bc import DDPMBCAgent
from jaxrl_m.agents.continuous.q_diffusion import q_diffusion_sample_actions
from jaxrl_m.common.evaluation import evaluate_with_trajectories_vectorized
from experiments.train_agent import (
    train_agent,
    preprocess_batch_with_q_diffusion,
    get_batch_from_dataset,
    resize_images_to_100x100,
    preprocess_dataset_for_ddpm,
)
from jaxrl_m.common.evaluation import supply_rng
from jaxrl_m.data.bridge_dataset import (
    BridgeDataset,
    glob_to_path_list,
    get_task_to_initial_eep,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS
flags.DEFINE_string("environment_name", "", "Environment name.")
flags.DEFINE_string("wandb_project_name", "Q-Diffusion", "WandB project name.")
flags.DEFINE_string("wandb_experiment_name", "", "WandB experiment name.")
flags.DEFINE_string("wandb_group", "", "WandB group.")
flags.DEFINE_string("ddpm_checkpoint_path", "", "Path to the trained DDPM model.")
flags.DEFINE_string("critic_agent_name", "iql", "Critic agent name.")
flags.DEFINE_string("critic_checkpoint_path", "", "Path to the trained critic model.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_parallel_envs", 8, "Number of parallel environments.")
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
flags.DEFINE_list("initial_eep_pos", None, "Initial eef xyz position")
flags.DEFINE_integer("eval_interval", 50, "Evaluate every n epochs")
flags.DEFINE_integer("validation_interval", 50, "Evaluate every n epochs")
flags.DEFINE_integer("save_interval", 10, "Save interval.")
flags.DEFINE_bool("save_model", True, "Save model.")
flags.DEFINE_bool("save_video", False, "Save video.")
flags.DEFINE_bool("save_replay_buffer", False, "Save replay buffer.")

flags.DEFINE_integer("num_online_epochs", 1000, "Number of online epochs.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e6), "Replay buffer capacity")
flags.DEFINE_integer(
    "n_online_trajs_per_epoch",
    1,
    "Number of trajectories collected from interaction per online epoch.",
)
flags.DEFINE_integer(
    "n_warmup_trajs",
    1,
    "Number of trajectories collected from interaction before training.",
)
flags.DEFINE_float("mixing_ratio", 0.5, "Ratio of offline data to online data.")
flags.DEFINE_float("clip_action", 0.99999, "Clip action magnitude.")
flags.DEFINE_integer("utd", 1, "update-to-data ratio of the critic")
flags.DEFINE_bool("debug", False, "Debug config")
flags.DEFINE_bool("retrain_ddpm_from_scratch", False, "Retrain DDPM from scratch.")
flags.DEFINE_float(
    "ddpm_utd",
    1,
    "Update-to-data ratio of the DDPM. Only used if retrain_ddpm_from_scratch is False.",
)
flags.DEFINE_integer("retrain_ddpm_interval", 0, "Retrain DDPM frequency.")
flags.DEFINE_integer(
    "retrain_ddpm_num_epochs", 1500, "Number of epochs to retrain DDPM."
)
config_flags.DEFINE_config_file(
    "ddpm_retrain_config", None, "DDPM retrain config.", lock_config=False
)
flags.DEFINE_string(
    "retrain_ddpm_online_filtering_function",
    "all",
    "Filtering function for online data.",
)
flags.DEFINE_string(
    "replay_buffer_filtering_function",
    "all",
    "Filtering function for online data.",
)
flags.DEFINE_bool("idql_eval", False, "Evaluate with IDQL.")
flags.DEFINE_string(
    "real_robot_task_instruction", "put pot into sink", "Real robot task name."
)
flags.DEFINE_string(
    "real_robot_task_name_for_reward", "orange_pot_to_sink", "Real robot task name."
)
flags.DEFINE_string(
    "real_robot_task_name_for_initial_eep",
    "pot_to_sink_0624",
    "Real robot task name.",
)
flags.DEFINE_bool(
    "collect_data_in_separate_process", False, "Collect data in separate process."
)
flags.DEFINE_string(
    "train_on_separate_computer_mode",
    "single_computer",  # "env_steps_only", "agent_training_only"
    "Training on separate computer mode.",
)
flags.DEFINE_bool(
    "resume_failed_run", False, "Whether this run is resuming a previous failed run."
)
flags.DEFINE_string("d4rl_dataset_path", "", "Path to D4RL dataset.")


def copy_gcs_dir_to_local(gcs_dir, local_dir):
    """Copies a directory from Google Cloud Storage to a local directory."""

    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # List all files in the GCS directory
    files = tf.io.gfile.glob(os.path.join(gcs_dir, "*"))

    # Copy each file from the GCS directory to the local directory
    for file in files:
        file_name = os.path.basename(file)
        local_file_path = os.path.join(local_dir, file_name)

        # Copy file to local
        tf.io.gfile.copy(file, local_file_path, overwrite=True)


def successes_only_filtering_function(
    ddpm_dataset: Dict[str, np.ndarray],
    new_online_data: Dict[str, np.ndarray],
    env_name: str,
    reward_bias: float,
    reward_scale: float,
) -> Dict[str, np.ndarray]:
    assert "max_trajectory_rewards" in new_online_data.keys()
    assert len(ddpm_dataset["observations"].shape) == 3
    assert len(new_online_data["observations"].shape) == 2

    if "kitchen" in env_name:
        assert reward_scale == 1.0
        successful_indices = np.where(
            new_online_data["max_trajectory_rewards"] == 4.0 + reward_bias
        )[0]
    elif "antmaze" in env_name or "real_robot" in env_name:
        assert reward_scale == 1.0
        successful_indices = np.where(
            new_online_data["max_trajectory_rewards"] == 1.0 + reward_bias
        )[0]
    else:
        raise NotImplementedError

    new_online_data = {
        key: value[successful_indices] for key, value in new_online_data.items()
    }
    new_online_data = preprocess_dataset_for_ddpm(new_online_data)

    # concatenate the new online data with the ddpm dataset
    ddpm_dataset = {
        key: np.concatenate([value, new_online_data[key]], axis=0)
        for key, value in ddpm_dataset.items()
    }

    return ddpm_dataset


def kitchen_good_data_filtering_function(
    ddpm_dataset: Dict[str, np.ndarray],
    new_online_data: Dict[str, np.ndarray],
    env_name: str,
    reward_bias: float,
    reward_scale: float,
) -> Dict[str, np.ndarray]:
    assert "max_trajectory_rewards" in new_online_data.keys()
    assert len(ddpm_dataset["observations"].shape) == 3
    assert len(new_online_data["observations"].shape) == 2

    assert reward_scale == 1.0
    successful_indices = np.where(
        new_online_data["max_trajectory_rewards"] >= 3.0 + reward_bias
    )[0]

    new_online_data = {
        key: value[successful_indices] for key, value in new_online_data.items()
    }
    new_online_data = preprocess_dataset_for_ddpm(new_online_data)

    # concatenate the new online data with the ddpm dataset
    ddpm_dataset = {
        key: np.concatenate([value, new_online_data[key]], axis=0)
        for key, value in ddpm_dataset.items()
    }

    return ddpm_dataset


def discard_online_data_filtering_function(
    ddpm_dataset: Dict[str, np.ndarray], *args, **kwargs
) -> Dict[str, np.ndarray]:
    return ddpm_dataset


def all_data_filtering_function(
    ddpm_dataset: Dict[str, np.ndarray],
    new_online_data: Dict[str, np.ndarray],
    *args,
    **kwargs,
) -> Dict[str, np.ndarray]:
    assert "max_trajectory_rewards" in new_online_data.keys()
    assert len(ddpm_dataset["observations"].shape) == 3
    assert len(new_online_data["observations"].shape) == 2

    new_online_data = preprocess_dataset_for_ddpm(new_online_data)

    # concatenate the new online data with the ddpm dataset
    ddpm_dataset = {
        key: np.concatenate([value, new_online_data[key]], axis=0)
        for key, value in ddpm_dataset.items()
    }

    return ddpm_dataset


ONLINE_DATA_FILTERING_FUNCTIONS = {
    "successes": successes_only_filtering_function,
    "discard": discard_online_data_filtering_function,
    "all": all_data_filtering_function,
    "kitchen_good_data": kitchen_good_data_filtering_function,
}


def plot_q_values_over_trajectory_time_step(
    trajectories: List[Dict[str, List[Union[np.ndarray, Dict[str, np.ndarray]]]]],
    critic_agent,
    sharding: jax.sharding.Sharding,
):
    trajectories = [trajectories[0]]  # only plot the first trajectory
    if isinstance(trajectories[0]["observation"][0], dict):
        observations = [
            {
                key: np.array([obs[key] for obs in trajectory["observation"]])
                for key in trajectory["observation"][0].keys()
            }
            for trajectory in trajectories
        ]
    else:
        observations = [
            shard_batch(jnp.array(trajectory["observation"]), sharding)
            for trajectory in trajectories
        ]

    actions = [
        shard_batch(jnp.array(trajectory["action"]), sharding)
        for trajectory in trajectories
    ]

    q_values = []
    for trajectory_index in range(len(trajectories)):
        q_values.append(
            critic_agent.forward_critic(
                observations[trajectory_index],
                actions[trajectory_index],
                jax.random.PRNGKey(0),
            ).mean(axis=0)
        )
    q_values = jnp.stack(q_values, axis=0).mean(axis=0)
    assert q_values.shape == (len(trajectories[0]["observation"]),)

    # Plot the q-values over the trajectory time step using seaborn, make it look nice
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plot = sns.lineplot(
        x=np.arange(len(q_values)),
        y=q_values,
        color="blue",
        linewidth=2.5,
    )
    plot.set_title("Q-values over trajectory time step")
    plot.set_xlabel("Time step")
    plot.set_ylabel("Q-value")

    return plot


def main(_):

    # if FLAGS.debug:
    #     import pdb

    #     pdb.set_trace()
    # jax.config.update("jax_disable_jit", True)
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    wandb.require("core")
    wandb_config = WandBLogger.get_default_config()
    wandb_experiment_name = FLAGS.wandb_experiment_name
    if FLAGS.train_on_separate_computer_mode != "single_computer":
        wandb_experiment_name = (
            wandb_experiment_name + f"_{FLAGS.train_on_separate_computer_mode}"
        )

    wandb_config.update(
        {
            "project": FLAGS.wandb_project_name,
            "exp_descriptor": wandb_experiment_name,
            "tag": None,
            "group": FLAGS.wandb_group,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.debug,
    )

    save_dir = tf.io.gfile.join(
        (
            os.path.abspath(FLAGS.config.save_dir)
            if "gs://" not in FLAGS.config.save_dir
            else FLAGS.config.save_dir
        ),
        wandb_logger.config.project,
        FLAGS.wandb_experiment_name,
        f"seed_{FLAGS.seed}",
    )

    if FLAGS.train_on_separate_computer_mode != "single_computer":
        tf.io.gfile.makedirs(
            tf.io.gfile.join(save_dir, "critic_checkpoints_from_agent_trainer")
        )
        tf.io.gfile.makedirs(
            tf.io.gfile.join(save_dir, "ddpm_checkpoints_from_agent_trainer")
        )

    goal_conditioned = FLAGS.config.goal_conditioned

    # Env and dataset
    environment_name = str(FLAGS.environment_name)

    if "kitchen" in environment_name or environment_name == "calvin":
        assert FLAGS.config.reward_bias == -4.0
        assert FLAGS.config.reward_scale == 1.0
    elif "antmaze" in environment_name:
        assert FLAGS.config.reward_bias == -1.0
        assert FLAGS.config.reward_scale == 1.0

    environment_action_space = None

    if environment_name == "calvin":
        dataset = get_calvin_tfrecord_dataset(
            tfrecord_regexp=FLAGS.config.calvin_tfrecord_regexp,
            **FLAGS.config.dataset_kwargs,
        )
        dataset_critic_iterator = dataset.iterator(
            batch_size=int(FLAGS.config.batch_size * FLAGS.mixing_ratio)
        )
        if FLAGS.ddpm_checkpoint_path != "":
            dataset_ddpm_iterator = dataset.iterator(
                batch_size=FLAGS.config.ddpm_agent_kwargs.batch_size // 2
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
        train_env = get_calvin_env(cfg=calvin_config)
        if FLAGS.num_parallel_envs > 1:
            num_parallel_envs = FLAGS.num_parallel_envs
            eval_env = gym.vector.AsyncVectorEnv(
                [
                    lambda: get_calvin_env(
                        goal_conditioned=goal_conditioned,
                        cfg=calvin_config,
                        fixed_reset_state=CALVIN_EVAL_RESET_STATE,
                    )
                    for _ in range(num_parallel_envs)
                ],
                context="forkserver",  # the default "fork" is incompatible with JAX
            )
        else:
            eval_env = gym.vector.SyncVectorEnv(
                [
                    lambda: get_calvin_env(
                        goal_conditioned=goal_conditioned,
                        cfg=calvin_config,
                    )
                ]
            )
        offline_dataset_size = None
    else:
        train_env = TruncationWrapper(
            gym.wrappers.TimeLimit(
                gym.make(environment_name),
                max_episode_steps=FLAGS.config.max_episode_steps,
            )
        )
        if FLAGS.debug or FLAGS.num_parallel_envs == 1:
            eval_env = gym.vector.SyncVectorEnv(
                [
                    lambda: gym.wrappers.TimeLimit(
                        gym.make(environment_name),
                        max_episode_steps=FLAGS.config.max_episode_steps,
                    )
                    for i in range(FLAGS.num_parallel_envs)
                ]
            )
        else:
            max_episode_steps = FLAGS.config.max_episode_steps
            eval_env = gym.vector.AsyncVectorEnv(
                [
                    lambda: gym.wrappers.TimeLimit(
                        gym.make(environment_name),
                        max_episode_steps=max_episode_steps,
                    )
                    for i in range(FLAGS.num_parallel_envs)
                ],
                context="forkserver",  # the default "fork" is incompatible with JAX
            )

        dataset = get_d4rl_dataset_with_mc_calculation(
            FLAGS.environment_name,
            reward_scale=FLAGS.config.reward_scale,
            reward_bias=FLAGS.config.reward_bias,
            clip_action=FLAGS.clip_action,
            gamma=FLAGS.config.critic_agent_kwargs.discount,
        )
        dataset["masks"] = 1.0 - dataset["dones"]
        del dataset["dones"]
        dataset["actions"] = np.clip(
            dataset["actions"], -FLAGS.clip_action, FLAGS.clip_action
        )
        # split dataset into train and validation
        # val_dataset = jax.tree_map(lambda x: x[: len(x) // 10], dataset)
        # dataset = jax.tree_map(lambda x: x[len(x) // 10 :], dataset)
        # ddpm_dataset = preprocess_dataset_for_ddpm(dataset)
        assert (
            FLAGS.retrain_ddpm_online_filtering_function == "all"
        ), "only 'all' is supported with this version"
        dataset_critic_iterator = None
        dataset_ddpm_iterator = None
        offline_dataset_size = len(dataset["actions"])

    if environment_action_space is None:
        environment_action_space = train_env.action_space
    print("environment_action_space", environment_action_space)
    critic_offline_batch_size = int(FLAGS.config.batch_size * FLAGS.mixing_ratio)
    critic_online_batch_size = FLAGS.config.batch_size - critic_offline_batch_size

    # replay buffer and sampler for online data
    if FLAGS.config.image_observations or environment_name == "calvin":
        tf.io.gfile.makedirs(tf.io.gfile.join(save_dir, "image_replay_buffer"))
        if (
            FLAGS.train_on_separate_computer_mode != "agent_training_only"
            and not FLAGS.resume_failed_run
        ):
            print(save_dir)
            assert not tf.io.gfile.exists(
                tf.io.gfile.join(save_dir, "image_replay_buffer", "episode_0.tfrecord")
            ), f"Image replay buffer already exists! ({tf.io.gfile.join(save_dir, 'image_replay_buffer', 'episode_0.tfrecord')})"
        image_replay_buffer = None
        image_replay_buffer_iterator_for_critic = None
        image_replay_buffer_iterator_for_ddpm = (
            None  # separate bc of different batch size
        )
        state_replay_buffer = None
    else:
        state_replay_buffer = ReplayBuffer(
            train_env.observation_space,
            environment_action_space,
            capacity=FLAGS.replay_buffer_capacity,
            # goal_space=finetune_env.observation_space if goal_conditioned else None,
            store_mc_return=True,
            store_max_trajectory_reward=True,
            seed=FLAGS.seed,
        )
        if FLAGS.resume_failed_run:
            replay_buffer_path = tf.io.gfile.join(save_dir, "replay_buffer.npz")
            if tf.io.gfile.exists(replay_buffer_path):
                with tf.io.gfile.GFile(replay_buffer_path, "rb") as f:
                    data_dict = dict(np.load(f))
                state_replay_buffer.batched_insert(data_dict)
            else:
                print(f"Replay buffer file {replay_buffer_path} not found!")

        image_replay_buffer = None
        image_replay_buffer_iterator_for_critic = None
        image_replay_buffer_iterator_for_ddpm = None
    train_sampler = TrajSampler(
        train_env,
        clip_action=FLAGS.clip_action,
        reward_scale=FLAGS.config.reward_scale,
        reward_bias=FLAGS.config.reward_bias,
        max_traj_length=FLAGS.config.max_episode_steps,
    )
    calc_mc_return_fn = lambda rewards, masks: calc_return_to_go(
        rewards,
        masks,
        FLAGS.config.critic_agent_kwargs.discount,
        push_failed_to_min=True if FLAGS.environment_name == "real_robot" else False,
        min_reward=FLAGS.config.reward_bias,
    )

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    rng = jax.random.PRNGKey(FLAGS.seed)
    # initialize DDPM agent
    latest_checkpoint_epoch = None
    is_pytorch_ddpm_agent = False
    is_openvla_agent = False
    is_transformer_agent = False
    is_cem_agent = False
    if FLAGS.ddpm_checkpoint_path != "":
        rng, construct_rng = jax.random.split(rng)
        example_ddpm_batch = get_batch_from_dataset(
            environment_name=FLAGS.environment_name,
            dataset=dataset,
            dataset_iterator=dataset_ddpm_iterator,
            batch_size=FLAGS.config.batch_size,
            preprocess_for_ddpm=True,
        )
        example_ddpm_batch = shard_batch(example_ddpm_batch, sharding)
        if FLAGS.ddpm_checkpoint_path.startswith("openvla"):
            from jaxrl_m.agents.continuous.openvla import OpenVLAAgent

            ddpm_agent_class = OpenVLAAgent
            FLAGS.config.ddpm_agent_kwargs["action_std"] = (
                FLAGS.bridgedata_config.action_proprio_metadata["action"]["std"]
            )
            is_openvla_agent = True
            is_pytorch_ddpm_agent = True
        elif FLAGS.ddpm_checkpoint_path.startswith("bc:"):
            from jaxrl_m.agents.continuous.bc import BCAgent

            ddpm_agent_class = BCAgent
            model_path = FLAGS.ddpm_checkpoint_path.replace("bc:", "")
            FLAGS.ddpm_checkpoint_path = model_path
            FLAGS.config.ddpm_agent_kwargs["agent_path"] = model_path
            is_pytorch_ddpm_agent = False

        elif FLAGS.ddpm_checkpoint_path.startswith("transformer:"):
            from jaxrl_m.agents.continuous.auto_regressive_transformer import (
                AutoRegressiveTransformerAgent,
            )

            ddpm_agent_class = AutoRegressiveTransformerAgent
            FLAGS.ddpm_checkpoint_path = FLAGS.ddpm_checkpoint_path.replace(
                "transformer:", ""
            )
            is_transformer_agent = True
        elif FLAGS.ddpm_checkpoint_path == "cem":
            from jaxrl_m.agents.continuous.cem_optimization import (
                CrossEntropyMethodOptimizationAgent,
            )

            ddpm_agent_class = CrossEntropyMethodOptimizationAgent
            is_cem_agent = True
            FLAGS.config.ddpm_agent_kwargs["action_space_low"] = (
                environment_action_space.low
            )
            FLAGS.config.ddpm_agent_kwargs["action_space_high"] = (
                environment_action_space.high
            )
        else:
            ddpm_agent_class = DDPMBCAgent

        if not FLAGS.config.image_observations:

            if is_transformer_agent or is_cem_agent:

                def ddpm_encoder_def(x, **kwargs):
                    if isinstance(x, dict):
                        return x["state"]
                    return x

                def critic_encoder_def(x, **kwargs):
                    if isinstance(x, dict):
                        return x["state"]
                    return x

            else:

                def ddpm_encoder_def(x, **kwargs):
                    if x.ndim == 3:
                        assert x.shape[1] == 1, x.shape
                        return x[:, 0]
                    return x

                # return x

                def critic_encoder_def(x, **kwargs):
                    return x

        else:
            ddpm_encoder_def = encoders[FLAGS.config.encoder](
                **FLAGS.config.encoder_kwargs
            )
            critic_encoder_def = encoders[FLAGS.config.encoder](
                **FLAGS.config.encoder_kwargs
            )

        if is_transformer_agent or is_cem_agent:
            ddpm_agent = ddpm_agent_class(
                rng=construct_rng,
                observations=example_ddpm_batch["observations"],
                goals=(
                    example_ddpm_batch["goals"]
                    if "goals" in example_ddpm_batch
                    else None
                ),
                actions=jax.tree_map(jnp.array, example_ddpm_batch["actions"]),
                encoder_def=ddpm_encoder_def,
                action_min=environment_action_space.low.min(),
                action_max=environment_action_space.high.max(),
                **FLAGS.config.ddpm_agent_kwargs,
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
                actions=jax.tree_map(jnp.array, example_ddpm_batch["actions"]),
                encoder_def=ddpm_encoder_def,
                action_min=environment_action_space.low.min(),
                action_max=environment_action_space.high.max(),
                **FLAGS.config.ddpm_agent_kwargs,
            )

        if FLAGS.ddpm_checkpoint_path.startswith("openvla"):
            if FLAGS.ddpm_checkpoint_path.startswith("openvla:"):
                model_path = FLAGS.ddpm_checkpoint_path.replace("openvla:", "")
                ddpm_agent.load_checkpoint(model_path)
            train_env.reward_function.openvla_agent = ddpm_agent

        if is_transformer_agent:
            ddpm_agent.restore_checkpoint(FLAGS.ddpm_checkpoint_path)
        elif not FLAGS.resume_failed_run and not (
            is_pytorch_ddpm_agent or is_cem_agent
        ):
            checkpoint_agent_path = FLAGS.ddpm_checkpoint_path
            if "bc" in checkpoint_agent_path:
                checkpoint_agent_path = checkpoint_agent_path.replace("bc:", "")
            ddpm_agent = orbax_checkpointer.restore(
                checkpoint_agent_path, item=ddpm_agent
            )
        elif FLAGS.resume_failed_run:
            if FLAGS.train_on_separate_computer_mode != "single_computer":
                checkpoint_paths = tf.io.gfile.glob(
                    tf.io.gfile.join(
                        save_dir,
                        "ddpm_checkpoints_from_agent_trainer",
                        "checkpoint_*",
                    )
                )
            else:
                checkpoint_paths = tf.io.gfile.glob(
                    tf.io.gfile.join(save_dir, "ddpm_checkpoints", "checkpoint_*")
                )
            if len(checkpoint_paths) == 0:
                logging.info(
                    "No checkpoints found from previous runs! Loading offline checkpoint"
                )
                ddpm_agent = orbax_checkpointer.restore(
                    FLAGS.ddpm_checkpoint_path, item=ddpm_agent
                )
                latest_checkpoint_epoch = -1
            else:
                checkpoint_epochs = [
                    int(path.split("_")[-1].split(".")[0]) for path in checkpoint_paths
                ]
                latest_checkpoint_epoch = max(checkpoint_epochs)
                latest_checkpoint_path = tf.io.gfile.join(
                    save_dir,
                    "ddpm_checkpoints_from_agent_trainer",
                    f"checkpoint_{latest_checkpoint_epoch}",
                )
                if not is_openvla_agent:
                    copy_gcs_dir_to_local(
                        latest_checkpoint_path, "./tmp_ddpm_checkpoint"
                    )
                    ddpm_agent = orbax_checkpointer.restore(
                        "./tmp_ddpm_checkpoint", item=ddpm_agent
                    )
                    shutil.rmtree("./tmp_ddpm_checkpoint")
                else:
                    ddpm_agent.load_checkpoint(latest_checkpoint_path)
        if not (is_pytorch_ddpm_agent or is_transformer_agent or is_cem_agent):
            ddpm_agent = jax.device_put(
                jax.tree_map(jnp.array, ddpm_agent), sharding.replicate()
            )

        if FLAGS.critic_agent_name in [
            "conservative_iql",
            "diffusion_cql",
            "mc_critic",
            "sarsa",
        ]:
            FLAGS.config.critic_agent_kwargs["ddpm_agent"] = (
                ddpm_agent if not (is_transformer_agent or is_cem_agent) else None
            )

        if FLAGS.critic_agent_name == "conservative_iql":
            FLAGS.config.critic_agent_kwargs["action_space"] = environment_action_space
        elif FLAGS.critic_agent_name in ["diffusion_cql", "mc_critic", "sarsa"]:
            FLAGS.config.critic_agent_kwargs["action_space_low"] = (
                environment_action_space.low
            )
            FLAGS.config.critic_agent_kwargs["action_space_high"] = (
                environment_action_space.high
            )
    elif FLAGS.config.get("use_metaworld_scripted_policy"):
        assert FLAGS.environment_name.startswith("metaworld-")
        from jaxrl_m.envs.metaworld import get_scripted_policy

        ddpm_agent = get_scripted_policy(environment_name.split("metaworld-")[-1])
    else:
        assert "diffusion" not in FLAGS.critic_agent_name
        ddpm_agent = None

    # initialize critic

    rng, construct_rng = jax.random.split(rng)
    example_critic_batch = get_batch_from_dataset(
        environment_name=FLAGS.environment_name,
        dataset=dataset,
        dataset_iterator=dataset_critic_iterator,
        batch_size=FLAGS.config.batch_size,
        preprocess_for_ddpm=False,
    )
    example_critic_batch = shard_batch(example_critic_batch, sharding)
    if is_openvla_agent:
        example_critic_batch["observations"]["image"] = resize_images_to_100x100(
            example_critic_batch["observations"]["image"]
        )
        example_critic_batch["next_observations"]["image"] = resize_images_to_100x100(
            example_critic_batch["next_observations"]["image"]
        )
    if FLAGS.config.get("bound_q_targets", False):
        min_q_target = dataset["rewards"].min() / (
            1 - FLAGS.config.critic_agent_kwargs.discount
        )
        max_q_target = dataset["rewards"].max() / (
            1 - FLAGS.config.critic_agent_kwargs.discount
        )
    else:
        min_q_target = None
        max_q_target = None
    critic_agent = agents[FLAGS.critic_agent_name].create(
        rng=construct_rng,
        observations=example_critic_batch["observations"],
        goals=(
            example_critic_batch["goals"] if "goals" in example_critic_batch else None
        ),
        actions=jax.tree_map(jnp.array, example_critic_batch["actions"]),
        encoder_def=critic_encoder_def,
        min_q_target=min_q_target,
        max_q_target=max_q_target,
        **FLAGS.config.critic_agent_kwargs,
    )
    if not FLAGS.resume_failed_run:
        if FLAGS.critic_checkpoint_path == "":
            assert "rlpd" in wandb_experiment_name
            logging.info("Not loading critic checkpoint!")
        else:
            critic_agent = orbax_checkpointer.restore(
                FLAGS.critic_checkpoint_path, item=critic_agent
            )
    else:
        assert latest_checkpoint_epoch is not None
        if latest_checkpoint_epoch == -1:
            critic_agent = orbax_checkpointer.restore(
                FLAGS.critic_checkpoint_path, item=critic_agent
            )
        else:
            if FLAGS.train_on_separate_computer_mode != "single_computer":
                latest_checkpoint_path = tf.io.gfile.join(
                    save_dir,
                    "critic_checkpoints_from_agent_trainer",
                    f"checkpoint_{latest_checkpoint_epoch}",
                )
            else:
                latest_checkpoint_path = tf.io.gfile.join(
                    save_dir,
                    "critic_checkpoints",
                    f"checkpoint_{latest_checkpoint_epoch}",
                )
            critic_agent = orbax_checkpointer.restore(
                latest_checkpoint_path, item=critic_agent
            )
    critic_agent = jax.device_put(
        jax.tree_map(jnp.array, critic_agent), sharding.replicate()
    )

    action_optimizer_state = None

    if ddpm_agent is not None:
        data_collection_q_diffusion_num_steps = (
            FLAGS.config.data_collection_q_diffusion_num_steps
            if FLAGS.config.data_collection_q_diffusion_num_steps >= 0
            else FLAGS.config.critic_agent_kwargs.q_diffusion_num_steps
        )
        data_collection_q_diffusion_step_size = (
            FLAGS.config.data_collection_q_diffusion_step_size
            if FLAGS.config.data_collection_q_diffusion_step_size >= 0
            else FLAGS.config.critic_agent_kwargs.q_diffusion_step_size
        )
    else:
        data_collection_q_diffusion_num_steps = None
        data_collection_q_diffusion_step_size = None

    timer = Timer()

    """training loop"""
    steps_elapsed = 0

    online_trajectories_added = 0
    if FLAGS.train_on_separate_computer_mode == "agent_training_only":
        online_trajectories_added = FLAGS.n_warmup_trajs
    online_env_steps = 0
    starting_epoch = 0
    if FLAGS.resume_failed_run:
        assert latest_checkpoint_epoch is not None
        online_trajectories_added = len(
            tf.io.gfile.glob(f"{save_dir}/image_replay_buffer/episode_*.tfrecord")
        )
        if state_replay_buffer is not None:
            online_env_steps = len(state_replay_buffer)
        else:
            online_env_steps = (
                online_trajectories_added * FLAGS.config.max_episode_steps
            )
        starting_epoch = latest_checkpoint_epoch
    elif FLAGS.environment_name != "real_robot":
        starting_epoch = -1  # To do first eval before any training
    for epoch in tqdm(range(starting_epoch, FLAGS.num_online_epochs)):
        print("Current epoch:", epoch)
        timer.tick("total_training_loop")

        with timer.context("env step"):
            rng, action_rng = jax.random.split(rng)

            info_metric_dicts = []
            if ddpm_agent is not None:

                def policy_fn(obs, seed, ddpm_agent=None):
                    nonlocal action_optimizer_state
                    assert FLAGS.config.data_collection_particle_choosing_strategy in [
                        "max_q_value",
                        "random_weighted",
                    ]
                    seed, action_rng = jax.random.split(seed)
                    if goal_conditioned:
                        assert "image" in obs and "goal" in obs and "proprio" in obs
                        obs = (
                            obs,
                            {
                                "image": obs["goal"],
                            },
                        )

                    if is_pytorch_ddpm_agent or is_transformer_agent or is_cem_agent:

                        ddpm_actions = ddpm_agent.sample_actions(
                            obs,
                            repeat=FLAGS.config.critic_agent_kwargs.num_ddpm_actions,
                            cache=False,
                            **({"timer": timer} if is_pytorch_ddpm_agent else {}),
                            **(
                                {"critic_agent": critic_agent, "seed": action_rng}
                                if is_cem_agent
                                else {}
                            ),
                        )
                        if not isinstance(obs, dict):
                            obs = {"state": obs, "ddpm_actions": ddpm_actions}
                        else:
                            obs["ddpm_actions"] = ddpm_actions

                    (
                        action_distribution,
                        info_metrics,
                        _,
                    ) = q_diffusion_sample_actions(
                        obs,
                        ddpm_agent=(
                            ddpm_agent
                            if not (
                                is_pytorch_ddpm_agent
                                or is_transformer_agent
                                or is_cem_agent
                            )
                            else 42.0
                        ),
                        critic_agent=critic_agent,
                        num_ddpm_actions=FLAGS.config.critic_agent_kwargs.num_ddpm_actions,
                        num_actions_to_keep_for_q_diffusion=FLAGS.config.critic_agent_kwargs.num_actions_to_keep_for_q_diffusion,
                        q_diffusion_num_steps=data_collection_q_diffusion_num_steps,
                        q_diffusion_step_size=data_collection_q_diffusion_step_size,
                        q_diffusion_optimize_critic_ensemble_min=FLAGS.config.critic_agent_kwargs.q_diffusion_optimize_critic_ensemble_min,
                        use_adam=FLAGS.config.critic_agent_kwargs.q_diffusion_use_adam,
                        rng=action_rng,
                        action_space_low=environment_action_space.low,
                        action_space_high=environment_action_space.high,
                        q_diffusion_adam_kwargs=FLAGS.config.critic_agent_kwargs.q_diffusion_adam_kwargs.to_dict(),
                        adam_state=action_optimizer_state,
                        argmax=FLAGS.config.data_collection_particle_choosing_strategy
                        == "max_q_value",
                        half_step_size_on_overshooting=FLAGS.config.critic_agent_kwargs.q_diffusion_half_step_size_on_overshooting,
                        overshooting_factor=FLAGS.config.critic_agent_kwargs.q_diffusion_overshooting_factor,
                    )
                    action = action_distribution.sample(seed=seed)
                    exploration_noise_std = (
                        FLAGS.config.critic_agent_kwargs.exploration_noise_std
                    )
                    if exploration_noise_std > 0:
                        exploration_noise_seed, seed = jax.random.split(seed)
                        action += (
                            jax.random.normal(exploration_noise_seed, action.shape)
                            * exploration_noise_std
                        )

                    nonlocal info_metric_dicts
                    if "adam_nu" in info_metrics:
                        del info_metrics["adam_nu"]
                    info_metric_dicts.append(info_metrics)
                    return jax.device_get(action.reshape((-1,)))

            else:

                def policy_fn(*args, **kwargs):
                    actions = critic_agent.sample_actions(*args, **kwargs)
                    return jax.device_get(actions)

            policy_fn = supply_rng(policy_fn, action_rng)

            policy_fn = partial(
                policy_fn,
                ddpm_agent=ddpm_agent,
            )

            def ingest_trajectories(
                trajectories,
                online_trajs_added,
                online_steps,
            ) -> Tuple[
                Iterable[Dict[str, Any]], Iterable[Dict[str, Any]], int, int, int
            ]:
                """Given trajectories from the environment, adds them to the image replay buffer (if
                using image observations), logs trajectory statistics, and counts elapsed steps.

                Returns:
                    - image_replay_buffer_iterator_for_critic
                    - image_replay_buffer_iterator_for_ddpm
                    - steps_elapsed
                    - online_trajectories_added
                    - online_env_steps
                """
                if FLAGS.config.image_observations:
                    for traj in trajectories:
                        should_save_trajectory = True
                        if FLAGS.replay_buffer_filtering_function == "successes":
                            assert FLAGS.config.reward_scale == 1.0
                            successful = (
                                traj["rewards"][-1] == 1 + FLAGS.config.reward_bias
                            )
                            should_save_trajectory = successful

                        if should_save_trajectory:
                            save_trajectory_as_tfrecord(
                                trajectory=traj,
                                path=tf.io.gfile.join(
                                    save_dir,
                                    "image_replay_buffer",
                                    f"episode_{online_trajs_added}.tfrecord",
                                ),
                            )
                            online_trajs_added += 1
                            online_steps += len(traj["rewards"])

                    # Recreate the image replay buffer
                    data_paths = glob_to_path_list(
                        tf.io.gfile.join(save_dir, "image_replay_buffer", "*.tfrecord")
                    )
                    timer.tick("create_image_replay_buffer")
                    image_replay_buffer = ImageReplayBuffer(
                        data_paths=data_paths,
                        seed=FLAGS.seed,
                        train=True,
                        **FLAGS.config.image_replay_buffer_kwargs,
                    )
                    image_replay_buffer_iterator_for_critic = (
                        image_replay_buffer.iterator(
                            batch_size=critic_online_batch_size
                        )
                    )
                    if ddpm_agent is not None and FLAGS.ddpm_utd != 0:
                        image_replay_buffer_iterator_for_ddpm = (
                            image_replay_buffer.iterator(
                                batch_size=(
                                    FLAGS.config.ddpm_agent_kwargs.batch_size // 2
                                    if offline_dataset_size is None
                                    else FLAGS.config.ddpm_agent_kwargs.batch_size
                                )
                            )
                        )
                    else:
                        image_replay_buffer_iterator_for_ddpm = None
                    timer.tock("create_image_replay_buffer")
                else:
                    image_replay_buffer_iterator_for_critic = None
                    image_replay_buffer_iterator_for_ddpm = None
                # Get trajectory statistics
                mean_trajectory_return = np.mean(
                    [np.sum(t["rewards"]) for t in trajectories]
                )
                mean_trajectory_length = np.mean(
                    [len(t["rewards"]) for t in trajectories]
                )
                mean_max_reward = np.mean([np.max(t["rewards"]) for t in trajectories])
                wandb_logger.log(
                    {
                        "train_env": {
                            "mean_trajectory_return": mean_trajectory_return,
                            "mean_trajectory_length": mean_trajectory_length,
                            "mean_max_reward": mean_max_reward,
                        },
                    },
                    step=max(epoch, 0),
                )
                steps_elapsed = np.sum([len(t["rewards"]) for t in trajectories])
                # Average info_metric_dicts and log
                nonlocal info_metric_dicts
                if len(info_metric_dicts) > 0:
                    info_metric_dicts = {
                        k: jnp.mean(
                            jnp.stack([d[k] for d in info_metric_dicts]), axis=-1
                        )
                        for k in info_metric_dicts[0].keys()
                    }
                    info_metric_dicts = jax.device_get(info_metric_dicts)

                    wandb_logger.log(
                        {
                            "train_env": {
                                "policy_info_metrics": {
                                    # if it's an array, histogram it
                                    k: wandb.Histogram(v) if np.prod(v.shape) > 1 else v
                                    for k, v in info_metric_dicts.items()
                                }
                            },
                        },
                        step=max(epoch, 0),
                    )
                return (
                    image_replay_buffer_iterator_for_critic,
                    image_replay_buffer_iterator_for_ddpm,
                    steps_elapsed,
                    online_trajs_added,
                    online_steps,
                )

            if FLAGS.collect_data_in_separate_process:
                assert FLAGS.train_on_separate_computer_mode == "single_computer"
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                trajectory_collection_future = executor.submit(
                    train_sampler.sample,
                    policy_fn,
                    FLAGS.n_online_trajs_per_epoch if epoch else FLAGS.n_warmup_trajs,
                    replay_buffer=state_replay_buffer,
                    calc_mc_return_fn=calc_mc_return_fn,
                    store_max_trajectory_reward=True,
                    terminate_on_success=FLAGS.config.early_terminate_on_success,
                )
            elif (
                FLAGS.train_on_separate_computer_mode == "single_computer"
                or FLAGS.train_on_separate_computer_mode == "env_steps_only"
            ):  # Don't collect if mode is agent_training_only
                num_trajs_to_collect = (
                    FLAGS.n_online_trajs_per_epoch if epoch else FLAGS.n_warmup_trajs
                )
                trajs = []
                for traj_index in range(num_trajs_to_collect):
                    trajs.append(
                        train_sampler.sample(
                            policy_fn,
                            1,
                            replay_buffer=state_replay_buffer,
                            calc_mc_return_fn=calc_mc_return_fn,
                            store_max_trajectory_reward=True,
                            terminate_on_success=FLAGS.config.early_terminate_on_success,
                        )[0]
                    )
                    if (
                        is_openvla_agent
                        and "-" in FLAGS.real_robot_task_name_for_reward
                    ):
                        # Need to reset the reward function after every episode
                        # because some stages might have been solved
                        train_env.reward_function.reset()
                (
                    image_replay_buffer_iterator_for_critic,
                    image_replay_buffer_iterator_for_ddpm,
                    steps_elapsed,
                    online_trajectories_added,
                    online_env_steps,
                ) = ingest_trajectories(
                    trajs, online_trajectories_added, online_env_steps
                )

        with timer.context("ddpm retraining"):
            if (
                ddpm_agent is not None
                and FLAGS.retrain_ddpm_from_scratch
                and FLAGS.retrain_ddpm_interval > 0
                and (epoch + 1) % FLAGS.retrain_ddpm_interval == 0
                and FLAGS.train_on_separate_computer_mode != "env_steps_only"
            ):
                logging.info(f"Retraining DDPM at epoch {epoch}")
                # Set save dir to the same as the finetuning path
                FLAGS.ddpm_retrain_config.save_dir = tf.io.gfile.join(
                    save_dir, f"ddpm_retrain_epoch_{epoch}"
                )
                FLAGS.ddpm_retrain_config.save_interval = FLAGS.retrain_ddpm_num_epochs
                FLAGS.ddpm_retrain_config.num_eval_episodes = 0

                train_agent(
                    config=FLAGS.ddpm_retrain_config,
                    train_dataset=ddpm_dataset,
                    eval_env=eval_env.env_fns[0](),
                    num_epochs=FLAGS.retrain_ddpm_num_epochs,
                    num_train_steps_per_epoch=1000,
                    reward_scale=FLAGS.config.reward_scale,
                    reward_bias=FLAGS.config.reward_bias,
                    clip_action=FLAGS.clip_action,
                    preprocess_dataset_with_q_diffusion=True,
                    q_diffusion_agent=critic_agent,
                    q_diffusion_n_steps=FLAGS.config.critic_agent_kwargs.q_diffusion_num_steps,
                    q_diffusion_step_size=FLAGS.config.critic_agent_kwargs.q_diffusion_step_size,
                )
                # Reload the retrained model
                ddpm_ckpt = path = tf.io.gfile.join(
                    FLAGS.ddpm_retrain_config.save_dir,
                    f"checkpoint_{FLAGS.retrain_ddpm_num_epochs}",
                )
                copy_gcs_dir_to_local(ddpm_ckpt, "./tmp_ddpm_ckpt")
                ddpm_agent = orbax_checkpointer.restore(
                    "./tmp_ddpm_ckpt",
                    item=ddpm_agent,
                )
                shutil.rmtree("./tmp_ddpm_ckpt")
                ddpm_agent = jax.device_put(
                    jax.tree_map(jnp.array, ddpm_agent), sharding.replicate()
                )
            if (
                ddpm_agent is not None
                and not FLAGS.retrain_ddpm_from_scratch
                and FLAGS.train_on_separate_computer_mode != "env_steps_only"
                and epoch > -1
            ):
                n_updates = int(steps_elapsed * FLAGS.ddpm_utd)
                print("n_updates {} in epoch {}".format(n_updates, epoch))
                for update_index in tqdm(range(n_updates), desc="DDPM update"):
                    timer.tick("ddpm_get_batch")
                    if offline_dataset_size is None:
                        offline_ddpm_batch_size = (
                            FLAGS.config.ddpm_agent_kwargs.batch_size // 2
                        )
                    else:
                        total_data_size = offline_dataset_size + online_env_steps
                        offline_ddpm_batch_size = int(
                            (offline_dataset_size / total_data_size)
                            * FLAGS.config.ddpm_agent_kwargs.batch_size
                        )
                    if FLAGS.mixing_ratio > 0:
                        offline_batch = get_batch_from_dataset(
                            environment_name=FLAGS.environment_name,
                            dataset=dataset,
                            dataset_iterator=dataset_ddpm_iterator,
                            batch_size=offline_ddpm_batch_size,
                        )
                    else:
                        offline_batch = None
                        offline_ddpm_batch_size = 0
                    if FLAGS.config.image_observations:
                        online_batch = next(image_replay_buffer_iterator_for_ddpm)
                        online_batch = subsample_batch(
                            online_batch,
                            FLAGS.config.ddpm_agent_kwargs.batch_size
                            - offline_ddpm_batch_size,
                        )
                    else:
                        online_batch = state_replay_buffer.sample(
                            FLAGS.config.ddpm_agent_kwargs.batch_size
                            - offline_ddpm_batch_size
                        ).unfreeze()
                    if offline_batch is not None:
                        batch = concatenate_batches([offline_batch, online_batch])
                    else:
                        batch = online_batch
                    assert batch["rewards"].shape == (
                        FLAGS.config.ddpm_agent_kwargs.batch_size,
                    )
                    if FLAGS.environment_name == "real_robot":
                        # Clip actions to action space
                        batch["actions"] = np.clip(
                            batch["actions"],
                            environment_action_space.low,
                            environment_action_space.high,
                        )

                    if is_pytorch_ddpm_agent or is_transformer_agent or is_cem_agent:
                        # It's a pytorch agent -- precompute its actions
                        cache_dir = tf.io.gfile.join(
                            save_dir,
                            "ddpm_checkpoints_from_agent_trainer",
                            f"checkpoint_{epoch-1}",
                            "openvla_cache",
                        )
                        if is_transformer_agent or is_cem_agent:
                            batch = shard_batch(batch, sharding)
                        # batch["observations"]["ddpm_actions"] = (
                        rng, key = jax.random.split(rng)
                        ddpm_actions = ddpm_agent.sample_actions(
                            batch["observations"],
                            repeat=FLAGS.config.critic_agent_kwargs.num_ddpm_actions,
                            **({"cache_dir": cache_dir} if is_openvla_agent else {}),
                            **({"timer": timer} if is_pytorch_ddpm_agent else {}),
                            **(
                                {"critic_agent": critic_agent, "seed": key}
                                if is_cem_agent
                                else {}
                            ),
                        )
                        if not isinstance(batch["observations"], dict):
                            batch["observations"] = {
                                "state": batch["observations"],
                                "ddpm_actions": ddpm_actions,
                            }
                        else:
                            batch["observations"]["ddpm_actions"] = ddpm_actions
                        if is_openvla_agent:
                            ddpm_agent.prepare_for_finetuning()
                    batch = preprocess_dataset_for_ddpm(batch)

                    batch = shard_batch(batch, sharding)
                    timer.tock("ddpm_get_batch")
                    timer.tick("preprocess_with_q_diffusion")
                    q_diffusion_num_steps_for_ddpm_training = (
                        FLAGS.config.critic_agent_kwargs.q_diffusion_num_steps
                    )
                    if FLAGS.config.q_diffusion_num_steps_for_ddpm_training >= 0:
                        q_diffusion_num_steps_for_ddpm_training = (
                            FLAGS.config.q_diffusion_num_steps_for_ddpm_training
                        )
                    q_diffusion_step_size_for_ddpm_training = (
                        FLAGS.config.critic_agent_kwargs.q_diffusion_step_size
                        if FLAGS.config.q_diffusion_step_size_for_ddpm_training < 0
                        else FLAGS.config.q_diffusion_step_size_for_ddpm_training
                    )
                    rng, key = jax.random.split(rng)
                    batch = preprocess_batch_with_q_diffusion(
                        batch,
                        critic_agent,
                        n_steps=q_diffusion_num_steps_for_ddpm_training,
                        step_size=q_diffusion_step_size_for_ddpm_training,
                        optimize_critic_ensemble_min=FLAGS.config.critic_agent_kwargs.q_diffusion_optimize_critic_ensemble_min,
                        use_adam=FLAGS.config.critic_agent_kwargs.q_diffusion_use_adam,
                        adam_kwargs=FLAGS.config.critic_agent_kwargs.q_diffusion_adam_kwargs.to_dict(),
                        action_optimizer_state=action_optimizer_state,
                        action_space_low=environment_action_space.low,
                        action_space_high=environment_action_space.high,
                        half_step_size_on_overshooting=FLAGS.config.critic_agent_kwargs.q_diffusion_half_step_size_on_overshooting,
                        overshooting_factor=FLAGS.config.critic_agent_kwargs.q_diffusion_overshooting_factor,
                        improve_actions_with_global_search=FLAGS.config.get(
                            "improve_ddpm_actions_with_global_search", False
                        ),
                        ddpm_agent=(
                            ddpm_agent
                            if not (
                                is_pytorch_ddpm_agent
                                or is_transformer_agent
                                or is_cem_agent
                            )
                            else 42.0
                        ),
                        num_ddpm_actions=FLAGS.config.critic_agent_kwargs.num_ddpm_actions,
                        num_actions_to_keep_for_q_diffusion=FLAGS.config.critic_agent_kwargs.num_actions_to_keep_for_q_diffusion,
                        distill_argmax=FLAGS.config.distill_argmax,
                        rng=key,
                    )
                    timer.tock("preprocess_with_q_diffusion")
                    ddpm_loss_weights = None
                    if FLAGS.config.get("weigh_ddpm_loss_by_advantage", False):
                        timer.tick("calculate_ddpm_advantage_weights")
                        # Calculate weights for DDPM loss

                        rng, key = jax.random.split(rng)
                        assert (
                            batch["observations"].ndim == 3
                            and batch["observations"].shape[1] == 1
                        )
                        assert (
                            batch["actions"].ndim == 3
                            and batch["actions"].shape[1] == 1
                        )
                        dataset_action_values = critic_agent.forward_critic(
                            batch["observations"][:, 0],
                            batch["actions"][:, 0],
                            rng=key,
                            train=False,
                        ).mean(axis=0)
                        rng, key = jax.random.split(rng)
                        policy_actions = ddpm_agent.sample_actions(
                            batch["observations"],
                            seed=key,
                        )[:, 0]
                        policy_action_values = critic_agent.forward_critic(
                            batch["observations"][:, 0],
                            policy_actions,
                            rng=key,
                            train=False,
                        ).mean(axis=0)
                        advantages = dataset_action_values - policy_action_values
                        ddpm_loss_weights = jnp.exp(
                            FLAGS.config.ddpm_advantage_weight_temperature * advantages
                        )
                        timer.tock("calculate_ddpm_advantage_weights")
                    (
                        ddpm_agent,
                        ddpm_update_info,
                    ) = ddpm_agent.update(
                        batch,
                        **(
                            {
                                "timer": timer,
                            }
                            if is_pytorch_ddpm_agent
                            else {}
                        ),
                        loss_weights=ddpm_loss_weights,
                    )
                    if update_index == 0:
                        ddpm_update_info = jax.device_get(ddpm_update_info)
                        wandb_logger.log(
                            {
                                "ddpm_update": ddpm_update_info,
                            },
                            step=epoch,
                        )
            if getattr(ddpm_agent, "pytorch", False):
                ddpm_agent.prepare_for_inference()
                if hasattr(ddpm_agent, "clear_cache") and epoch % 10 == 0:
                    ddpm_agent.clear_cache()
            elif (
                ddpm_agent is not None
                and FLAGS.critic_agent_name != "iql"
                and not (is_transformer_agent or is_cem_agent)
            ):
                critic_agent = critic_agent.replace(ddpm_agent=ddpm_agent)

            if FLAGS.train_on_separate_computer_mode == "agent_training_only":
                # Save ddpm checkpoint for this epoch
                ddpm_path = tf.io.gfile.join(
                    save_dir,
                    "ddpm_checkpoints_from_agent_trainer",
                )
                if not getattr(ddpm_agent, "pytorch", False):
                    checkpoints.save_checkpoint(
                        ddpm_path, ddpm_agent, step=epoch, keep=2, overwrite=True
                    )
                else:
                    ddpm_path = os.path.join(ddpm_path, f"checkpoint_{epoch}")
                    ddpm_agent.save_checkpoint(ddpm_path)

        with timer.context("train critic"):
            n_updates = int(steps_elapsed * FLAGS.utd)
            # n_updates = int(steps_elapsed)
            if FLAGS.train_on_separate_computer_mode == "env_steps_only" or epoch < 0:
                n_updates = 0
            for update_index in tqdm(range(n_updates), desc="critic_updates"):

                if is_openvla_agent:
                    timer.tick("openvla_load_cache_from_filesystem")
                    cache_path = tf.io.gfile.join(
                        save_dir,
                        "ddpm_checkpoints_from_agent_trainer",
                        f"checkpoint_{epoch}",
                        "openvla_cache",
                    )
                    ddpm_agent.load_cache_from_filesystem(cache_path)
                    timer.tock("openvla_load_cache_from_filesystem")

                timer.tick("critic_get_batch")
                if FLAGS.mixing_ratio > 0:
                    offline_batch = get_batch_from_dataset(
                        environment_name=FLAGS.environment_name,
                        dataset=dataset,
                        dataset_iterator=dataset_critic_iterator,
                        batch_size=int(FLAGS.config.batch_size * FLAGS.mixing_ratio),
                    )
                else:
                    offline_batch = None
                if FLAGS.config.image_observations:
                    online_batch = next(image_replay_buffer_iterator_for_critic)
                else:
                    online_batch = state_replay_buffer.sample(
                        critic_online_batch_size
                    ).unfreeze()
                timer.tick("critic_get_batch_mask_setting")
                if (
                    "antmaze" in FLAGS.environment_name
                    or FLAGS.environment_name == "real_robot"
                ):
                    online_batch["masks"] = (
                        online_batch["rewards"] != (1 + FLAGS.config.reward_bias)
                    ).astype(np.float32)
                elif (
                    "kitchen" in FLAGS.environment_name
                    or FLAGS.environment_name == "calvin"
                ):
                    online_batch["masks"] = (
                        online_batch["rewards"] != (4 + FLAGS.config.reward_bias)
                    ).astype(np.float32)
                timer.tock("critic_get_batch_mask_setting")
                if offline_batch is not None:
                    batch = concatenate_batches([offline_batch, online_batch])
                else:
                    batch = online_batch
                assert batch["rewards"].shape == (FLAGS.config.batch_size,)
                # Masks checks
                timer.tick("critic_get_batch_masks_checks")
                if (
                    "antmaze" in FLAGS.environment_name
                    or FLAGS.environment_name == "real_robot"
                ):
                    assert np.all(
                        batch["masks"]
                        == (batch["rewards"] != (1 + FLAGS.config.reward_bias))
                    )
                elif (
                    "kitchen" in FLAGS.environment_name
                    or FLAGS.environment_name == "calvin"
                ):
                    assert np.all(
                        np.logical_or(
                            batch["masks"] == 1,
                            batch["rewards"] == (4 + FLAGS.config.reward_bias),
                        )
                    )
                timer.tock("critic_get_batch_masks_checks")

                if FLAGS.environment_name == "real_robot":
                    # Clip actions to action space
                    batch["actions"] = np.clip(
                        batch["actions"],
                        environment_action_space.low,
                        environment_action_space.high,
                    )

                if is_pytorch_ddpm_agent or is_transformer_agent or is_cem_agent:
                    cache_dir = tf.io.gfile.join(
                        save_dir,
                        "ddpm_checkpoints_from_agent_trainer",
                        f"checkpoint_{epoch}",
                        "ddpm_cache",
                    )
                    if is_transformer_agent or is_cem_agent:
                        batch = shard_batch(batch, sharding)
                    rng, key = jax.random.split(rng)
                    ddpm_actions = ddpm_agent.sample_actions(
                        batch["observations"],
                        repeat=FLAGS.config.critic_agent_kwargs.num_ddpm_actions,
                        **(
                            {"timer": timer, "cache_dir": cache_dir}
                            if is_openvla_agent
                            else {}
                        ),
                        **(
                            {"critic_agent": critic_agent, "seed": key}
                            if is_cem_agent
                            else {}
                        ),
                    )
                    rng, key = jax.random.split(rng)
                    ddpm_next_actions = ddpm_agent.sample_actions(
                        batch["next_observations"],
                        repeat=FLAGS.config.critic_agent_kwargs.num_ddpm_actions,
                        **(
                            {"timer": timer, "cache_dir": cache_dir}
                            if is_openvla_agent
                            else {}
                        ),
                        **(
                            {"critic_agent": critic_agent, "seed": key}
                            if is_cem_agent
                            else {}
                        ),
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

                batch = shard_batch(batch, sharding)
                if is_openvla_agent:
                    batch["observations"]["image"] = resize_images_to_100x100(
                        batch["observations"]["image"]
                    )
                    batch["next_observations"]["image"] = resize_images_to_100x100(
                        batch["next_observations"]["image"]
                    )
                timer.tock("critic_get_batch")
                if FLAGS.critic_agent_name in ["diffusion_cql", "mc_critic", "sarsa"]:
                    (
                        critic_agent,
                        update_info,
                        action_optimizer_state,
                    ) = critic_agent.update(
                        batch,
                        action_optimizer_state=action_optimizer_state,
                    )
                else:
                    critic_agent, update_info = critic_agent.update(
                        batch,
                    )
                if update_index == 0:
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
                                "mc_returns": batch["mc_returns"],
                            }
                        )
                    try:
                        wandb_logger.log(
                            {"training": update_info, "batch_info": batch_info},
                            step=epoch,
                        )
                    except ValueError:
                        print("Failed to log to wandb", update_info, batch_info)
                        breakpoint()

        # After having done the training, if we are collecting data in a separate process, we need to
        # ingest the new collected trajectories
        if FLAGS.collect_data_in_separate_process and epoch > -1:
            trajs = trajectory_collection_future.result()
            (
                image_replay_buffer_iterator_for_critic,
                image_replay_buffer_iterator_for_ddpm,
                steps_elapsed,
                online_trajectories_added,
                online_env_steps,
            ) = ingest_trajectories(trajs, online_trajectories_added, online_env_steps)

        with timer.context("evaluation"):
            if (
                (epoch + 1) % FLAGS.eval_interval == 0 or epoch == -1
            ) and FLAGS.environment_name != "real_robot":
                logging.info("Evaluating policy with Q-diffusion")
                rng, eval_rng = jax.random.split(rng)

                if ddpm_agent is None:

                    def policy_fn(*args, **kwargs):
                        actions = critic_agent.sample_actions(*args, **kwargs)
                        return jax.device_get(actions)

                else:

                    def policy_fn(obs, seed, **kwargs):
                        if goal_conditioned:
                            assert "image" in obs and "goal" in obs and "proprio" in obs
                            obs = (
                                obs,
                                {
                                    "image": obs["goal"],
                                },
                            )
                        elif not FLAGS.config.image_observations:
                            obs = obs.reshape((-1, obs.shape[-1]))

                        if is_transformer_agent or is_cem_agent:
                            if not isinstance(obs, dict):
                                obs = {"state": obs}
                            seed, key = jax.random.split(seed)
                            obs["ddpm_actions"] = ddpm_agent.sample_actions(
                                obs,
                                repeat=FLAGS.config.critic_agent_kwargs.num_ddpm_actions,
                                **(
                                    {"critic_agent": critic_agent, "seed": key}
                                    if is_cem_agent
                                    else {}
                                ),
                            )

                        nonlocal action_optimizer_state
                        seed, action_rng = jax.random.split(seed)
                        (
                            action_distribution,
                            info_metrics,
                            _,
                        ) = q_diffusion_sample_actions(
                            obs,
                            ddpm_agent=(
                                ddpm_agent
                                if not (is_transformer_agent or is_cem_agent)
                                else 42.0
                            ),
                            critic_agent=critic_agent,
                            num_ddpm_actions=FLAGS.config.critic_agent_kwargs.num_ddpm_actions,
                            num_actions_to_keep_for_q_diffusion=FLAGS.config.critic_agent_kwargs.num_actions_to_keep_for_q_diffusion,
                            q_diffusion_num_steps=data_collection_q_diffusion_num_steps,
                            q_diffusion_step_size=data_collection_q_diffusion_step_size,
                            q_diffusion_optimize_critic_ensemble_min=FLAGS.config.critic_agent_kwargs.q_diffusion_optimize_critic_ensemble_min,
                            use_adam=FLAGS.config.critic_agent_kwargs.q_diffusion_use_adam,
                            rng=action_rng,
                            action_space_low=environment_action_space.low,
                            action_space_high=environment_action_space.high,
                            q_diffusion_adam_kwargs=FLAGS.config.critic_agent_kwargs.q_diffusion_adam_kwargs.to_dict(),
                            adam_state=action_optimizer_state,
                            argmax=FLAGS.config.evaluation_particle_choosing_strategy
                            == "max_q_value",
                            half_step_size_on_overshooting=FLAGS.config.critic_agent_kwargs.q_diffusion_half_step_size_on_overshooting,
                            overshooting_factor=FLAGS.config.critic_agent_kwargs.q_diffusion_overshooting_factor,
                        )
                        action = action_distribution.sample(seed=seed)
                        return jax.device_get(action)

                policy_fn = supply_rng(policy_fn, eval_rng)
                trajectories = evaluate_with_trajectories_vectorized(
                    policy_fn,
                    eval_env,
                    num_episodes=FLAGS.config.num_eval_episodes,
                    save_video=(FLAGS.save_video and environment_name != "calvin"),
                )

                if FLAGS.save_video:
                    if environment_name == "calvin":
                        trajectories_to_save = trajectories[
                            : FLAGS.config.num_episodes_per_video
                        ]
                    else:
                        trajectories_to_save = [
                            t for t in trajectories if "image" in t
                        ][: FLAGS.config.num_episodes_per_video]
                    frames = []
                    for traj in trajectories_to_save:
                        trajectory_images = []
                        if environment_name == "calvin":
                            for transition in traj["observation"]:
                                image = transition["image"]
                                assert image.shape[-1] == 3
                                if len(image.shape) == 4:
                                    image = image[0]

                                if goal_conditioned:
                                    goal_image = transition["image_goal"]
                                    image = np.concatenate([image, goal_image], axis=1)
                                trajectory_images.append(image)
                            trajectory_images = np.stack(trajectory_images, axis=0)
                        else:
                            trajectory_images = np.stack(traj["image"], axis=0)
                            del traj["image"]

                        trajectory_return = 0
                        for i, reward in enumerate(traj["reward"]):
                            trajectory_return += reward
                            frames.append(
                                cv2.putText(
                                    trajectory_images[i],
                                    f"rew: {reward}. ret: {trajectory_return}",
                                    (10, 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3,
                                    (255, 255, 255),
                                    1,
                                ).transpose(2, 0, 1)
                            )
                    frames = np.array(frames)
                    wandb.log(
                        {
                            "video": wandb.Video(
                                frames,
                                fps=24,
                                format="mp4",
                            )
                        },
                        step=max(epoch, 0),
                    )

                q_values_over_trajectory_time_step_figure = (
                    plot_q_values_over_trajectory_time_step(
                        trajectories=trajectories,
                        critic_agent=critic_agent,
                        sharding=sharding,
                    )
                )
                wandb.log(
                    {
                        "q_values_over_trajectory_time_step": q_values_over_trajectory_time_step_figure
                    },
                    step=max(epoch, 0),
                )

                eval_metrics = {
                    "eval/average_return": np.mean(
                        [np.sum(t["reward"]) for t in trajectories]
                    ),
                    "eval/average_episode_length": np.mean(
                        [len(t["reward"]) for t in trajectories]
                    ),
                    "eval/average_max_reward": np.mean(
                        [np.max(t["reward"]) for t in trajectories]
                    ),
                }
                wandb_logger.log(eval_metrics, step=max(epoch, 0))

        with timer.context("save"):
            if (epoch + 1) % FLAGS.save_interval == 0:
                if FLAGS.save_model:
                    logging.info("Saving critic model")
                    critic_checkpoint_path = tf.io.gfile.join(
                        save_dir, "critic_checkpoints"
                    )
                    checkpoint_path = checkpoints.save_checkpoint(
                        critic_checkpoint_path,
                        critic_agent,
                        step=epoch,
                        keep=200,
                        overwrite=True,
                    )
                    if (
                        ddpm_agent is not None
                        and not FLAGS.config.get("use_metaworld_scripted_policy")
                        and not is_pytorch_ddpm_agent
                    ):
                        logging.info("Saving ddpm model")
                        ddpm_checkpoint_path = tf.io.gfile.join(
                            save_dir, "ddpm_checkpoints"
                        )
                        checkpoint_path = checkpoints.save_checkpoint(
                            ddpm_checkpoint_path,
                            ddpm_agent,
                            step=epoch,
                            keep=200,
                            overwrite=True,
                        )
                if FLAGS.save_replay_buffer:
                    logging.info("Saving replay buffer")
                    state_replay_buffer.save(save_dir)

        timer.tock("total_training_loop")
        wandb_logger.log(
            {"timer/total_times": timer.get_total_times(reset=False)}, step=epoch
        )
        wandb_logger.log({"timer/average_times": timer.get_average_times()}, step=epoch)

        if FLAGS.train_on_separate_computer_mode != "single_computer":
            # Wait until peer is done with their part.
            if FLAGS.train_on_separate_computer_mode == "env_steps_only":
                # Save steps_elapsed to a file.
                assert steps_elapsed > 0
                with tf.io.gfile.GFile(
                    os.path.join(save_dir, "latest_steps_elapsed.txt"), "w"
                ) as f:
                    f.write(str(steps_elapsed))
                # Wait until checkpoint for this epoch is available, and load it.
                restored_checkpoint = False
                while not restored_checkpoint:
                    # "checkpoint_{epoch}"
                    critic_path = tf.io.gfile.join(
                        save_dir,
                        "critic_checkpoints_from_agent_trainer",
                        f"checkpoint_{epoch}",
                    )
                    ddpm_path = tf.io.gfile.join(
                        save_dir,
                        "ddpm_checkpoints_from_agent_trainer",
                        f"checkpoint_{epoch}",
                    )

                    if tf.io.gfile.exists(critic_path) and tf.io.gfile.exists(
                        ddpm_path
                    ):
                        # move critic and agent into temporary local file
                        copy_gcs_dir_to_local(critic_path, "./tmp_critic")
                        critic_agent = orbax_checkpointer.restore(
                            "./tmp_critic", item=critic_agent
                        )
                        shutil.rmtree("./tmp_critic")
                        critic_agent = jax.device_put(
                            jax.tree_map(jnp.array, critic_agent), sharding.replicate()
                        )
                        if not getattr(ddpm_agent, "pytorch", False):
                            copy_gcs_dir_to_local(ddpm_path, "./tmp_ddpm")
                            ddpm_agent = orbax_checkpointer.restore(
                                "./tmp_ddpm", item=ddpm_agent
                            )
                            shutil.rmtree("./tmp_ddpm")
                            ddpm_agent = jax.device_put(
                                jax.tree_map(jnp.array, ddpm_agent),
                                sharding.replicate(),
                            )
                        else:
                            ddpm_agent.load_checkpoint(ddpm_path)

                        restored_checkpoint = True

                    else:
                        print(
                            f"Waiting for the checkpoint for epoch {epoch} to be available."
                        )
                        time.sleep(0.5)
                        continue
            elif FLAGS.train_on_separate_computer_mode == "agent_training_only":
                # Save the checkpoint for this epoch.
                critic_path = tf.io.gfile.join(
                    save_dir,
                    "critic_checkpoints_from_agent_trainer",
                )

                checkpoints.save_checkpoint(
                    critic_path, critic_agent, step=epoch, keep=3, overwrite=True
                )

                # Wait until newest episode appears on the replay buffer.
                newest_episode_exists = False
                while not newest_episode_exists:
                    if tf.io.gfile.exists(
                        os.path.join(
                            save_dir,
                            "image_replay_buffer",
                            f"episode_{online_trajectories_added - 1}.tfrecord",
                        )
                    ):
                        newest_episode_exists = True
                        online_trajectories_added += FLAGS.n_online_trajs_per_epoch
                    else:
                        print(
                            f"Waiting for episode_{online_trajectories_added-1}.tfrecord to be available."
                        )
                        time.sleep(0.5)

                # Recreate the image replay buffer with new episode
                data_paths = glob_to_path_list(
                    tf.io.gfile.join(save_dir, "image_replay_buffer", "*.tfrecord")
                )
                timer.tick("create_image_replay_buffer")
                image_replay_buffer = ImageReplayBuffer(
                    data_paths=data_paths,
                    seed=FLAGS.seed,
                    train=True,
                    **FLAGS.config.image_replay_buffer_kwargs,
                )
                image_replay_buffer_iterator_for_critic = image_replay_buffer.iterator(
                    batch_size=critic_online_batch_size
                )
                image_replay_buffer_iterator_for_ddpm = image_replay_buffer.iterator(
                    batch_size=(FLAGS.config.ddpm_agent_kwargs.batch_size)
                )
                timer.tock("create_image_replay_buffer")

                # Wait until the file exists
                steps_elapsed_file_path = os.path.join(
                    save_dir, "latest_steps_elapsed.txt"
                )
                while not tf.io.gfile.exists(steps_elapsed_file_path):
                    print(f"Waiting for {steps_elapsed_file_path} to be created...")
                    time.sleep(1)  # Wait for 1 second before checking again

                print(
                    f"{steps_elapsed_file_path} has been created. Reading the file..."
                )

                # Load steps_elapsed from file, add it to online_env_steps.
                with tf.io.gfile.GFile(
                    os.path.join(save_dir, "latest_steps_elapsed.txt"), "r"
                ) as f:
                    steps_elapsed = int(f.read())
                online_env_steps += steps_elapsed

            else:
                raise ValueError(
                    f"Invalid train_on_separate_computer_mode: {FLAGS.train_on_separate_computer_mode}"
                )


if __name__ == "__main__":
    app.run(main)
