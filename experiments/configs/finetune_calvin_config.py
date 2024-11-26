import os
import tensorflow as tf
import ml_collections
from jaxrl_m.agents.continuous.cql import (
    get_default_config as get_continuous_cql_config,
)

SAVE_DIR_PREFIX = os.environ.get("SAVE_DIR_PREFIX", "./")


def get_config(config_string):
    possible_structures = {
        "non_gc_q-diffusion_cql": ml_collections.ConfigDict(
            dict(
                save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
                ddpm_agent_kwargs=dict(
                    score_network_kwargs=dict(
                        time_dim=128,
                        num_blocks=3,
                        dropout_rate=0.1,
                        hidden_dim=256,
                        use_layer_norm=True,
                    ),
                    use_proprio=False,
                    beta_schedule="cosine",
                    diffusion_steps=5,
                    action_samples=64,
                    repeat_last_step=0,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    image_observations=True,
                    batch_size=256,
                    early_goal_concat=True,
                    shared_goal_encoder=True,
                    drq_padding=0,
                ),
                critic_agent_kwargs=get_continuous_cql_config(
                    updates=dict(
                        discount=0.99,
                        batch_size=256,
                        distributional_critic=True,
                        distributional_critic_kwargs=dict(
                            q_min=-400.0,
                            q_max=0.0,
                            num_bins=128,
                        ),
                        critic_network_type="layer_input_mlp",
                        critic_kwargs=dict(
                            kernel_init_type="orthogonal",
                            kernel_init_params=dict(
                                scale=1e-2,
                            ),
                            network_separate_action_input=True,
                        ),
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="uniform",
                        ),
                        critic_network_kwargs=dict(
                            hidden_dims=(512, 512, 512),
                            activate_final=True,
                            kernel_scale_final=1e-2,
                            use_feature_normalization=False,
                            use_layer_norm=True,
                        ),
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activate_final": True,
                            "use_layer_norm": False,
                            "use_group_norm": False,
                            "kernel_scale_final": 1e-2,
                        },
                        actor_optimizer_kwargs=dict(
                            learning_rate=1e-4,
                            warmup_steps=0,
                        ),
                        critic_optimizer_kwargs={
                            "learning_rate": 3e-4,
                            "warmup_steps": 0,
                            "weight_decay": 0.0,
                        },
                        temperature_optimizer_kwargs={
                            "learning_rate": 1e-4,
                        },
                        num_ddpm_actions=32,
                        num_actions_to_keep_for_q_diffusion=10,
                        q_diffusion_num_steps=10,
                        q_diffusion_num_steps_for_cql=-1,  # -1 means use q_diffusion_num_steps
                        q_diffusion_step_size=3e-4,
                        q_diffusion_optimize_critic_ensemble_min=False,
                        q_diffusion_use_adam=False,
                        q_diffusion_adam_kwargs=dict(
                            b1=0.9,
                            b2=0.999,
                        ),
                        q_diffusion_half_step_size_on_overshooting=False,
                        q_diffusion_overshooting_factor=0.5,
                        train_gaussian_policy=False,
                        use_gaussian_policy_for_critic_training=False,
                        always_use_argmax_for_q_diffusion=False,
                        use_calql=True,
                        use_calql_on_random_actions=False,
                        autotune_entropy=False,
                        cql_autotune_alpha=False,
                        use_target_critic_for_q_diffusion_steps=False,
                        cql_n_actions=4,
                        cql_importance_sample=False,
                        use_dataset_actions_for_cql_regularization=False,
                        critic_ensemble_size=10,
                        critic_subsample_size=2,
                        goal_conditioned=False,
                        exploration_noise_std=0.0,
                        use_td_loss=True,
                        drq_padding=0,
                    ),
                ),
                data_collection_q_diffusion_num_steps=-1.0,
                data_collection_q_diffusion_step_size=-1.0,
                bound_q_targets=False,
                num_eval_episodes=50,
                num_episodes_per_video=5,
                num_idql_action_samples=128,
                reward_scale=1.0,
                reward_bias=0.0,
                max_episode_steps=1000,
                batch_size=256,
                data_collection_particle_choosing_strategy="random_weighted",
                evaluation_particle_choosing_strategy="max_q_value",
                deterministic_eval=True,
                q_diffusion_num_steps_for_ddpm_training=-1,  # use q_diffusion_num_steps
                q_diffusion_step_size_for_ddpm_training=-1.0,  # use q_diffusion_step_size
                improve_ddpm_actions_with_global_search=True,
                distill_argmax=True,
                image_observations=True,
                goal_conditioned=False,
                early_terminate_on_success=False,
                dataset_kwargs=dict(
                    cache=False,
                    tfrecords_include_next_observations=False,
                ),
                image_replay_buffer_kwargs=dict(
                    cache=False,
                    tfrecords_include_next_observations=True,
                ),
                encoder="resnetv1-18-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=False,
                    act="swish",
                ),
                calvin_tfrecord_regexp="./calvin/dataset/task_D_D/training_tfrecords_cpu_images_rewards_float_masks/?*.tfrecord",
            )
        ),
        "cem_parl": ml_collections.ConfigDict(
            dict(
                save_dir=tf.io.gfile.join(SAVE_DIR_PREFIX, "results"),
                ddpm_agent_kwargs=dict(
                    batch_size=256,
                    ddpm_checkpoint_path="",
                    ddpm_agent_kwargs=dict(
                        batch_size=256,
                        score_network_kwargs=dict(
                            time_dim=128,
                            num_blocks=3,
                            dropout_rate=0.1,
                            hidden_dim=256,
                            use_layer_norm=True,
                        ),
                        use_proprio=False,
                        beta_schedule="cosine",
                        diffusion_steps=5,
                        action_samples=64,
                        repeat_last_step=0,
                        image_observations=True,
                    ),
                ),
                critic_agent_kwargs=get_continuous_cql_config(
                    updates=dict(
                        discount=0.99,
                        batch_size=256,
                        distributional_critic=False,
                        critic_network_type="layer_input_mlp",
                        critic_kwargs=dict(
                            kernel_init_type="orthogonal",
                            kernel_init_params=dict(
                                scale=1e-2,
                            ),
                            network_separate_action_input=True,
                        ),
                        critic_network_kwargs=dict(
                            hidden_dims=(512, 512, 512),
                            activate_final=True,
                            kernel_scale_final=1e-2,
                            use_feature_normalization=False,
                            use_layer_norm=True,
                        ),
                        critic_optimizer_kwargs={
                            "learning_rate": 3e-4,
                            "warmup_steps": 0,
                            "weight_decay": 0.0,
                        },
                        num_ddpm_actions=1,
                        num_actions_to_keep_for_q_diffusion=1,
                        q_diffusion_num_steps=0,
                        q_diffusion_num_steps_for_cql=-1,  # -1 means use q_diffusion_num_steps
                        q_diffusion_step_size=0,
                        q_diffusion_optimize_critic_ensemble_min=False,
                        q_diffusion_use_adam=False,
                        q_diffusion_adam_kwargs=dict(
                            b1=0.9,
                            b2=0.999,
                        ),
                        q_diffusion_half_step_size_on_overshooting=False,
                        q_diffusion_overshooting_factor=0.5,
                        train_gaussian_policy=False,
                        use_gaussian_policy_for_critic_training=False,
                        always_use_argmax_for_q_diffusion=False,
                        use_calql=True,
                        use_calql_on_random_actions=False,
                        autotune_entropy=False,
                        cql_autotune_alpha=False,
                        use_target_critic_for_q_diffusion_steps=False,
                        cql_n_actions=4,
                        cql_importance_sample=False,
                        use_dataset_actions_for_cql_regularization=False,
                        critic_ensemble_size=10,
                        critic_subsample_size=2,
                        goal_conditioned=False,
                        exploration_noise_std=0.0,
                        use_td_loss=True,
                        drq_padding=4,
                    ),
                ),
                data_collection_q_diffusion_num_steps=-1.0,
                data_collection_q_diffusion_step_size=-1.0,
                bound_q_targets=False,
                num_eval_episodes=50,
                num_episodes_per_video=5,
                num_idql_action_samples=128,
                reward_scale=1.0,
                reward_bias=0.0,
                max_episode_steps=1000,
                batch_size=256,
                data_collection_particle_choosing_strategy="random_weighted",
                evaluation_particle_choosing_strategy="max_q_value",
                deterministic_eval=True,
                q_diffusion_num_steps_for_ddpm_training=-1,  # use q_diffusion_num_steps
                q_diffusion_step_size_for_ddpm_training=-1.0,  # use q_diffusion_step_size
                improve_ddpm_actions_with_global_search=False,
                distill_argmax=True,
                image_observations=True,
                goal_conditioned=False,
                early_terminate_on_success=False,
                dataset_kwargs=dict(
                    cache=False,
                    tfrecords_include_next_observations=False,
                ),
                image_replay_buffer_kwargs=dict(
                    cache=False,
                    tfrecords_include_next_observations=True,
                ),
                encoder="resnetv1-18-bridge",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=False,
                    act="swish",
                ),
                calvin_tfrecord_regexp="./calvin/dataset/task_D_D/training_tfrecords_cpu_images_rewards_float_masks/?*.tfrecord",
            )
        ),
    }

    return possible_structures[config_string]
