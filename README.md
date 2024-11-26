# PA-RL Code for Reproducing Sim Experiments

## Environment
```
conda create -n parl python=3.11
conda activate parl
pip install -e .
pip install -r requirements.txt

```
For TPU
```
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```


## Replicating FrankaKitchen results:

First pre-train a Diffusion Policy with BC:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./experiments/train_agent.py --environment_name=kitchen-{dataset:mixed,complete,partial}-v0 --wandb_experiment_name=ddpm_bc_kitchen-{dataset}-v0 --config=./experiments/configs/offline_state_config.py:ddpm --n_epochs=3000 --config.seed={seed:0-4}
```

Then run PA-RL offline RL pre-training:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./experiments/train_agent.py --environment_name=kitchen-{dataset:mixed,complete,partial}-v0 --wandb_experiment_name=parl_kitchen-{dataset} --config=./experiments/configs/offline_state_config.py:diffusion_cql --config.seed={seed:0-4} --reward_bias=-4 --config.agent_kwargs.critic_network_kwargs.hidden_dims=512,512,512 --config.ddpm_agent_path=./results/Q-Diffusion/ddpm_bc_kitchen-{dataset}-v0/seed_{seed}/checkpoint_3000/ --config.agent_kwargs.cql_alpha=0.005 --n_epochs=500 --config.agent_kwargs.distributional_critic_kwargs.q_min=-400 --skip_if_last_checkpoint_exists=True
```

Finally run online fine-tuning:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./experiments/q_diffusion_finetuning.py --environment_name=kitchen-{dataset}-v0 --wandb_experiment_name=parl_ft_kitchen-{dataset} --config=./experiments/configs/finetune_state_config.py:q-diffusion_cql --ddpm_checkpoint_path=./results/Q-Diffusion/ddpm_bc_kitchen-{dataset}-v0/seed_{seed}/checkpoint_3000/ --critic_agent_name=diffusion_cql --critic_checkpoint_path=./results/Q-Diffusion/parl_kitchen-{dataset}/seed_{seed}/checkpoint_500/ --seed={seed} --mixing_ratio=0.25 --config.critic_agent_kwargs.critic_network_kwargs.hidden_dims=512,512,512 --config.critic_agent_kwargs.distributional_critic_kwargs.q_min=-400 --config.reward_bias=-4 --config.ddpm_agent_kwargs.learning_rate=1e-5
```


## Replicating AntMaze results:

Diffusion Policy BC training:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./experiments/train_agent.py --environment_name=antmaze-large-{dataset:diverse,play}-v2 --wandb_experiment_name=ddpm_bc_antmaze-large-{dataset}-v2 --config=./experiments/configs/offline_state_config.py:ddpm --n_epochs=3000 --config.seed={seed:0-4}
```

PA-RL offline RL pre-training:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./experiments/train_agent.py --environment_name=antmaze-large-{dataset}-v2 --wandb_experiment_name=parl_antmaze-large-{dataset} --config=./experiments/configs/offline_state_config.py:diffusion_cql --config.seed={seed:0-4} --reward_bias=-1 --config.agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 --config.ddpm_agent_path=./results/Q-Diffusion/ddpm_bc_antmaze-large-{dataset}-v2/seed_{seed}/checkpoint_3000/ --config.agent_kwargs.cql_alpha=0.005 --n_epochs=1000 --skip_if_last_checkpoint_exists=True
```

Online fine-tuning:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./experiments/q_diffusion_finetuning.py --environment_name=antmaze-large-{dataset}-v2 --wandb_experiment_name=parl_ft_antmaze-large-{dataset} --config=./experiments/configs/finetune_state_config.py:q-diffusion_cql --ddpm_checkpoint_path=./results/Q-Diffusion/ddpm_bc_antmaze-large-{dataset}-v2/seed_{seed}/checkpoint_3000/ --critic_agent_name=diffusion_cql --critic_checkpoint_path=./results/Q-Diffusion/parl_antmaze-large-{dataset}/seed_{seed}/checkpoint_1000/ --seed={seed} --mixing_ratio=0.5 --config.critic_agent_kwargs.critic_network_kwargs.hidden_dims=256,256,256,256 --config.critic_agent_kwargs.cql_alpha=0.005 --config.reward_bias=-1 --config.data_collection_particle_choosing_strategy=max_q_value --config.ddpm_agent_kwargs.learning_rate=5e-5
```


## Replicating CALVIN results:

Diffusion Policy BC training:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python ./experiments/train_agent.py --environment_name=calvin --wandb_experiment_name=ddpm_bc_calvin --config=./experiments/configs/offline_calvin_config.py:ddpm --n_epochs=3000 --config.seed={seed} --config.agent_kwargs.drq_padding=4 --config.calvin_tfrecord_regexp=gs://path_to_calvin_training_tfrecords/?*.tfrecord
```

PA-RL offline RL pre-training:
```
SAVE_DIR_PREFIX=gs://your_bucket/jaxrl_logs XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./experiments/train_agent.py --environment_name=calvin --wandb_experiment_name=parl_calvin --config=./experiments/configs/offline_calvin_config.py:non_gc_diffusion_cql --config.seed={seed} --reward_bias=-4 --config.ddpm_agent_path=./results/Q-Diffusion/ddpm_bc_calvin/seed_{seed}/checkpoint_3000/ --config.agent_kwargs.cql_alpha=0.01 --n_epochs=500 --config.agent_kwargs.distributional_critic=False --config.agent_kwargs.drq_padding=4 --config.calvin_tfrecord_regexp=gs://path_to_calvin_training_tfrecords/?*.tfrecord --skip_if_last_checkpoint_exists=True
```

Online fine-tuning:
```
SAVE_DIR_PREFIX=gs://your_bucket/jaxrl_logs XLA_PYTHON_CLIENT_PREALLOCATE=false env -u PYOPENGL_PLATFORM python ./experiments/q_diffusion_finetuning.py --environment_name=calvin --wandb_experiment_name=parl_ft_calvin --config=./experiments/configs/finetune_calvin_config.py:non_gc_q-diffusion_cql --ddpm_checkpoint_path=./results/Q-Diffusion/ddpm_bc_calvin/seed_{seed}/checkpoint_3000/ --critic_agent_name=diffusion_cql --critic_checkpoint_path=gs://your_bucket/jaxrl_logs/results/Q-Diffusion/parl_calvin/seed_{seed}/checkpoint_500/ --seed={seed} --mixing_ratio=0.5 --config.critic_agent_kwargs.critic_network_kwargs.hidden_dims=512,512,512 --config.critic_agent_kwargs.cql_alpha=0.01 --config.reward_bias=-4 --config.critic_agent_kwargs.cql_n_actions=4 --config.critic_agent_kwargs.distributional_critic=False --config.ddpm_agent_kwargs.drq_padding=4 --config.critic_agent_kwargs.drq_padding=4 --config.calvin_tfrecord_regexp=gs://path_to_calvin_tfrecords/?*.tfrecord --config.ddpm_agent_kwargs.learning_rate=5e-5
```
