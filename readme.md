# RL Algorithms

## Installation

System requirements: python3.6

For linux/osx:

```bash
git clone https://github.com/bhatiaabhinav/RL-v2.git
cd RL-v2
python3 -m venv env
source env/bin/activate
pip install wheel
pip install -r requirements_ubuntu1804.txt
pip install -e .
mkdir logs
```
or set env variable RL_LOGDIR to specify logs directory.

---

## Usage

Activate env:
```bash
cd RL-v2
source env/bin/activate
```

Then:
```bash
python -m RL env_id algo_id num_steps_to_run --algo_suffix=my_run_name --tags tag1 tag2
```

- To run an experiment for fixed number of episodes instead of a fixed number of steps, set parameter `--num_episodes_to_run` accordingly and set `num_steps_to_run` to a very high value (say 1000000000)

### Wandb
A [Weights & Biases](https://wandb.ai) account is needed to run the module. It is tool to track, log, graph and visualize machine learning projects.
On running the module for the first time, it will ask for the authorization key of the account. Once, that is set, all the future runs will be logged to that account.

A run will be recorded by the name `{algo_id}_{algo_suffix}` in the `{env_name}` project in the wandb account. On starting a run, the link to view it in the wandb dashboard will be printed in the beginning.

### Logs

The logs for an experiment are stored in directory: `{logs folder}/{env name}/{algo_id}_{algo_suffix}`.

- Attempting to re-run an experiment with the same <env_name, algo_id, algo_suffix> combo will result in the module refusing to run to prevent overwriting previous logs. Speficy `--overwrite` flag to force the re-run and overwrite the logs.
- A different logs folder can be specified (instead of default 'logs' folder in the working directory, or $RL_LOGDIR if that environment variable is set) using parameter `--rl_logdir`.
- To run in debug mode (i.e. record all logs created using logger.debug('') in the log file), set `--debug` flag. By default, INFO and above level logs are recorded. To disable INFO level logs, set `--no_logs` flag.

### Rendering
By default env's rendering is turned on. To turn off, specify `--no_render` flag.

### OpenAI Gym Monitor
If OpenAI Gym monitor is used to record videos of episodes, the videos will be saved in run's logs directory, and will also be available for visualization in the wandb dashboard.
Specify `--no_monitor` flag to disable monitor. Or if monitor is used, specify episode intervals for recording videos using `--monitor_video_freq` (default 100) parameter.

### GPU Use
By default the module will use Nvidia cuda if a GPU is available. To disable GPU use, set `--no_gpu` flag.

---

## Features, Flags and Hyperparams


### For Deep Q-Network (DQN):
- DQN Vanilla
- Double DQN. Speficy `--double_dqn` flag.
- Dueling DQN. Specify `--dueling_dqn` flag.
- N-Step DQN. Specify `--nsteps` parameter (default=1).
- To turn on soft-copying of Q network parameters to target network (like in DDPG), specify e.g. `--target_q_polyak=0.999` (default=0) and also set `--target_q_freq=1` (default 10000) so as to copy the parameters every training step.
- Soft (Boltzman) policy. Set temperature using `--dqn_ptemp` (default=0).
- `--ep` (default=0.1) Value of epsilon in epsilon-greedy action selection.
- `-ep_anneal_steps` (default=1000000). Numner of steps over which the epsilon should be annealed from 1 to `ep`. The annealing begins after `min_explore_steps` phase. `min_explore_steps` is explained below.

### For DDPG
- DDPG Vanilla (but without OH noise exploration)
- Adaptive Param Noise Exploration. Set target deviation of noisy policy using `--ddpg_noise` (default 0.2).
- N-Step DDPG. Specify `--nsteps` parameter (default=1).

### For SAC
- SAC Vanilla
- Adaptive alpha to maintain constant entropy (turned on default). Specify initial alpha `--sac_alpha` (default 0.2). To turn off adaptive alpha, specify `--fix_alpha` flag.
- N-Step SAC. Specify `--nsteps` parameter (default=1).

Note: DDPG and SAC algorithms soft-copy Q net params to target net

### General Training related

- Network: Upto 3 convolutional layer can be specified, followed by any number of hidden layers. All conv layers are automatically ignored when the input is not an image.
    - `--conv1` (default= 32 8 4 0). Parameters of the first convolution layer. Specify like this: `--conv1 channels kernel_size stride padding`. Specify `--conv1 0` to skip this layer.
    - `--conv2` (default= 64 4 2 0). Specification format same as conv1. Specify `--conv2 0` to skip this layer.
    - `--conv3` (default= 64 3 1 0). Specification format same as conv1. Specify `--conv3 0` to skip this layer.
    - `--hiddens` (default= 512). Hidden layers specify like: `--hiddens h1 h2 h3`. E.g. `--hiddens 512 256 64 32` to create 4 hidden layers with respective number of nodes. To specify no hidden layers, pass `--hiddens ` i.e. the argument name followed by only a whitespace.
- `--gamma` (default=0.99). Discount factor for training.
- `--exp_buff_len` (default=1M). Experience buffer length.
- `--no_ignore_done_on_timlimit`. By default, the experience buffer records `done` as false when an episode is terminated artifically due to timelimit wrapper (because the episode did not really end, and recording done=True in such cases would cause difficulty in learning the value function, since the markov property of the preceding state would break). Specify this flag to disable this functionality.
- `--min_explore_steps` (default=1000000). In the beginning of training, execute a random policy for these many steps.
- `--exploit_freq` (default=None). Play the learnt policy (so far) for an episode without exploration every exploit_freq episodes.
- `--train_freq` (default=4). Train step every these many episode steps.
- `--sgd_steps` (default=1). Number of SGD updates per training step.
- `--lr` (default=1e-4). Learning rate for Adam optimizer.
- `--mb_size` (default=32). Minibatch size.
- `--td_clip` (default=None). Clip temporal difference errors to this magnitude.
- `--grad_clip` (default=None). Clip gradients (by norm) to this magnitude.


### General
- `--seed`. None by default, leading to inderministic behavior. If set to some value, the following get seeded with the specified value: python's random module, torch and numpy. Also, every episode is seeded deterministically (= speficied seed + episode ID) so that episodes are comparable across runs.
- `--reward_scaling` (default=1). Scales rewards for training. The logs & graphs also record scaled returns, unless `--record_unscaled` flag is specified.
- `record_discounted` flag. When set, causes logs & graphs to record discounted returns per episode, instead of sum of rewards per episode.
- `--model_save_freq` (default=1000000). The model is saved every these many steps to checkpoints directory

### Wrappers
- For non-atari environments (both image-based or non image-based):
    - Framestack e.g. `--framestack=4` (default=1)
    - Frameskip e.g. `--frameskip=4` (default=1)
    - Artifical timelimit e.g. `--artificial_timelimit=500` (default None). It uses built-in gym's Timelimit wrapper to force termination of episodes at these many steps.
- Atari Specific:
    - Framestack e.g. `--atari_framestack=4` (default=4)
    - Frameskip e.g. `--atari_frameskip=4` (default=4)
    - Max num of noops in beginnning of episode e.g. `--atari_noop_max=30` (default=30).
    - reward clipping flag. `--atari_clip_rewards`.

Framestack is always applied on top of frameskip. e.g. if frameskip is 4 and framestack is 3, then frames 0,4,8 are stacked together to form the first _step's_ observation. Then frame 4,8,12 are stacked together to form the second _step's_ observation. and so on.


#### **Notes for Atari**:
For Atari, use only environments:
- ending in `NoFrameskip-v4`. e.g. `BreakoutNoFrameskip-v4` or `PongNoFrameskip-v4` etc. to play from pixels.
- or ending in `-ramNoFrameskip-v4` e.g. `Breakout-ramNoFrameskip-v4' to play from RAM state.

Any other environment will be treated as non-atari environments.

Another note: `--atari_framestack` is ignored for RAM based atari environments. To force frame stacking, use the general `--framestack` argument.




---

## Examples


### DQN Examples:

```bash
python -m RL CartPole-v0 DQN 20000 --algo_suffix=quicktest --seed=0 --hiddens 64 32 --train_freq=1 --target_q_freq=2000 --nsteps=3 --min_explore_steps=10000 --ep_anneal_steps=10000 --ep=0.01 --no_render
```
or
```bash
python -m RL BreakoutNoFrameskip-v4 DQN 10000000 --algo_suffix=mnih --seed=0 --conv1 32 8 4 0 --conv2 64 4 2 0 --conv3 64 3 1 0 --hiddens 512 --no_render
```
or
```bash
python -m RL BreakoutNoFrameskip-v4 DQN 10000000 --algo_suffix=3step_mnih_small --seed=0 --conv1 16 8 4 0 --conv2 32 4 2 0 --conv3 0 --hiddens 256 --target_q_freq=8000 --nsteps=3 --min_explore_steps=100000 --ep_anneal_steps=100000 --exp_buff_len=100000 --no_render
```
or
```bash
python -m RL BreakoutNoFrameskip-v4 DQN 10000000 --algo_suffix=mnih_big --seed=0 --conv1 64 6 2 0 --conv2 64 6 2 2 --conv3 64 6 2 2 --hiddens 1024 --no_render
```

### SAC Examples
```bash
python -m RL Pendulum-v0 SAC 1000000 --algo_suffix=quicktest_gc1 --seed=0 --hiddens 64 32 --train_freq=1 --min_explore_steps=10000 --grad_clip=1 --no_render
```

```bash
python -m RL BipedalWalker-v3 SAC 1000000 --algo_suffix=T200_gc1 --seed=0 --hiddens 64 32 --train_freq=1 --min_explore_steps=10000 --grad_clip=1 --artificial_timelimit=200 --no_render
```

#### On a discrete action space domain:
```bash
python -m RL LunarLander-v2 SAC 200000 --algo_suffix=T500_3step --artificial_timelimit=500 --seed=0 --hiddens 64 32 --train_freq=1 --nsteps=3 --min_explore_steps=10000 --no_render --monitor_video_freq=20
```

### Hyperparams

To get list of specifiable hyperparams:
```bash
python -m RL -h
```

