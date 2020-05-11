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
pip install -r requirements.txt
pip install -e .
mkdir logs
```
or set env variable RL_LOGDIR to specify logs directory.

## Usage

Activate env:
```bash
cd RL-v2
source env/bin/activate
```

Then:
```bash
python -m RL env_id algo_id num_steps_to_run
```


DQN Example:
```bash
python -m RL CartPole-v0 DQN 20000 --algo_suffix=fast --seed=0 --hiddens 64 32 --mb_size=32 --train_freq=1 --target_q_freq=2000 --nsteps=3 --ep_anneal_steps=10000 --ep=0.01 --no_render
```
or
```bash
python -m RL BreakoutNoFrameskip-v4 DQN 1000000 --algo_suffix=mnih --seed=0 --conv1 32 8 4 0 --conv2 64 4 2 0 --conv3 64 3 1 0 --hiddens 512 --mb_size=32 --train_freq=4 --target_q_freq=10000 --min_explore_steps=50000 --ep_anneal_steps=1000000 --ep=0.1 --exp_buff_len=1000000 --no_render
```
or
```bash
python -m RL BreakoutNoFrameskip-v4 DQN 1000000 --algo_suffix=mnih_small --seed=0 --conv1 16 8 4 0 --conv2 32 4 2 0 --conv3 0 --hiddens 256 --mb_size=32 --train_freq=4 --target_q_freq=8000 --min_explore_steps=10000 --ep_anneal_steps=100000 --ep=0.1 --exp_buff_len=100000 --no_render
```

To get list of specifiable hyperparams:
```bash
python -m RL -h
```
