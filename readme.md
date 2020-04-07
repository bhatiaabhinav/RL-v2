# RL Algorithms

## Installation

System requirements: python3

For linux/osx:

```bash
git clone https://github.com/bhatiaabhinav/RL-v2.git
cd RL-v2
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
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
python -m RL CartPole-v0 DQN 20000 --algo_suffix=test --seed=0 --nsteps=3 --ep_anneal_steps=10000
```

To get list of specifiable hyperparams:
```bash
python -m RL -h
```
