import logging
import time
from collections import deque

import wandb

import RL

logger = logging.getLogger(__name__)


class StatsRecordingAgent(RL.Agent):
    def __init__(self, name, algo, reward_scaling, cost_scaling, record_unscaled, gamma, cost_gamma, record_undiscounted, frameskip):
        global logger
        logger = logging.getLogger(__name__)
        super().__init__(name, algo, supports_multiple_envs=False)
        self.reward_scaling = reward_scaling
        self.cost_scaling = cost_scaling
        self.record_unscaled = record_unscaled
        self.gamma = gamma
        self.cost_gamma = cost_gamma
        self.record_undiscounted = record_undiscounted
        self.frameskip = frameskip

    def start(self):
        self._start_time = time.time()
        self._episode_returns = deque(maxlen=100)
        self._episode_cost_returns = deque(maxlen=100)
        self._rewards = deque(maxlen=100000)
        self._costs = deque(maxlen=100000)
        self._rpe = 0
        self._cpe = 0
        self._rpe100 = 0
        self._cpe100 = 0
        self._rr = 0
        self._cr = 0
        self._rr10000 = 0
        self._cr10000 = 0

    def pre_episode(self):
        self._episode_return = 0
        self._episode_cost_return = 0
        self._episode_start_time = time.time()
        self._cur_gamma = 1
        self._cur_cost_gamma = 1

    def new_moving_mean(self, old_mean, x, q: deque):
        old_sum = old_mean * len(q)
        if len(q) == q.maxlen:
            new_sum = old_sum + x - q[0]
        else:
            new_sum = old_sum + x
        q.append(x)
        return new_sum / len(q)

    def new_mean(self, old_mean, x, n):
        return (old_mean * (n - 1) + x) / n

    def post_act(self):
        r = self.manager.reward / \
            self.reward_scaling if self.record_unscaled else self.manager.reward
        c = self.manager.cost / self.cost_scaling if self.record_unscaled else self.manager.cost
        self._episode_return += self._cur_gamma * r
        self._episode_cost_return += self._cur_cost_gamma * c

        if not self.record_undiscounted:
            self._cur_gamma *= self.gamma
            self._cur_cost_gamma *= self.cost_gamma

        self._rr = self.new_mean(self._rr, r, self.manager.num_steps)
        self._cr = self.new_mean(self._cr, c, self.manager.num_steps)
        self._rr10000 = self.new_moving_mean(self._rr10000, r, self._rewards)
        self._cr10000 = self.new_moving_mean(self._cr10000, c, self._costs)

        if self.manager.step_id % 1000 == 0:
            self.record_summary_stats()

    def post_episode(self):
        self._rpe = self.new_mean(
            self._rpe, self._episode_return, self.manager.num_episodes)
        self._cpe = self.new_mean(
            self._cpe, self._episode_cost_return, self.manager.num_episodes)
        self._rpe100 = self.new_moving_mean(
            self._rpe100, self._episode_return, self._episode_returns)
        self._cpe100 = self.new_moving_mean(
            self._cpe100, self._episode_cost_return, self._episode_cost_returns)

        self.record_episode_stats()
        self.record_summary_stats()

    def pre_close(self):
        self.record_summary_stats()

    def record_episode_stats(self):
        wandb.log({
            'Episode/ID': self.manager.episode_id,
            'Episode/Type': self.manager.episode_type,
            'Episode/Steps': self.manager.num_episode_steps,
            'Episode/EndTimestamp': time.time() - self._start_time,
            'Episode/SPS': self.manager.num_episode_steps / (time.time() - self._episode_start_time),
            'Episode/Reward': self._episode_return,
            'Episode/Cost': self._episode_cost_return,
        }, step=self.manager.step_id)

    def record_summary_stats(self):
        wandb.log({
            'Total/Episodes': self.manager.num_episodes,
            'Total/Steps': self.manager.num_steps,
            'Total/Frames': self.frameskip * self.manager.num_steps,
            'Total/Reward': self._rr * self.manager.num_steps,
            'Total/Cost': self._cr * self.manager.num_steps,
            'Average/RPE': self._rpe,
            'Average/CPE': self._cpe,
            'Average/RPE (Last 100)': self._rpe100,
            'Average/CPE (Last 100)': self._cpe100,
            'Average/RPS': self._rr,
            'Average/CPS': self._cr,
            'Average/RPS (Last 100k)': self._rr10000,
            'Average/CPS (Last 100k)': self._cr10000
        }, step=self.manager.step_id)
