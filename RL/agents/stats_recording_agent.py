import logging
import time
from collections import deque

import wandb

import RL

logger = logging.getLogger(__name__)


class StatsRecordingAgent(RL.Agent):
    def __init__(self, name, algo, reward_scaling, cost_scaling, record_unscaled, gamma, cost_gamma, record_undiscounted, frameskip, RPE_av_over=100, RPS_av_over=100000):
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
        self.last_recorded_for_step_id = -1
        self.RPE_av_over = RPE_av_over
        self.RPS_av_over = RPS_av_over

    def start(self):
        self._start_time = time.time()
        self._episode_returns = deque(maxlen=self.RPE_av_over)
        self._episode_cost_returns = deque(maxlen=self.RPE_av_over)
        self._rewards = deque(maxlen=self.RPS_av_over)
        self._costs = deque(maxlen=self.RPS_av_over)
        self._rpe = 0
        self._cpe = 0
        self._rpe_av = 0
        self._cpe_av = 0
        self._rr = 0
        self._cr = 0
        self._rr_av = 0
        self._cr_av = 0

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
        self._rr_av = self.new_moving_mean(self._rr_av, r, self._rewards)
        self._cr_av = self.new_moving_mean(self._cr_av, c, self._costs)

        if (self.manager.num_steps - 1) % 1000 == 0:
            if self.manager.num_steps - 1 > self.last_recorded_for_step_id:
                self.record_summary_stats()
                self.last_recorded_for_step_id = self.manager.num_steps - 1

    def post_episode(self):
        self._rpe = self.new_mean(
            self._rpe, self._episode_return, self.manager.num_episodes)
        self._cpe = self.new_mean(
            self._cpe, self._episode_cost_return, self.manager.num_episodes)
        self._rpe_av = self.new_moving_mean(
            self._rpe_av, self._episode_return, self._episode_returns)
        self._cpe_av = self.new_moving_mean(
            self._cpe_av, self._episode_cost_return, self._episode_cost_returns)

        self.record_episode_stats()
        self.record_summary_stats()
        self.last_recorded_for_step_id = self.manager.num_steps - 1

    def pre_close(self):
        self.record_summary_stats()

    def record_episode_stats(self):
        wandb.log({
            'Episode/ID': self.manager.num_episodes - 1,
            'Episode/Type': self.manager.episode_type,
            'Episode/Steps': self.manager.num_episode_steps,
            'Episode/EndTimestamp': time.time() - self._start_time,
            'Episode/SPS': self.manager.num_episode_steps / (1e-6 + time.time() - self._episode_start_time),
            'Episode/Reward': self._episode_return,
            'Episode/Cost': self._episode_cost_return,
            'Episode/Info': self.manager.info
        }, step=self.manager.num_steps - 1)

    def record_summary_stats(self):
        wandb.log({
            'Total/Episodes': self.manager.num_episodes,
            'Total/Steps': self.manager.num_steps,
            'Total/Frames': self.frameskip * self.manager.num_steps,
            'Total/Reward': self._rr * self.manager.num_steps,
            'Total/Cost': self._cr * self.manager.num_steps,
            'Average/RPE': self._rpe,
            'Average/CPE': self._cpe,
            f'Average/RPE (Last {self.RPE_av_over})': self._rpe_av,
            f'Average/CPE (Last {self.RPE_av_over})': self._cpe_av,
            'Average/RPS': self._rr,
            'Average/CPS': self._cr,
            f'Average/RPS (Last {self.RPS_av_over})': self._rr_av,
            f'Average/CPS (Last {self.RPS_av_over})': self._cr_av
        }, step=self.manager.num_steps - 1)
