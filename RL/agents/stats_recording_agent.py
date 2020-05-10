import logging
import time

import numpy as np

import RL

logger = logging.getLogger(__name__)


class StatsRecordingAgent(RL.Agent):
    def __init__(self, name, algo, reward_scaling, cost_scaling, record_unscaled, gamma, cost_gamma, record_undiscounted, frameskip, should_exploit_fn):
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
        self.should_exploit_fn = should_exploit_fn

        self._stats = RL.stats.stats

        self._stats['episode_ids'] = []
        self._stats['episode_exploit_types'] = []
        self._stats['episode_lengths'] = []
        self._stats['total_steps'] = []
        self._stats['total_frames'] = []
        self._stats['episode_timestamps'] = []
        self._stats['episode_returns'] = []
        self._stats['episode_cost_returns'] = []
        self._stats['episode_sps'] = []

    def start(self):
        self._start_time = time.time()

    def pre_episode(self):
        self._episode_return = 0
        self._episode_cost_return = 0
        self._episode_start_time = time.time()
        self._cur_gamma = 1
        self._cur_cost_gamma = 1

    def post_act(self):
        r = self.manager.reward / \
            self.reward_scaling if self.record_unscaled else self.manager.reward
        c = self.manager.cost / self.cost_scaling if self.record_unscaled else self.manager.cost
        self._episode_return += self._cur_gamma * r
        self._episode_cost_return += self._cur_cost_gamma * c

        if not self.record_undiscounted:
            self._cur_gamma *= self.gamma
            self._cur_cost_gamma *= self.cost_gamma

    def post_episode(self):
        self._stats['episode_ids'].append(self.manager.num_episodes - 1)
        self._stats['episode_exploit_types'].append(self.should_exploit_fn())
        self._stats['episode_lengths'].append(self.manager.num_episode_steps)
        self._stats['total_steps'].append(self.manager.num_steps)
        self._stats['total_frames'].append(
            self.manager.num_steps * self.frameskip)
        self._stats['episode_timestamps'].append(
            time.time() - self._start_time)
        self._stats['episode_returns'].append(self._episode_return)
        self._stats['episode_cost_returns'].append(self._episode_cost_return)
        self._stats['episode_sps'].append(
            self.manager.num_episode_steps / (time.time() - self._episode_start_time))
        RL.stats.record_kvstat('Av RPE', np.mean(
            self._stats['episode_returns']))
        RL.stats.record_kvstat('Av RPE (Last 100)', np.mean(
            self._stats['episode_returns'][-100:]))
        RL.stats.record_kvstat('Av CPE', np.mean(
            self._stats['episode_cost_returns']))
        RL.stats.record_kvstat('Av CPE (Last 100)', np.mean(
            self._stats['episode_cost_returns'][-100:]))
        # latest_stats = RL.stats.get_latest_stats()
        # logger.info(f'All latest stats: {latest_stats}')
        # logger.info(f'All kv stats: {self.kvstats}')

    def pre_close(self):
        RL.stats.record_kvstat('Total Steps', self.manager.num_steps)
        RL.stats.record_kvstat('Total Episodes', self.manager.num_episodes)
