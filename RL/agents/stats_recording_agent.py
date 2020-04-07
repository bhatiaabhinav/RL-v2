import logging
import time

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

        self.stats = {
            'episode_ids': [],
            'episode_exploit_types': [],
            'episode_lengths': [],
            'total_steps': [],
            'total_frames': [],
            'episode_timestamps': [],
            'episode_returns': [],
            'episode_cost_returns': [],
            'episode_sps': []
        }

        self.kvstats = {}

    def get_latest_stats(self):
        return dict(((k, v[-1]) for k, v in self.stats.items()))

    def record_stat(self, key, value):
        '''creates key if neccessary'''
        if key not in self.stats:
            self.stats[key] = []
        self.stats[key].append(value)

    def record_kvstat(self, key, value):
        '''a convinience method for kvstats[key]=value'''
        self.kvstats[key] = value

    def get_one(self, key, default=0):
        if key in self.kvstats:
            return self.kvstats[key]
        elif key in self.stats:
            return self.stats[key][-1]
        else:
            return default

    def start(self):
        self._start_time = time.time()

    def pre_episode(self):
        self._episode_return = 0
        self._episode_cost_return = 0
        self._episode_start_time = time.time()
        self._cur_gamma = 1
        self._cur_cost_gamma = 1

    def post_act(self):
        r = self.manager.reward / self.reward_scaling if self.record_unscaled else self.manager.reward
        c = self.manager.cost / self.cost_scaling if self.record_unscaled else self.manager.cost
        self._episode_return += self._cur_gamma * r
        self._episode_cost_return += self._cur_cost_gamma * c

        if not self.record_undiscounted:
            self._cur_gamma *= self.gamma
            self._cur_cost_gamma *= self.cost_gamma

    def post_episode(self):
        self.stats['episode_ids'].append(self.manager.num_episodes - 1)
        self.stats['episode_exploit_types'].append(self.should_exploit_fn())
        self.stats['episode_lengths'].append(self.manager.num_episode_steps)
        self.stats['total_steps'].append(self.manager.num_steps)
        self.stats['total_frames'].append(self.manager.num_steps * self.frameskip)
        self.stats['episode_timestamps'].append(time.time() - self._start_time)
        self.stats['episode_returns'].append(self._episode_return)
        self.stats['episode_cost_returns'].append(self._episode_cost_return)
        self.stats['episode_sps'].append(self.manager.num_episode_steps / (time.time() - self._episode_start_time))
        latest_stats = self.get_latest_stats()
        logger.info(f'All latest stats: {latest_stats}')
        logger.info(f'All kv stats: {self.kvstats}')
