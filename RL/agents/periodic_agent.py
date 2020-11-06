import RL


class PeriodicAgent(RL.Agent):
    def __init__(self, name, algo, fn, step_freq, episode_freq=None, min_steps=0, min_episodes=0, call_at_start=True, call_at_pre_close=True, call_at_post_close=False):
        '''The fn should take as args step_id, episode_id'''
        super().__init__(name, algo, False)
        self.fn = fn
        self.step_freq = step_freq
        self.episode_freq = episode_freq
        self.min_steps = min_steps
        self.min_episodes = min_episodes
        self.call_at_start = call_at_start
        self.call_at_pre_close = call_at_pre_close
        self.call_at_post_close = call_at_post_close

    def start(self):
        if self.call_at_start:
            self.fn(0, 0)

    def post_act(self):
        if self.step_freq is not None and self.manager.num_steps >= self.min_steps and (self.manager.num_steps - 1) % self.step_freq == 0:
            self.fn(self.manager.num_steps - 1, self.manager.num_episodes)

    def post_episode(self):
        if self.episode_freq is not None and self.manager.num_episodes > self.no_copy_for_episodes and (self.manager.num_episodes - 1) % self.episode_freq == 0:
            self.fn(self.manager.num_steps - 1, self.manager.num_episodes - 1)

    def pre_close(self):
        if self.call_at_pre_close:
            self.fn(self.manager.num_steps - 1, self.manager.num_episodes - 1)

    def post_close(self):
        if self.call_at_post_close:
            self.fn(self.manager.num_steps - 1, self.manager.num_episodes - 1)
