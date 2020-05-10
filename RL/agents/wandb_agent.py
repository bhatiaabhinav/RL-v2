import wandb

import RL


class WandbAgent(RL.Agent):
    def __init__(self, name, algo, episode_freq=1, step_freq=None):
        super().__init__(name, algo, supports_multiple_envs=False)
        self.episode_freq = episode_freq
        self.step_freq = step_freq

    def write_to_wandb(self):
        s = RL.stats.get_latest_stats()
        wandb.log(s, sync=False)

    def post_episode(self):
        if self.episode_freq is not None and (self.manager.num_episodes - 1) % self.episode_freq == 0:
            self.write_to_wandb()

    def post_act(self):
        if self.step_freq is not None and (self.manager.num_steps - 1) % self.step_freq == 0:
            self.write_to_wandb()

    def pre_close(self):
        self.write_to_wandb()
