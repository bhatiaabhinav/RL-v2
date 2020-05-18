import logging

from torch import nn

import RL


class ModelCopyAgent(RL.Agent):
    def __init__(self, name, algo, model_from: nn.Module, model_to: nn.Module, step_freq, polyak, no_copy_for_steps, episode_freq=None, no_copy_for_episodes=0):
        super().__init__(name, algo, False)
        self.model_from = model_from
        self.model_to = model_to
        self.step_freq = step_freq
        self.polyak = polyak
        self.no_copy_for_steps = no_copy_for_steps
        self.episode_freq = episode_freq
        self.no_copy_for_episodes = no_copy_for_episodes

    def copy(self, polyak):
        logging.getLogger(__name__).debug(
            f'Copying model with polyak {polyak}')
        params_from = self.model_from.parameters()
        params_to = self.model_to.parameters()
        for pf, pt in zip(params_from, params_to):
            if polyak == 0:
                pt.data.copy_(pf.data)
            else:
                pt.data.copy_((1 - polyak) * pf.data + polyak * pt.data)

    def start(self):
        self.copy(0)

    def post_act(self):
        if self.step_freq is not None and self.manager.num_steps > self.no_copy_for_steps and (self.manager.num_steps - 1) % self.step_freq == 0:
            self.copy(self.polyak)

    def post_episode(self):
        if self.episode_freq is not None and self.manager.num_episodes > self.no_copy_for_episodes and (self.manager.num_episodes - 1) % self.episode_freq == 0:
            self.copy(self.polyak)
