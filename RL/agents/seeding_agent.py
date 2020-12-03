import logging
import random

import numpy as np
import torch

import RL


class SeedingAgent(RL.Agent):
    def __init__(self, name, algo, seed):
        super().__init__(name, algo, supports_multiple_envs=False)
        self.seed = seed
        if self.seed is not None:
            logging.getLogger(__name__).info(
                f'{self.name}:Seeding with seed {self.seed} during init')
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.env.seed(seed)
            self.env.action_space.seed(seed)

    def start(self):
        if self.seed is not None:
            logging.getLogger(__name__).info(
                f'{self.name}:Seeding with seed {self.seed} at start')
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.env.seed(self.seed)
            self.env.action_space.seed(self.seed)

    def pre_episode(self):
        # self.env.seed(self.seed + self.manager.num_episodes)
        pass
