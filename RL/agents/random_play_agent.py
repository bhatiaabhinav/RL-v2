import RL


class RandomPlayAgent(RL.Agent):
    def __init__(self, name, algo, play_for_steps=None):
        super().__init__(name, algo, supports_multiple_envs=False)
        self.play_for_steps = play_for_steps

    def act(self):
        if self.play_for_steps is not None and self.manager.num_steps >= self.play_for_steps:
            return None
        else:
            return self.env.action_space.sample(), {}
