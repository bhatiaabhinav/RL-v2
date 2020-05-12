import RL


class LinearAnnealingAgent(RL.Agent):
    def __init__(self, name, algo, obj, variable_name, start_delay, start_val, final_val, duration):
        super().__init__(name, algo)
        self.obj = obj
        self.start_delay = start_delay
        self.start_val = start_val
        self.final_val = final_val
        self.duration = duration
        self.variable_name = variable_name
        self.val = start_val

    def start(self):
        setattr(self.obj, self.variable_name, self.start_val)
        self.val = self.start_val

    def pre_act(self):
        if self.manager.num_steps <= self.start_delay:
            val = self.start_val
        else:
            steps = self.manager.num_steps - self.start_delay
            val = self.final_val + \
                (self.start_val - self.final_val) * \
                (1 - min(steps / self.duration, 1))
        self.val = val
        setattr(self.obj, self.variable_name, val)
