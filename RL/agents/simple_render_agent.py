import logging

import RL


class SimpleRenderAgent(RL.Agent):
    def post_act(self):
        try:
            self.env.render()
        except Exception:
            logging.getLogger(__name__).exception(f'{self.name}: Unable to render. Disabling agent!')
            self.disable()
