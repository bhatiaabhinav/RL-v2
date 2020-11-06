import logging

from matplotlib.figure import Figure
from gym.envs.classic_control.rendering import SimpleImageViewer
import numpy as np

import RL


class SimpleRenderAgent(RL.Agent):
    def __init__(self, plotfig_getter=None, image_getter=None, render_fn=None) -> None:
        self.render_fn = render_fn
        self.image_getter = image_getter
        self.plotfig_getter = plotfig_getter
        self.viewer = None

    def post_act(self):
        try:
            if self.render_fn is not None:
                self.render_fn()
            elif self.image_getter is not None:
                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                img = self.image_getter()
                self.viewer.imshow(img)
            elif self.plotfig_getter is not None:
                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                fig = self.plotfig_getter()  # type: Figure
                data = np.fromstring(fig.canvas.tostring_rgb(),
                                     dtype=np.uint8)
                img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                self.viewer.imshow(img)
            else:
                self.env.render()
        except Exception:
            logging.getLogger(__name__).exception(f'{self.name}: Unable to render. Disabling agent!')
            self.disable()
