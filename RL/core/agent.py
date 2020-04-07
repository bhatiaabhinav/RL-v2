import logging

import numpy as np

zero_id = np.array([0])
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, name, algo, supports_multiple_envs=False):
        global logger
        logger = logging.getLogger(__name__)
        self.name = name
        self.full_name = f'{algo.algo_id}_{algo.manager.algo_suffix}_{name}'
        logger.info(f"Initializing agent {self.name}")
        from .algorithm import Algorithm
        from .manager import Manager
        self.algo = algo  # type: Algorithm
        self.manager = self.algo.manager  # type: Manager
        self.env = self.manager.env
        self._enabled = True
        self.supports_multiple_envs = supports_multiple_envs

    @property
    def enabled(self):
        return self._enabled

    def start(self):
        pass

    def pre_episode(self):
        if self.supports_multiple_envs:
            self.pre_episode_multienv(zero_id)

    def pre_episode_multienv(self, env_id_nos):
        if not self.supports_multiple_envs:
            try:
                raise NotImplementedError(f"Agent {self.name} does not support multiple parallel envs yet.")
            except NotImplementedError as e:
                logger.exception(f"{self.name}: Exception happened")
                raise e

    def pre_act(self):
        pass

    def act(self):
        if self.supports_multiple_envs:
            acts = self.act_multienv()
            return None if acts is None else acts[0]
        else:
            return None

    def act_multienv(self):
        if not self.supports_multiple_envs:
            try:
                raise NotImplementedError(f"Agent {self.name} does not support multiple parallel envs yet.")
            except NotImplementedError as e:
                logger.exception(f"{self.name}: Exception happened")
                raise e

    def post_act(self):
        pass

    def post_episode(self):
        if self.supports_multiple_envs:
            self.post_episode_multienv(zero_id)

    def post_episode_multienv(self, env_id_nos):
        if not self.supports_multiple_envs:
            try:
                raise NotImplementedError(f"Agent {self.name} does not support multiple parallel envs yet.")
            except NotImplementedError as e:
                logger.exception(f"{self.name}: Exception happened")
                raise e

    def pre_close(self):
        pass

    def post_close(self):
        pass

    def on_enable(self):
        pass

    def on_disable(self):
        pass

    def enable(self):
        if not self._enabled:
            self._enabled = True
            logger.info(f'Agent {self.name} enabled. Calling on_enable')
            self.on_enable()

    def disable(self):
        if self._enabled:
            self._enabled = False
            logger.info(f'Agent {self.name} disabled. Calling on_disable')
            self.on_disable()
