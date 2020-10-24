import logging

import RL

logger = logging.getLogger(__name__)


class EpisodeTypeControlAgent(RL.Agent):
    """Sets the `manager.episode_type` based on `eval_mode`, `min_explore_steps`, and `exploit_freq`
    """

    def __init__(self, name, algo, eval_mode, min_explore_steps, exploit_freq):
        global logger
        logger = logging.getLogger(__name__)
        super().__init__(name, algo, supports_multiple_envs=False)
        self.eval_mode = eval_mode
        self.exploit_freq = exploit_freq
        self.min_explore_steps = min_explore_steps
        self._should_exploit = False

    def pre_episode(self):
        if self.eval_mode:
            self.manager.episode_type = 2
            logger.info("episode_type=2 because of eval_mode")
        else:
            if self.min_explore_steps is not None and self.manager.num_steps < self.min_explore_steps:
                self.manager.episode_type = 0
                logger.info(
                    f"episode_type=0 because num_steps({self.manager.num_steps})<min_explore_steps({self.min_explore_steps}) and no eval_mode")
            else:
                if self.exploit_freq is not None:
                    self.manager.episode_type = int(
                        self.manager.num_episodes % self.exploit_freq == 0)
                    logger.info(
                        f"episode_type={self.manager.episode_type} based on episode_id({self.manager.num_episodes}) and exploit_freq({self.exploit_freq}). No eval mode. min_explore_steps is None or expired.")
                else:
                    self.manager.episode_type = 0
                    logger.info(
                        "episode_type=0 because no eval_mode, min_explore_steps None or expired, exploit_freq is None")
