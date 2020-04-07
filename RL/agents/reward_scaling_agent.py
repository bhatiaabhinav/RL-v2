import logging

import RL

logger = logging.getLogger(__name__)


class RewardScalingAgent(RL.Agent):
    def __init__(self, name, algo, reward_scaling, cost_scaling):
        super().__init__(name, algo, supports_multiple_envs=False)
        self.reward_scaling = reward_scaling
        self.cost_scaling = reward_scaling
        logging.getLogger(__name__).info(f"Reward Scaling={reward_scaling}, Cost Scaling={cost_scaling}")

    def post_act(self):
        logger.debug('Scaling rewards and costs')
        self.manager.reward *= self.reward_scaling
        self.manager.cost *= self.cost_scaling
