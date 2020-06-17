import logging
import time

import gym
import numpy as np

from .algorithm import Algorithm, make_algo, registered_algos

logger = logging.getLogger(__name__)


class Manager:
    '''runs the registered algo on the registered gym environment. The manager is agnostic to how the algo is implemented. It assumes RL.Algorithm API'''

    def __init__(self, env_id, algo_id, algo_suffix, num_steps_to_run, num_episodes_to_run, logdir):
        global logger
        logger = logging.getLogger(__name__)
        logger.info('Initializing Manager')
        self.env_id = env_id
        self.env = None  # type: gym.Env
        self.algo_id = algo_id
        self.algo_suffix = algo_suffix
        self.num_steps_to_run = num_steps_to_run
        self.num_episodes_to_run = num_episodes_to_run
        self.logdir = logdir
        self.num_steps = 0
        self.step_id = 0
        self.num_episode_steps = 0
        self.episode_step_id = 0
        self.num_episodes = 0
        self.episode_id = 0
        self.episode_type = 0  # 0 means explore, 1 means exploit, 2 means eval
        self.prev_obs = None
        self.action = None
        self.obs = None
        self.reward = 0
        self.sum_of_rewards = 0
        self.cost = 0
        self.sum_of_costs = 0
        self.done = True
        self.info = {}

        logger.debug(f'List of algos found {registered_algos}')
        # Create algo:
        logger.info(f'Creating algo {self.algo_id}')
        self.algo = make_algo(self.algo_id, self)  # type: Algorithm
        # Create wrapped env:
        logger.info(f'Creating env {self.env_id} and wrapping it')
        self.env = self.algo.wrap_env(gym.make(self.env_id))
        # tell the algo to setup
        logger.info(f'Calling algo setup')
        self.algo.setup()

        logger.info(f'Manager initialization done')

    def _should_stop(self):
        return (self.num_steps_to_run is not None and self.num_steps >= self.num_steps_to_run) or (self.num_episodes_to_run is not None and self.num_episodes >= self.num_episodes_to_run)

    def run(self):
        start_time = time.time()
        logger.info('================= Starting run =================')
        logger.info(f'Calling algo start')
        self.algo.start()
        need_reset = True
        while not self._should_stop():
            # do the reset
            if need_reset:
                logger.info(
                    f'------------------ Reseting env. Staring episode #{self.num_episodes} -----------------')
                self.prev_obs = self.obs
                self.obs = self.env.reset()
                self.obs = np.asarray(self.obs)
                self.reward = 0
                self.sum_of_rewards = 0
                self.cost = 0
                self.sum_of_costs = 0
                self.done = False
                self.info = {}
                self.episode_step_id = 0
                self.num_episode_steps = 0
                # pre episode
                logger.info('Calling algo pre_episode')
                self.algo.pre_episode()
            if logger.isEnabledFor(logging.DEBUG):
                if len(self.obs.shape) == 1:
                    logger.debug(f'obs #{self.num_episode_steps}: {self.obs}')
                else:
                    logger.debug(
                        f'obs #{self.num_episode_steps} shape: {self.obs.shape}')
            # pre act
            logger.debug('Calling algo pre_act')
            self.algo.pre_act()
            # act
            logger.debug('Calling algo act')
            self.action, self.action_info = self.algo.act()
            logger.debug(
                f'action #{self.num_episode_steps}: {self.action}. info: {self.action_info}')
            if self.action is None:
                raise RuntimeError(
                    "The algorithm returned no action. The env cannot be stepped")
            # step
            self.prev_obs = self.obs
            logger.debug('Stepping env')
            self.obs, self.reward, self.done, self.info = self.env.step(
                self.action)
            self.info.update(self.action_info)
            self.obs = np.asarray(self.obs)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'reward={self.reward}, done={self.done}, info={self.info}')
            logger.debug(
                "Incrementing rewards, costs, num_steps, num_episode_steps")
            self.sum_of_rewards += self.reward
            self.cost = self.info.get('cost', 0)
            self.sum_of_costs += self.cost
            self.num_episode_steps += 1
            self.num_steps += 1
            # post act
            logger.debug('Calling algo post_act')
            self.algo.post_act()
            # post episode for envs which are done:
            need_reset = self.done
            if self.done:
                logger.info(
                    f"Episode #{self.num_episodes} ended. Length={self.num_episode_steps} Sum_rewards={self.sum_of_rewards}. Sum_costs={self.sum_of_costs}")
                logger.info(
                    f"Incrementing num_episodes to {self.num_episodes + 1}")
                self.num_episodes += 1
                logger.info('Calling algo post_episode')
                self.algo.post_episode()
                self.episode_id += 1
            self.episode_step_id += 1
            self.step_id += 1
        self.episode_step_id -= 1  # reverting the extra increment in the final iteration
        self.step_id -= 1  # reverting the extra increment in final iteration
        logger.info(
            f'------------------ stopping run at num_steps={self.num_steps} and num_episodes={self.num_episodes} ------------------')
        logger.info('Calling algo pre_close')
        self.algo.pre_close()
        logger.info('Closing env')
        self.env.close()
        logger.info('Calling algo post_close')
        self.algo.post_close()
        logger.info('================= Run stopped =================')
        total_time = time.time() - start_time
        logger.info(f'Run Finished in {total_time} seconds')
        print(f'Run Finished in {total_time} seconds')
