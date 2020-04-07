import logging

import gym
from gym.wrappers import AtariPreprocessing, FrameStack, TimeLimit

import RL
from RL import argparser as p
from RL.wrappers.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,
                                        FireResetEnv, NoopResetEnv)
from RL.wrappers.wrappers import FrameSkipWrapper

logger = logging.getLogger(__name__)

p.add_argument('--frameskip', default=1, type=int)
p.add_argument('--artificial_timelimit', default=None, type=int)
p.add_argument('--atari_noop_max', default=30, type=int)
p.add_argument('--atari_frameskip', default=4, type=int)
p.add_argument('--atari_framestack', default=4, type=int)
p.add_argument('--atari_episodic_life', action='store_true')
p.add_argument('--atari_clip_rewards', action='store_true')


class StandardEnvWrapAlgo(RL.Algorithm):
    def wrap_env(self, env: gym.Env):
        global logger
        logger = logging.getLogger(__name__)
        args = p.parse_args()
        if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) >= 2 and '-v4' in self.manager.env_id:
            logger.info('Atari env detected')
            logger.info('Wrapping with Fire Reset')
            env = FireResetEnv(env)
            logger.info('Wrapping with AtariPreprocessing')
            env = AtariPreprocessing(env, noop_max=args.atari_noop_max, frame_skip=args.atari_frameskip, terminal_on_life_loss=args.atari_episodic_life)
            logger.info('Wrapping with Framestack')
            env = FrameStack(env, args.atari_framestack)
            if args.atari_clip_rewards:
                logger.info('Wrapping with ClipRewards')
                env = ClipRewardEnv(env)
            args.frameskip = args.atari_frameskip
        elif '-ram' in self.manager.env_id and '-v4' in self.manager.env_id:  # for playing atari from ram
            logger.info('Atari RAM env detected')
            logger.info('Wrapping with Fire Reset')
            env = FireResetEnv(env)
            if args.atari_episodic_life:
                logger.info('Wrapping with EpisodicLife')
                env = EpisodicLifeEnv(env)
            logger.info('Wrapping with NoopReset')
            env = NoopResetEnv(env, noop_max=args.atari_noop_max)
            logger.info('Wrapping with Frameskip')
            env = FrameSkipWrapper(env, skip=args.atari_frameskip)
            if args.atari_clip_rewards:
                logger.info('Wrapping with ClipRewards')
                env = ClipRewardEnv(env)
            args.frameskip = args.atari_frameskip
        else:
            if args.frameskip > 1:
                logger.info('Wrapping with Frameskip')
                env = FrameSkipWrapper(env, skip=args.frameskip)
            # TODO: Add Framestack here:
        if args.artificial_timelimit:
            logger.info('Wrapping with Timelimit')
            env = TimeLimit(env, max_episode_steps=args.artificial_timelimit)
        return env
