import logging
import os.path as osp

import gym
from gym.wrappers import AtariPreprocessing, FrameStack, Monitor, TimeLimit

import RL
from RL import argparser as p
from RL.wrappers.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,
                                        FireResetEnv, NoopResetEnv)
from RL.wrappers.wrappers import FrameSkipWrapper

logger = logging.getLogger(__name__)


def capped_quadratic_video_schedule(episode_id, cap):
    if episode_id < cap:
        return int(round(episode_id ** (1. / 2))) ** 2 == episode_id
    else:
        return episode_id % cap == 0


class StandardEnvWrapAlgo(RL.Algorithm):
    def wrap_env(self, env: gym.Env):
        global logger
        logger = logging.getLogger(__name__)
        args = p.parse_args()
        if args.artificial_timelimit:
            logger.info('Wrapping with Timelimit')
            env = TimeLimit(env, max_episode_steps=args.artificial_timelimit)
        if not args.no_monitor:
            env = Monitor(env, osp.join(
                self.manager.logdir, 'openai_monitor'), video_callable=lambda ep_id: capped_quadratic_video_schedule(ep_id, args.monitor_video_freq), force=True, mode='evaluation' if args.eval_mode else 'training')
        if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) >= 2 and '-v4' in self.manager.env_id:
            logger.info('Atari env detected')
            logger.info('Wrapping with Fire Reset')
            env = FireResetEnv(env)
            logger.info('Wrapping with AtariPreprocessing')
            env = AtariPreprocessing(env, noop_max=args.atari_noop_max,
                                     frame_skip=args.atari_frameskip, terminal_on_life_loss=args.atari_episodic_life)
            logger.info('Wrapping with Framestack')
            env = FrameStack(env, args.atari_framestack)
            if args.atari_clip_rewards:
                logger.info('Wrapping with ClipRewards')
                env = ClipRewardEnv(env)
            self.frameskip = args.atari_frameskip
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
            self.frameskip = args.atari_frameskip
        else:
            if args.frameskip > 1:
                logger.info('Wrapping with Frameskip')
                env = FrameSkipWrapper(env, skip=args.frameskip)
            self.frameskip = args.frameskip
            # TODO: Add Framestack here:
        return env
