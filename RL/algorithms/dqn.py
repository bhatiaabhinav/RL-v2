import os.path as osp

import numpy as np
from gym.wrappers import Monitor

from RL import argparser as p
from RL import register_algo
from RL.agents.console_print_agent import ConsolePrintAgent
from RL.agents.dqn_agent import DQNCoreAgent
from RL.agents.exp_buff_agent import ExperienceBufferAgent
from RL.agents.exploit_control_agent import ExploitControlAgent
from RL.agents.linear_anneal_agent import LinearAnnealingAgent
from RL.agents.model_copy_agent import ModelCopyAgent
from RL.agents.reward_scaling_agent import RewardScalingAgent
from RL.agents.seeding_agent import SeedingAgent
from RL.agents.simple_render_agent import SimpleRenderAgent
from RL.agents.stats_recording_agent import StatsRecordingAgent
from RL.wrappers.perception_wrapper import PerceptionWrapper  # noqa

from .standard_wrap_algo import (StandardEnvWrapAlgo,
                                 capped_quadratic_video_schedule)

p.add_argument('--seed', default=None, type=int)
p.add_argument('--reward_scaling', default=1, type=float)
p.add_argument('--cost_scaling', default=1, type=float)
p.add_argument('--record_unscaled', action='store_true')
p.add_argument('--gamma', default=0.99, type=float)
p.add_argument('--cost_gamma', default=1.0, type=float)
p.add_argument('--record_discounted', action='store_true')
p.add_argument('--frameskip', type=int, default=1)
p.add_argument('--no_render', action='store_true')

p.add_argument('--eval_mode', action='store_true')
p.add_argument('--min_explore_steps', type=int, default=10000)
p.add_argument('--exploit_freq', type=int, default=None)
p.add_argument('--nsteps', type=int, default=1)
p.add_argument('--exp_buff_len', type=int, default=100000)
p.add_argument('--train_freq', type=int, default=1)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--target_q_freq', type=int, default=2000)
p.add_argument('--target_q_tau', type=float, default=1)
p.add_argument('--ep', type=float, default=0.01)
p.add_argument('--ep_anneal_steps', type=int, default=50000)
p.add_argument('--mb_size', type=int, default=32)
p.add_argument('--double_dqn', action='store_true')
p.add_argument('--td_clip', type=float, default=None)
p.add_argument('--grad_clip', type=float, default=None)
p.add_argument('--no_ignore_done_on_timelimit', action='store_true')
p.add_argument('--death_cost', type=float, default=0)
p.add_argument('--dqn_ptemp', type=float, default=0)
p.add_argument('--perception_wrap', action='store_true')


class DQN(StandardEnvWrapAlgo):
    def wrap_env(self, env):
        env = super().wrap_env(env)
        args = p.parse_args()
        if args.perception_wrap:
            env = PerceptionWrapper(
                env, [(64, 6, 2, 0), (64, 6, 2, 2), (64, 6, 2, 2)], [], 20000, 1024, 4, 32)
            if not args.no_monitor:
                env = Monitor(env, osp.join(self.manager.logdir, 'perception_monitor'), video_callable=lambda ep_id: capped_quadratic_video_schedule(
                    ep_id, args.monitor_video_freq), force=True, mode='evaluation' if args.eval_mode else 'training')
        return env

    def setup(self):
        args = p.parse_args()
        self.register_agent(SeedingAgent("SeedingAgent", self, args.seed))
        self.register_agent(RewardScalingAgent(
            "RewardScaler", self, reward_scaling=args.reward_scaling, cost_scaling=args.cost_scaling))

        exploit_controller = self.register_agent(ExploitControlAgent(
            'ExploitController', self, args.eval_mode, args.min_explore_steps, args.exploit_freq))  # type: ExploitControlAgent

        exp_buff_agent = self.register_agent(ExperienceBufferAgent(
            "ExpBuffAgent", self, args.nsteps, args.gamma, args.cost_gamma, args.exp_buff_len, None, not args.no_ignore_done_on_timelimit))

        dqn_core_agent = self.register_agent(DQNCoreAgent('DQNCoreAgent', self, [(64, 6, 2, 0), (64, 6, 2, 2), (64, 6, 2, 2)], [512], args.train_freq, args.mb_size, args.double_dqn, args.gamma, args.nsteps,
                                                          args.td_clip, args.grad_clip, args.lr, args.ep, lambda: exploit_controller.should_exploit, args.eval_mode, args.min_explore_steps, exp_buff_agent.experience_buffer, args.dqn_ptemp, args.death_cost))  # type: DQNCoreAgent

        self.register_agent(LinearAnnealingAgent('EpsilonAnnealer', self, dqn_core_agent,
                                                 'epsilon', args.min_explore_steps, 1, args.ep, args.ep_anneal_steps))

        self.register_agent(ModelCopyAgent('TargetNetCopier', self, dqn_core_agent.q,
                                           dqn_core_agent.target_q, args.target_q_freq, args.target_q_tau, args.min_explore_steps))

        stats_agent = self.register_agent(StatsRecordingAgent("StatsRecorder", self, reward_scaling=args.reward_scaling, cost_scaling=args.cost_scaling, record_unscaled=args.record_unscaled,
                                                              gamma=args.gamma, cost_gamma=args.cost_gamma, record_undiscounted=not args.record_discounted, frameskip=args.frameskip, should_exploit_fn=lambda: True))  # type: StatsRecordingAgent

        self.register_agent(ConsolePrintAgent("ConsolePrinter", self, lambda: {
            'Steps': self.manager.num_steps,
            'Episodes': self.manager.num_episodes,
            'Len': self.manager.num_episode_steps,
            'R': stats_agent.get_one('episode_returns'),
            'R(100)': np.mean(stats_agent.stats['episode_returns'][-100:]),
            'loss': stats_agent.get_one('loss'),
            'mb_v': stats_agent.get_one('mb_v'),
            'ep': dqn_core_agent.epsilon
        }, lambda: {
            'Total Steps': self.manager.num_steps,
            'Total Episodes': self.manager.num_episodes,
            'Av Return Per Ep': sum(stats_agent.stats['episode_returns']) / self.manager.num_episodes,
            'Av Cost Per Ep': sum(stats_agent.stats['episode_cost_returns']) / self.manager.num_episodes
        }))

        if not args.no_render:
            self.register_agent(SimpleRenderAgent("SimpleRenderAgent", self))


register_algo('DQN', DQN)


# Standard Scripts:
'''
python -m RL CartPole-v0 DQN 20000 --algo_suffix=test --seed=0 --nsteps=3 --ep_anneal_steps=10000 --no_render
'''

'''
python -m RL BreakoutNoFrameskip-v4 DQN 1000000 --algo_suffix=nsteps3_f4_big --seed=0 --nsteps=3 --train_freq=4 --no_render --monitor_video_freq=100
'''
