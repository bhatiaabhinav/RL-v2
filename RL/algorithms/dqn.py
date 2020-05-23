import wandb

from RL import argparser as p
from RL import register_algo
from RL.agents.console_print_agent import ConsolePrintAgent
from RL.agents.dqn_agent import DQNCoreAgent
from RL.agents.episode_type_control_agent import EpisodeTypeControlAgent
from RL.agents.exp_buff_agent import ExperienceBufferAgent
from RL.agents.linear_anneal_agent import LinearAnnealingAgent
from RL.agents.model_copy_agent import ModelCopyAgent
from RL.agents.reward_scaling_agent import RewardScalingAgent
from RL.agents.seeding_agent import SeedingAgent
from RL.agents.simple_render_agent import SimpleRenderAgent
from RL.agents.stats_recording_agent import StatsRecordingAgent

from .standard_wrap_algo import StandardEnvWrapAlgo


class DQN(StandardEnvWrapAlgo):
    def setup(self):
        args = p.parse_args()
        self.register_agent(SeedingAgent("SeedingAgent", self, args.seed))
        self.register_agent(RewardScalingAgent(
            "RewardScaler", self, reward_scaling=args.reward_scaling, cost_scaling=args.cost_scaling))

        self.register_agent(EpisodeTypeControlAgent('EpisodeTypeController', self, args.eval_mode,
                                                    args.min_explore_steps, args.exploit_freq))  # type: EpisodeTypeControlAgent

        if not args.eval_mode:
            exp_buff_agent = self.register_agent(ExperienceBufferAgent(
                "ExpBuffAgent", self, args.nsteps, args.gamma, args.cost_gamma, args.exp_buff_len, None, not args.no_ignore_done_on_timelimit))

        dqn_core_agent = self.register_agent(DQNCoreAgent('DQNCoreAgent', self, list(filter(lambda x: x != [0], [args.conv1, args.conv2, args.conv3])), args.hiddens,
                                                          args.train_freq, args.mb_size, args.double_dqn, args.gamma, args.nsteps,
                                                          args.td_clip, args.grad_clip, args.lr, args.ep, args.eval_mode, args.min_explore_steps, None if args.eval_mode else exp_buff_agent.experience_buffer, args.dqn_ptemp))  # type: DQNCoreAgent

        if not args.eval_mode:
            self.register_agent(LinearAnnealingAgent('EpsilonAnnealer', self, dqn_core_agent,
                                                     'epsilon', args.min_explore_steps, 1, args.ep, args.ep_anneal_steps))

            self.register_agent(ModelCopyAgent('TargetNetCopier', self, dqn_core_agent.q,
                                               dqn_core_agent.target_q, args.target_q_freq, args.target_q_polyak, args.min_explore_steps))

        self.register_agent(StatsRecordingAgent("StatsRecorder", self, reward_scaling=args.reward_scaling, cost_scaling=args.cost_scaling, record_unscaled=args.record_unscaled,
                                                gamma=args.gamma, cost_gamma=args.cost_gamma, record_undiscounted=not args.record_discounted, frameskip=self.frameskip))  # type: StatsRecordingAgent

        self.register_agent(ConsolePrintAgent("ConsolePrinter", self, lambda: {
            'Steps': self.manager.num_steps,
            'Episodes': self.manager.num_episodes,
            'Len': self.manager.num_episode_steps,
            'R': wandb.run.history.row['Episode/Reward'],
            'R(100)': wandb.run.history.row['Average/RPE (Last 100)'],
            'loss': wandb.run.history.row['DQN/Loss'],
            'mb_v': wandb.run.history.row['DQN/Value'],
            'ep': wandb.run.history.row['DQN/Epsilon']
        }, lambda: {
            'Total Steps': self.manager.num_steps,
            'Total Episodes': self.manager.num_episodes,
            'Average RPE': wandb.run.history.row['Average/RPE'],
            'Average CPE': wandb.run.history.row['Average/CPE']
        }))

        if not args.no_render:
            self.register_agent(SimpleRenderAgent("SimpleRenderAgent", self))


register_algo('DQN', DQN)
