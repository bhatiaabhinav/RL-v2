from RL import argparser as p
from RL import register_algo, stats
from RL.agents.console_print_agent import ConsolePrintAgent
from RL.agents.random_play_agent import RandomPlayAgent
from RL.agents.reward_scaling_agent import RewardScalingAgent
from RL.agents.seeding_agent import SeedingAgent
from RL.agents.simple_render_agent import SimpleRenderAgent
from RL.agents.stats_recording_agent import StatsRecordingAgent
from RL.agents.wandb_agent import WandbAgent

from .standard_wrap_algo import StandardEnvWrapAlgo

p.add_argument('--seed', default=None, type=int)
p.add_argument('--reward_scaling', default=1, type=float)
p.add_argument('--cost_scaling', default=1, type=float)
p.add_argument('--record_unscaled', action='store_true')
p.add_argument('--gamma', default=0.99, type=float)
p.add_argument('--cost_gamma', default=1.0, type=float)
p.add_argument('--record_discounted', action='store_true')
p.add_argument('--frameskip', type=int, default=1)
p.add_argument('--no_render', action='store_true')


class Random(StandardEnvWrapAlgo):
    def setup(self):
        args = p.parse_args()
        self.register_agent(SeedingAgent("SeedingAgent", self, args.seed))
        self.register_agent(RewardScalingAgent(
            "RewardScaler", self, reward_scaling=args.reward_scaling, cost_scaling=args.cost_scaling))

        self.register_agent(RandomPlayAgent(
            "RandomAgent", self, play_for_steps=None))

        self.register_agent(StatsRecordingAgent("StatsRecorder", self, reward_scaling=args.reward_scaling, cost_scaling=args.cost_scaling, record_unscaled=args.record_unscaled,
                                                gamma=args.gamma, cost_gamma=args.cost_gamma, record_undiscounted=not args.record_discounted, frameskip=args.frameskip, should_exploit_fn=lambda: True))  # type: StatsRecordingAgent

        self.register_agent(ConsolePrintAgent("ConsolePrinter", self, lambda: {
            'Steps': self.manager.num_steps,
            'Episodes': self.manager.num_episodes,
            'Len': self.manager.num_episode_steps,
            'R': stats.get_latest('episode_returns'),
            'C': stats.get_latest('episode_cost_returns')
        }, lambda: {
            'Total Steps': self.manager.num_steps,
            'Total Episodes': self.manager.num_episodes,
            'Av Return Per Ep': sum(stats.stats['episode_returns']) / self.manager.num_episodes,
            'Av Cost Per Ep': sum(stats.stats['episode_cost_returns']) / self.manager.num_episodes
        }))

        self.register_agent(WandbAgent('WandbAgent', self,
                                       episode_freq=1, step_freq=None))

        if not args.no_render:
            self.register_agent(SimpleRenderAgent("SimpleRenderAgent", self))


register_algo('Random', Random)
