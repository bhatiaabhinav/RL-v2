from RL import argparser as p

# Algos in these files will get registered on doing `import RL.algorithms`
from . import dqn, random, sac  # noqa

# standard wrap args
p.add_argument('--frameskip', default=1, type=int)
p.add_argument('--artificial_timelimit', default=None, type=int)
p.add_argument('--atari_noop_max', default=30, type=int)
p.add_argument('--atari_frameskip', default=4, type=int)
p.add_argument('--atari_framestack', default=4, type=int)
p.add_argument('--atari_episodic_life', action='store_true')
p.add_argument('--atari_clip_rewards', action='store_true')
p.add_argument('--no_monitor', action='store_true')
p.add_argument('--monitor_video_freq', default=100, type=int)
p.add_argument('--eval_mode', action='store_true')


# standard args
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
p.add_argument('--min_explore_steps', type=int, default=50000)
p.add_argument('--exploit_freq', type=int, default=None)
p.add_argument('--nsteps', type=int, default=1)
p.add_argument('--exp_buff_len', type=int, default=1000000)
p.add_argument('--train_freq', type=int, default=4)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--mb_size', type=int, default=32)
p.add_argument('--td_clip', type=float, default=None)
p.add_argument('--grad_clip', type=float, default=None)
p.add_argument('--no_ignore_done_on_timelimit', action='store_true')

p.add_argument('--conv1', nargs='+', default=[32, 8, 4, 0],
               type=int, help='In format: channels kernel_size stride padding. Specify 0 to skip this layer.')
p.add_argument('--conv2', nargs='+', default=[64, 4, 2, 0],
               type=int, help='In format: channels kernel_size stride padding. Specify 0 to skip this layer.')
p.add_argument('--conv3', nargs='+', default=[64, 3, 1, 0],
               type=int, help='In format: channels kernel_size stride padding. Specify 0 to skip this layer.')
p.add_argument('--hiddens', nargs='*', default=[512], type=int)


# DQN args
p.add_argument('--target_q_freq', type=int, default=10000)
p.add_argument('--target_q_polyak', type=float, default=0)
p.add_argument('--ep', type=float, default=0.1)
p.add_argument('--ep_anneal_steps', type=int, default=1000000)
p.add_argument('--double_dqn', action='store_true')
p.add_argument('--dqn_ptemp', type=float, default=0)
p.add_argument('--perception_wrap', action='store_true')

# SAC args
p.add_argument('--a_lr', type=float, default=1e-4)
p.add_argument('--polyak', type=float, default=0.995)
p.add_argument('--sac_alpha', type=float, default=0.2)
p.add_argument('--fix_alpha', action='store_true')
