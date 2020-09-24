import logging
import os
import shutil
import sys

import wandb
from wandb import config

import RL
from RL import argparser as p

p.add_argument('env_id')
p.add_argument('algo_id')
p.add_argument('num_steps_to_run', type=int)
p.add_argument('--tags', nargs='*', default=[], type=str)
DEFAULT_ALGO_SUFFIX = ''
p.add_argument('--algo_suffix', default=DEFAULT_ALGO_SUFFIX)
p.add_argument('--num_episodes_to_run', default=None, type=int)
p.add_argument('--rl_logdir', default=os.getenv('RL_LOGDIR', 'logs'))
p.add_argument('--debug', action="store_true")
p.add_argument('--no_logs', action="store_true")
p.add_argument('--no_gpu', action="store_true")
p.add_argument('--overwrite', action="store_true")

args, unknown = p.parse_known_args()

if args.no_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ' '

logdir = os.path.join(args.rl_logdir, args.env_id,
                      args.algo_id + '_' + args.algo_suffix)
checkpoints_dir = os.path.join(logdir, 'checkpoints')
logfile = os.path.join(logdir, 'logs.log')

deleted_prev_logs = False
if os.path.isdir(logdir):
    if args.overwrite:
        try:
            shutil.rmtree(logdir)
            deleted_prev_logs = True
        except Exception as e:
            raise RuntimeError(
                f"Unable to delete/overwrite existing dir {logdir} for the following reason:\n{e}\n. Can you try removing it manually? Or considering running this algo with a different --algo_suffix?")
    else:
        raise ValueError(
            "You already ran this algo on this env using this algo_suffix. Change something! Or set --overwrite flag to overwrite previous logs")

os.makedirs(logdir)
os.makedirs(checkpoints_dir)
level = logging.DEBUG if args.debug else logging.INFO
if args.no_logs:
    level = logging.WARN
logging.basicConfig(filename=logfile, filemode='w', level=level)
logger = logging.getLogger(__name__)
if deleted_prev_logs:
    logger.warn(
        f'There were logs already present by the name {args.algo_id + "_" + args.algo_suffix} which were overwitten because --overwrite flag was passed. Avoid abusing this flag.')
if args.algo_suffix == '':
    logger.warn(
        f'You are running this algo using the default algo suffix. This is highly discouraged')

with open(os.path.join(logdir, 'command.sh'), 'w') as command_file:
    command_file.write(' '.join(['python', '-m', 'RL'] + sys.argv[1:]))

with open(os.path.join(logdir, 'args.json'), 'w') as args_file:
    args_file.write(str(vars(args)))

wandb.init(dir=logdir, project=args.env_id,
           name=f'{args.algo_id}_{args.algo_suffix}', monitor_gym=True, tags=args.tags)
wandb.config.update(args)

for tag in args.tags:
    wandb.config.update({tag: True})
# wandb.config.update(unknown)
wandb.save(logfile)
# wandb.save(command_file)
# wandb.save(args_file)
wandb.save(checkpoints_dir)
# images_dir = os.path.join(logdir, 'images')
# os.makedirs(images_dir)
# wandb.save(images_dir)

try:
    import RL.algorithms
    if 'BulletEnv-' in args.env_id:
        import pybullet  # noqa
        import pybullet_envs  # noqa
    args = p.parse_args()
    wandb.config.update(args)
    m = RL.Manager(args.env_id, args.algo_id, args.algo_suffix, num_steps_to_run=args.num_steps_to_run,
                   num_episodes_to_run=args.num_episodes_to_run, logdir=logdir)
    m.run()
except Exception as e:
    logger.exception("The script crashed due to an exception")
    raise e
