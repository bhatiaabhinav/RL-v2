import logging
import os
import shutil
import sys

import RL
import RL.algorithms
from RL import argparser as p

p.add_argument('env_id')
p.add_argument('algo_id')
p.add_argument('num_steps_to_run', type=int)
DEFAULT_ALGO_SUFFIX = ''
p.add_argument('--algo_suffix', default=DEFAULT_ALGO_SUFFIX)
p.add_argument('--num_episodes_to_run', default=None, type=int)
p.add_argument('--rl_logdir', default=os.getenv('RL_LOGDIR', 'logs'))
p.add_argument('--debug', action="store_true")
p.add_argument('--no_logs', action="store_true")
p.add_argument('--overwrite', action="store_true")

args = p.parse_args()

logdir = os.path.join(args.rl_logdir, args.env_id,
                      args.algo_id + '_' + args.algo_suffix)
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

with open(os.path.join(logdir, 'command.sh'), 'w') as f:
    f.write(' '.join(['python', '-m', 'RL'] + sys.argv[1:]))

with open(os.path.join(logdir, 'args.json'), 'w') as f:
    f.write(str(vars(args)))

try:
    m = RL.Manager(args.env_id, args.algo_id, args.algo_suffix, num_steps_to_run=args.num_steps_to_run,
                   num_episodes_to_run=args.num_episodes_to_run, logdir=logdir)
    m.run()
except Exception as e:
    logger.exception("The script crashed due to an exception")
    raise e
