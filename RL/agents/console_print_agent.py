import logging

import numpy as np

import RL

logger = logging.getLogger(__name__)


class ConsolePrintAgent(RL.Agent):
    def __init__(self, name, algo, stats_dict_fn, concluding_stats_dict_fn):
        global logger
        logger = logging.getLogger(__name__)
        super().__init__(name, algo, supports_multiple_envs=False)
        self.stats_dict_fn = stats_dict_fn
        self.concluding_stats_dict_fn = concluding_stats_dict_fn

    def post_episode(self):
        stats = self.stats_dict_fn()
        print_line = '\t'.join(f'{k}:{np.round(v, 4)}' for k, v in stats.items())
        logger.info(f'Printing to console: {print_line}')
        print(print_line, sep='\t')

    def pre_close(self):
        stats = self.concluding_stats_dict_fn()
        max_keylen = max(len(k) for k in stats.keys())
        print_line_pretty = '\n'.join(f'{k}{" "*(max_keylen - len(k))}\t\t:\t\t{np.round(v, 4)}' for k, v in stats.items())
        print_line = '\t'.join(f'{k}:{np.round(v, 4)}' for k, v in stats.items())
        logger.info(f'Printing conclusion to console: {print_line}')
        print("---------------------------- Concluding Stats -------------------------------")
        print(print_line_pretty)
        print("-----------------------------------------------------------------------------")
