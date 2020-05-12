'''
Module wide stuff which is not RL specific happens here.
For Example:
- Configuring root logger
- Providing root argparse
'''
from argparse import ArgumentParser

from .core import (Agent, Algorithm, Manager, make_algo, register_algo,  # noqa
                   registered_algos)

argparser = ArgumentParser(conflict_handler='resolve')
