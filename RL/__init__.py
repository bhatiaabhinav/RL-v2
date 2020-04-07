'''
Module wide stuff which is not RL specific happens here.
For Example:
- Configuring root logger
- Providing root argparse
'''
from .core import Manager, Algorithm, Agent, register_algo, make_algo, registered_algos  # noqa
from argparse import ArgumentParser

argparser = ArgumentParser(conflict_handler='resolve')
