import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="RL-v2",
    version="2.4.3b",
    author="Abhinav Bhatia",
    author_email="bhatiaabhinav93@gmail.com",
    description=("Some RL algorithms"),
    license="MIT",
    keywords="Deep Reinforcement Learning",
    url="https://github.com/bhatiaabhinav/RL-v2",
    packages=['RL'],
    long_description=read('readme.md'),
    python_requires='>=3.6',
    install_requires=[
        'gym[atari,box2d]>=0.17.1',
        'matplotlib>=3.2.0',
        'numpy>=1.18.1',
        'nvidia-ml-py3>=7.352.0',
        'pyglet>=1.5.0',
        'pytest>=5.4.1',
        'scipy>=1.4.1',
        'torch>=1.4.0',
        'wandb>=0.8.36',
        'pybullet>=2.7.8'
    ]
)
