import logging
from typing import List

import gym

from .agent import Agent

logger = logging.getLogger(__name__)

registered_algos = {}


def register_algo(algo_id, Cls, *args, **kwargs):
    registered_algos[algo_id] = {
        'class': Cls,
        'args': args,
        'kwargs': kwargs
    }


def make_algo(algo_id, manager):
    '''factory method'''
    if algo_id not in registered_algos:
        try:
            raise KeyError(
                f'No algorithm registered by id {algo_id}. Check list of registered algorithms using RL.registered_algos.keys()')
        except KeyError:
            logging.getLogger(__name__).exception('Exception happened')
            raise
    Cls = registered_algos[algo_id]['class']
    args = registered_algos[algo_id]['args']
    kwargs = registered_algos[algo_id]['kwargs']
    return Cls(algo_id, manager, *args, **kwargs)


class Algorithm:
    '''
    Base class for implmenting RL algorithms. Concrete implementations would typically override `wrap_env(env)`, `setup` (called after creating a wrapped env) and `act`.
    By defaut, this base class also provides an agent facility for creating algorithms. Create agents subclassing the RL.Agent API and register them in the setup method using `self.register_agent(agent)`. Then this algorithm will automatically call the agents' methods at appropriate times. Read more about the control flow in the readme.
    '''

    def __init__(self, algo_id, manager, supports_multiple_envs=False):
        global logger
        logger = logging.getLogger(__name__)
        self.algo_id = algo_id
        from .manager import Manager
        self.manager = manager  # type: Manager
        self.agents = []  # type: List[Agent]
        self._agent_name_to_agent_map = {}
        self.supports_multiple_envs = supports_multiple_envs

    def wrap_env(self, env: gym.Env) -> gym.Env:
        '''meant to be overriden'''
        return env

    def setup(self):
        pass

    def register_agent(self, agent: Agent):
        logger.info(f'Registering agent {agent.name}')
        self.agents.append(agent)
        self._agent_name_to_agent_map[agent.name] = Agent
        return agent

    def enabled_agents(self):
        return list(filter(lambda agent: agent.enabled, self.agents))

    def get_agent(self, name):
        '''returns none if no agent found by this name'''
        return self._agent_name_to_agent_map.get(name)

    def get_agent_by_type(self, typ):
        '''returns list of agents which are instances of subclass of typ'''
        lis = list(filter(lambda agent: isinstance(agent, typ), self.agents))
        if len(lis) == 1:
            return lis[0]

    def start(self):
        [agent.start() for agent in self.enabled_agents()]

    def pre_episode(self):
        [agent.pre_episode() for agent in self.enabled_agents()]

    def pre_episode_multienv(self, env_id_nos):
        if not self.supports_multiple_envs:
            raise NotImplementedError(
                f"Algorithm {self.algo_id} does not support multiple parallel envs yet.")
        else:
            [agent.pre_episode_multienv() for agent in self.enabled_agents()]

    def pre_act(self):
        [agent.pre_act() for agent in self.enabled_agents()]

    def act(self):
        actions_per_agent = [agent.act() for agent in self.enabled_agents()]
        actions_per_agent = list(
            filter(lambda x: x is not None, actions_per_agent))
        if len(actions_per_agent) == 0:
            return None
        else:
            return actions_per_agent[-1]

    def act_multienv(self):
        if not self.supports_multiple_envs:
            raise NotImplementedError(
                f"Algorithm {self.algo_id} does not support multiple parallel envs yet.")
        else:
            # TODO: Check this logic. The last agent which acted might not have acted in all envs.
            actions_per_agent = [agent.act_multienv()
                                 for agent in self.enabled_agents()]
            actions_per_agent = list(
                filter(lambda x: x is not None, actions_per_agent))
            if len(actions_per_agent) == 0:
                return None
            else:
                return actions_per_agent[-1]

    def post_act(self):
        [agent.post_act() for agent in self.enabled_agents()]

    def post_episode(self):
        [agent.post_episode() for agent in self.enabled_agents()]

    def post_episode_multienv(self, env_id_nos):
        if not self.supports_multiple_envs:
            raise NotImplementedError(
                f"Algorithm {self.algo_id} does not support multiple parallel envs yet.")
        else:
            [agent.post_episode_multienv() for agent in self.enabled_agents()]

    def pre_close(self):
        [agent.pre_close() for agent in self.enabled_agents()]

    def post_close(self):
        [agent.post_close() for agent in self.enabled_agents()]
