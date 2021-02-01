from RL.utils.util_fns import update_mean_std
import logging
import sys
from typing import List
import itertools

import numpy as np

import RL

ids = 0
logger = logging.getLogger(__name__)
ldebug = logger.isEnabledFor(logging.DEBUG)


class Experience:
    def __init__(self, state, action, reward, done, info, next_state, cost=0, **kwargs):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
        self.next_state = next_state
        self.cost = cost
        self.other_args = kwargs
        self.id = 0

    def __sizeof__(self):
        return sys.getsizeof(self.state) + sys.getsizeof(self.action) + \
            sys.getsizeof(self.reward) + sys.getsizeof(self.done) + \
            sum([sys.getsizeof(v) for v in self.info.values()]) + sys.getsizeof(self.next_state) + \
            sys.getsizeof(self.cost) + sum([sys.getsizeof(v)
                                            for v in self.other_args.values()]) + sys.getsizeof(self.id)


class ExperienceBuffer:
    '''A circular buffer to hold experiences'''

    def __init__(self, length=1e6, size_in_bytes=None):
        self.buffer = []  # type: List[Experience]
        self.buffer_length = length
        self.count = 0
        self.size_in_bytes = size_in_bytes
        self.next_index = 0
        self.reward_stats = (0, 0, 0)

    @property
    def reward_mean(self):
        return self.reward_stats[0]

    @property
    def reward_std(self):
        return self.reward_stats[1]

    def __len__(self):
        return self.count

    def add(self, exp: Experience):
        if self.count == 0:
            if self.size_in_bytes is not None:
                self.buffer_length = self.size_in_bytes / sys.getsizeof(exp)
            self.buffer_length = int(self.buffer_length)
            logging.getLogger(__name__).info('Initializing experience buffer of length {0}. Est. memory: {1} MB'.format(
                self.buffer_length, self.buffer_length * sys.getsizeof(exp) // (1024 * 1024)))
            self.buffer = [None] * self.buffer_length
        if self.count == self.buffer_length:
            self.reward_stats = update_mean_std(
                *self.reward_stats, self.buffer[self.next_index].reward, remove=True)
        self.reward_stats = update_mean_std(*self.reward_stats, exp.reward)
        self.buffer[self.next_index] = exp
        self.next_index = (self.next_index + 1) % self.buffer_length
        self.count = min(self.count + 1, self.buffer_length)
        assert self.count == self.reward_stats[-1]
        global ids
        self.id = ids
        ids += 1

    def to_tuples(self, exps, return_costs=False):
        for exp in exps:
            if return_costs:
                yield (exp.state, exp.action, exp.reward, exp.done, exp.info, exp.next_state, exp.cost, *exp.other_args.values())
            else:
                yield (exp.state, exp.action, exp.reward, exp.done, exp.info, exp.next_state, *exp.other_args.values())

    def random_experiences(self, count):
        indices = np.random.randint(0, self.count, size=count)
        for i in indices:
            yield self.buffer[i]

    def random_experiences_unzipped(self, count, return_costs=False):
        exps = self.random_experiences(count)
        ans = tuple(np.asarray(tup) for tup in zip(
            *self.to_tuples(exps, return_costs=return_costs)))
        return ans

    def random_rollouts(self, count, rollout_size):
        starting_indices = np.random.randint(
            0, self.count - rollout_size, size=count)
        for i in starting_indices:
            yield self.buffer[i:i + rollout_size]

    def random_rollouts_unzipped(self, count, rollout_size, return_costs=False, return_flattened=False):
        '''batch major'''
        rollouts = self.random_rollouts(count, rollout_size)
        rollouts_flattened = itertools.chain(*rollouts)
        # if linearized `states` is of shape: [count*rollout, 1, 84, 84]
        # need to reshape it to [count, rollout, 1, 84, 84]
        rollouts_unzipped_flattened = tuple(np.asarray(tup) for tup in zip(
            *self.to_tuples(rollouts_flattened, return_costs=return_costs)))  # states, actions, rewards, dones, infos, next_states, etc.
        if return_flattened:
            return rollouts_unzipped_flattened
        else:
            rollouts_unzipped = tuple(tup.reshape(
                count, rollout_size, *tup.shape[1:]) for tup in rollouts_unzipped_flattened)
            return rollouts_unzipped

    def random_states(self, count):
        experiences = list(self.random_experiences(count))
        return np.asarray([e.state for e in experiences])


class ExperienceBufferAgent(RL.Agent):
    def __init__(self, name, algo, nsteps=None, gamma=1, cost_gamma=1, buffer_length=None, buffer_size_MB=None, ignore_done_on_timelimit=False):
        super().__init__(name, algo, False)
        self.nsteps = nsteps
        self.buffer_length = buffer_length
        self.buffer_size_MB = buffer_size_MB
        self.gamma = gamma
        self.cost_gamma = cost_gamma
        if self.buffer_length is None:
            self.experience_buffer = ExperienceBuffer(
                size_in_bytes=self.buffer_size_MB * (1024**2))
        else:
            self.experience_buffer = ExperienceBuffer(
                length=self.buffer_length)
        self.nstep_buffer = []  # type: List[Experience]
        if hasattr(self.env.spec, 'max_episode_steps') and self.env.spec.max_episode_steps and ignore_done_on_timelimit:
            logging.getLogger(__name__).warn(
                "{0}: This is a timelimit environment. Done signal will be ignored if either TimeLimit.truncated info is provided or if not provided, then if length of episode is same as env.spec.max_episode_steps. Set --no_ignore_done_on_timelimit flag to disable this behavior.".format(self.name))
            self.ignore_done_on_timelimit = True
        else:
            self.ignore_done_on_timelimit = False

    def add_to_experience_buffer(self, exp: Experience):
        if self.nsteps == 1:
            self.experience_buffer.add(exp)
        else:
            nstep_buffer = self.nstep_buffer
            reward_to_prop_back = exp.reward
            cost_to_prop_back = exp.cost
            for old_exp in reversed(nstep_buffer):
                if old_exp.done:
                    break
                reward_to_prop_back = self.gamma * reward_to_prop_back
                cost_to_prop_back = self.cost_gamma * cost_to_prop_back
                old_exp.reward += reward_to_prop_back
                old_exp.cost += cost_to_prop_back
                old_exp.next_state = exp.next_state
                old_exp.done = exp.done
            nstep_buffer.append(exp)
            if len(nstep_buffer) >= self.nsteps:
                self.experience_buffer.add(nstep_buffer.pop(0))
                assert len(nstep_buffer) == self.nsteps - 1

    def get_done(self):
        done = self.manager.done
        if done and self.ignore_done_on_timelimit:
            info = self.manager.info
            if 'TimeLimit.truncated' in info:
                done = not info.get('TimeLimit.truncated')
            elif self.manager.num_episode_steps == self.env.spec.max_episode_steps:
                done = False
            if not done:
                ldebug and logger.debug(
                    'Done happened due to timelimit so recorded as false')
        return done

    def post_act(self):
        exp = Experience(self.manager.prev_obs, self.manager.action, self.manager.reward,
                         self.get_done(), self.manager.info, self.manager.obs, cost=self.manager.cost)
        self.add_to_experience_buffer(exp)
        ldebug and logger.debug(
            f'{self.name}: Added an exp to replay buff. Count={self.experience_buffer.count}')
