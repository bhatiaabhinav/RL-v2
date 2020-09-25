import logging
from collections import deque

import gym
import numpy as np

logger = logging.getLogger(__name__)


class DummyWrapper(gym.Wrapper):
    def __init__(self, env):
        logger.info("Wrapping with", str(type(self)))
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class CartPoleContinuousWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1., 1., shape=[1])

    def step(self, action):
        if action[0] < 0:
            a = 0
        else:
            a = 1
        return super().step(a)


class ActionSpaceNormalizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._ac_low = self.action_space.low
        self._ac_high = self.action_space.high
        self.action_space = gym.spaces.Box(
            -1, 1, shape=self.env.action_space.shape, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        action_correct = self._ac_low + \
            (self._ac_high - self._ac_low) * (action + 1) / 2
        return super().step(action_correct)


class LinearFrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k=3):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=self.k)
        space = env.observation_space  # type: gym.spaces.Box
        assert len(space.shape) == 1  # can only stack 1-D frames
        self.observation_space = gym.spaces.Box(
            low=np.array(list(space.low) * self.k), high=np.array(list(space.high) * self.k))

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        obs = np.concatenate(self.frames, axis=0)
        assert list(np.shape(obs)) == list(self.observation_space.shape)
        return obs


class BipedalWrapper(gym.Wrapper):
    max_episode_length = 400

    def __init__(self, env):
        super().__init__(env)
        self.frame_count = 0

    def reset(self):
        self.frame_count = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frame_count += 1
        if self.frame_count >= self.max_episode_length:
            # reward -= 100
            done = True
        return obs, reward, done, info


class DiscreteToContinousWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(0, 1, shape=[env.action_space.n])

    def step(self, action):
        a = np.argmax(action)
        return super().step(a)


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        r_total = 0
        obs = None
        d = False
        last_info = {}
        for count in range(self.skip):
            obs, r, d, info = self.env.step(action)
            r_total += r
            for key in info.keys():
                if 'reward' in key.lower():
                    info[key] = info[key] + last_info.get(key, 0)
                last_info = info
            if d:
                break
        return obs, r_total, d, last_info


class CostInfoWrapper(gym.Wrapper):
    def __init__(self, env, cost_extractor_fn):
        super().__init__(env)
        self.cost_extractor_fn = cost_extractor_fn

    def step(self, action):
        o, r, d, i = self.env.step(action)
        i['cost'] = i.get('cost', 0) + self.cost_extractor_fn(o, r, d, i)
        return o, r, d, i


class CostObserveWrapper(gym.Wrapper):
    def __init__(self, env, cost_threshold):
        super().__init__(env)
        self.cost_threshold = max(cost_threshold, 1e-8)
        space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.array(list(space.low) + [0]), high=np.array(list(space.high) + [np.inf]))

    def reset(self):
        o = self.env.reset()  # type: np.ndarray
        return np.append(o, 0 / self.cost_threshold)

    def step(self, action):
        o, r, d, i = self.env.step(action)
        o = np.append(o, i.get('cost', 0) / self.cost_threshold)
        return o, r, d, i
