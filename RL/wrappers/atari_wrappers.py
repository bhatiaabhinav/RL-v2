from collections import deque

import gym
import numpy as np
from gym import spaces
from PIL import Image


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        last_info = {}
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            for key in info.keys():
                if 'reward' in key.lower():
                    info[key] = info[key] + last_info.get(key, 0)
            last_info = info
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, last_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class MaxEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        obs, reward, done, info = self.env.step(action)
        self._obs_buffer.append(obs)
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        last_info = {}
        done = None
        obs = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            for key in info.keys():
                if 'reward' in key.lower():
                    info[key] = info[key] + last_info.get(key, 0)
            last_info = info
            if done:
                break

        return obs, total_reward, done, last_info


class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array(
            [0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
                                                       resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1))


class FrameStack(gym.Wrapper):
    default_k = 3

    def __init__(self, env, k=None):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        if k is None:
            k = FrameStack.default_k
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        # assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], k * shp[2]))

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
        obs = np.concatenate(self.frames, axis=2)
        assert list(np.shape(obs)) == list(self.observation_space.shape)
        return obs


class SkipAndFrameStack(gym.Wrapper):
    def __init__(self, env, skip=4, k=4):
        """Equivalent to SkipEnv(FrameStack(env, k), skip) but more efficient"""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self._skip = skip
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], k))

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._observation()

    def step(self, action):
        total_reward = 0.0
        last_info = {}
        done = None
        for _ in range(self._skip):
            ob, reward, done, info = self.env.step(action)
            total_reward += reward
            for key in info.keys():
                if 'reward' in key.lower():
                    info[key] = info[key] + last_info.get(key, 0)
            last_info = info
            self.frames.append(ob)
            if done:
                break
        return self._observation(), total_reward, done, last_info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class ObsExpandWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        self.observation_space = gym.spaces.Box(low if np.isscalar(low) else np.asarray(low).item(0), high if np.isscalar(high) else np.asarray(high).item(0),
                                                shape=(
            env.observation_space.shape[0],
            env.observation_space.shape[1] if len(
                env.observation_space.shape) >= 2 else 1,
            env.observation_space.shape[2] if len(
                env.observation_space.shape) >= 3 else 1
        ))

    def reset(self):
        ob = super().reset()
        if ob.ndim == 1:
            ob = ob[:, np.newaxis, np.newaxis]
        elif ob.ndim == 2:
            ob = ob[:, :, np.newaxis]
        return ob

    def step(self, action):
        ob, r, d, _ = super().step(action)
        if ob.ndim == 1:
            ob = ob[:, np.newaxis, np.newaxis]
        elif ob.ndim == 2:
            ob = ob[:, :, np.newaxis]
        return ob, r, d, _


class NoopFrameskipWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.FRAMESKIP_ON_NOOP = 2
        self.gamma = gamma
        self.noop_phase = False
        self.skipped_already = 0

    def _is_noop(self, action):
        return action == 0

    def step(self, action):
        if self._is_noop(action):
            R = 0
            last_info = {}
            ob = None
            d = False
            for i in range(self.FRAMESKIP_ON_NOOP):
                ob, r, d, info = super().step(action)
                R += r
                for key in info.keys():
                    if 'reward' in key.lower():
                        info[key] = info[key] + last_info.get(key, 0)
                    last_info = info
                if d:
                    break
            return ob, R, d, last_info
        else:
            return super().step(action)


class BreakoutContinuousActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-1, 1, shape=[1])

    def step(self, action):
        if action < -1 / 3:
            action = 3
        elif action >= -1 / 3 and action <= 1 / 3:
            action = 0
        else:
            action = 2
        return self.env.step(action)


def wrap_deepmind(env, episode_life=True, clip_rewards=True):
    """Configure environment for DeepMind-style Atari.

    Note: this does not include frame stacking!"""
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env


def wrap_deepmind_with_framestack(env, episode_life=True, clip_rewards=True, framestack_k=4, frameskip_k=4, noop_max=30):
    """Configure environment for DeepMind-style Atari."""
    assert 'NoFrameskip' in env.spec.id  # required for DeepMind-style skip
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frameskip_k)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=framestack_k)
    return env


wrap_atari = wrap_deepmind_with_framestack
