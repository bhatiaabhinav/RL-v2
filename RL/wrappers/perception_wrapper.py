import logging
import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from RL.utils.standard_models import FFModel
from RL.utils.util_fns import toNpFloat32

logger = logging.getLogger(__name__)


class Encoder(FFModel):
    '''Float input to float output'''
    def __init__(self, input_shape, convs, hidden_layers, code_size):
        super().__init__(input_shape, convs, hidden_layers, code_size)
        self.code_size = code_size

    def forward(self, x):
        x = super().forward(x)
        x = torch.tanh(x)
        return x


class Decoder(FFModel):
    '''Float code to float output'''
    def __init__(self, code_size, hidden_layers, deconvs, output_shape):
        x = np.zeros(output_shape)
        n_out = np.squeeze(x).shape[0]
        super().__init__([code_size], [], hidden_layers, n_out)


class PerceptionWrapper(gym.Wrapper):
    def __init__(self, env, convs, hidden_layers, buffer_size, code_size, train_interval, mb_size):
        super().__init__(env)
        self.buffer_size = buffer_size
        self.states_buffer = [None] * buffer_size
        self.counter = 0
        self.buffer_full = False
        self.code_size = code_size
        self.train_interval = train_interval
        self.mb_size = mb_size
        self.enc = Encoder(self.env.observation_space.shape, convs, hidden_layers, code_size)
        self.dec = Decoder(code_size, hidden_layers[::-1], convs[::-1], self.env.observation_space.shape)
        self.optim = optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()))
        self.observation_space = gym.spaces.Box(-1, 1, shape=[code_size])

    def loss(self, inp, outp):
        return F.mse_loss(inp, outp)

    def enc_obs(self, obs):
        obs = torch.from_numpy(toNpFloat32(obs, expand_dims=True))
        with torch.no_grad():
            obs = self.enc(obs).detach().numpy()[0]
        return obs

    @property
    def buffer_count(self):
        if self.buffer_full:
            return self.buffer_size
        else:
            return self.counter

    def train(self):
        buff = self.states_buffer[:self.buffer_count]
        obs = np.asarray(random.sample(buff, self.mb_size))
        obs = toNpFloat32(obs)
        self.optim.zero_grad()
        obs = torch.from_numpy(obs)
        code = self.enc(obs)
        dec_obs = self.dec(code)
        loss = self.loss(obs, dec_obs)
        loss.backward()
        # print(loss)
        self.optim.step()

    def add_to_buff(self, obs):
        self.states_buffer[self.counter] = obs
        count_new = (self.counter + 1) % self.buffer_size
        if count_new < self.counter:
            self.buffer_full = True
        self.counter = count_new

        if (self.counter - 1) % self.train_interval == 0 and self.buffer_count > 2 * self.mb_size:
            self.train()

    def reset(self):
        obs = self.env.reset()
        self.add_to_buff(obs)
        return self.enc_obs(obs)

    def step(self, action):
        obs, r, d, info = self.env.step(action)
        self.add_to_buff(obs)
        return self.enc_obs(obs), r, d, info
