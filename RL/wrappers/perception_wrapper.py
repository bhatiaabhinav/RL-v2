import logging
import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from RL.utils.standard_models import FFModel
from RL.utils.util_fns import toNpFloat32

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Encoder(FFModel):
    '''Float input to float output'''

    def __init__(self, input_shape, convs, linears):
        super().__init__(input_shape, convs, linears)
        self.bn = nn.BatchNorm2d(input_shape[0], affine=False)

    def forward(self, x):
        # x = self.bn(x)
        x = super().forward(x)
        # x = x.view(x.shape[0], -1, 2)
        # x = torch.softmax(x, 2)
        # x = x.view(x.shape[0], -1)
        x = torch.relu(x)
        return x


class Decoder(FFModel):
    '''Float code to float output'''

    def __init__(self, input_shape, linears, unflatten_shape, deconvs):
        super().__init__(input_shape, [], linears, unflatten_shape, deconvs)

    def forward(self, x):
        x = super().forward(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(FFModel):
    '''To classify (log probability) whether a generated obs is real or not. To classify whether a generated obs is same against a real obs, concat the two obs and pass them as one input'''

    def __init__(self, input_shape, convs, linears):
        super().__init__(input_shape, convs, linears)

    def forward(self, x):
        x = super().forward(x)
        # x = torch.tanh(x)
        return x[:, 0]


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
        self.obs_shape = self.env.observation_space.shape
        self.enc = Encoder(self.obs_shape, convs, hidden_layers + [code_size])
        self.enc.to(device)
        self.cur_obs = None
        self.cur_enc_obs = None
        self.viewer = None
        print(self.enc)
        if len(self.obs_shape) > 1:
            channels, kernels, strides = list(zip(*convs[::-1]))
            channels = channels[1:] + (self.obs_shape[0], )
            deconvs = list(zip(channels, kernels, strides))
        else:
            deconvs = None
        self.dec = Decoder([code_size], hidden_layers[::-1] + list(
            self.enc.flatten_output_shape), self.enc.convs_output_shape, deconvs)
        self.dec.to(device)
        print(self.dec)
        disc_input_shape = list(self.obs_shape)
        disc_input_shape[-1] *= 2
        self.disc = Discriminator(disc_input_shape, convs, hidden_layers + [1])
        self.disc.to(device)
        print(self.disc)
        self.optim_autoenc = optim.Adam(
            list(self.enc.parameters()) + list(self.dec.parameters()))
        self.optim_gan = optim.Adam(self.disc.parameters())
        self.observation_space = gym.spaces.Box(-1, 1, shape=[code_size])

    def autoenc_loss(self, inp, outp, code):
        # return torch.mean((torch.std(inp + 0.0001 * torch.randn_like(inp), axis=0, keepdim=True) * (inp - outp) ** 2)) + torch.mean((torch.std(code + 0.0001 * torch.randn_like(code), axis=0, keepdim=True) * (code - code2) ** 2))
        # return torch.mean((torch.std(inp + 0.0001 * torch.randn_like(inp), axis=0, keepdim=True) * (inp - outp) ** 2)) + torch.mean((code - code2) ** 2)
        # return 0.001 * torch.mean((inp - outp) ** 2) + torch.mean((code - code2) ** 2)
        # return torch.mean((code - code2) ** 2)
        return -torch.mean(self.disc(torch.cat((inp, outp), dim=-1)))

    def disc_loss(self, obs, fakes):
        inp_label0 = torch.cat((obs, fakes), dim=-1)
        inp_label1 = torch.cat((obs, obs), dim=-1)
        inp = torch.cat((inp_label1, inp_label0), dim=0)
        targets = torch.cat((torch.ones(len(obs)), torch.zeros(len(obs))))
        pred_logits = self.disc(inp)
        return F.binary_cross_entropy_with_logits(pred_logits, targets)

    def enc_obs(self, obs):
        obs = torch.from_numpy(toNpFloat32(obs, expand_dims=True)).to(device)
        with torch.no_grad():
            code = self.enc(obs).cpu().detach().numpy()[0]
        return code

    def dec_obs(self, code):
        code = torch.from_numpy(np.expand_dims(code, 0)).to(device)
        with torch.no_grad():
            obs = self.dec(code).cpu().detach().numpy()[0]
        return obs

    def enc_dec_obs(self, obs):
        obs = torch.from_numpy(toNpFloat32(obs, expand_dims=True)).to(device)
        with torch.no_grad():
            code = self.enc(obs)
            decode = self.dec(code).cpu().detach().numpy()[0]
            code = code.cpu().detach().numpy()[0]
        return code, decode

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

        # autoenc train:
        self.enc.train(True)
        self.dec.train(True)
        self.optim_autoenc.zero_grad()
        obs = torch.from_numpy(obs).to(device)
        code = self.enc(obs)
        dec_obs = self.dec(code)
        auto_enc_loss = self.autoenc_loss(obs, dec_obs, code)
        auto_enc_loss.backward(retain_graph=True)
        self.optim_autoenc.step()
        self.enc.train(False)
        self.dec.train(False)

        # gan train:
        self.disc.train(True)
        self.optim_gan.zero_grad()
        disc_loss = self.disc_loss(obs, dec_obs)
        disc_loss.backward()
        self.optim_gan.step()
        self.disc.train(False)

        logger.debug(
            f'perception loss, {auto_enc_loss.item()}, {disc_loss.item()}')
        print(auto_enc_loss.item(), disc_loss.item())

    def add_to_buff(self, obs):
        self.cur_obs = obs
        self.cur_enc_obs = self.enc_obs(obs)
        self.cur_dec_obs = None

        self.states_buffer[self.counter] = obs
        count_new = (self.counter + 1) % self.buffer_size
        if count_new < self.counter:
            self.buffer_full = True
        self.counter = count_new

        if (self.counter - 1) % self.train_interval == 0 and self.buffer_count > 2 * self.mb_size:
            self.train()

        return self.cur_enc_obs

    def reset(self):
        obs = self.env.reset()
        return self.add_to_buff(obs)

    def step(self, action):
        obs, r, d, info = self.env.step(action)
        obs = self.add_to_buff(obs)
        return obs, r, d, info

    def render(self, mode='human', **kwargs):
        if self.cur_dec_obs is None:
            self.cur_dec_obs = self.dec_obs(self.cur_enc_obs)
        obs = np.clip(self.cur_dec_obs * 255, 0, 255).astype(np.uint8)
        img = np.transpose(obs[0:3, :, :], [1, 2, 0])
        if mode == 'human':
            from gym.envs.classic_control.rendering import SimpleImageViewer
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError(f'Render mode {mode} is not supported')

        if kwargs.get('close', False):
            self.close()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
