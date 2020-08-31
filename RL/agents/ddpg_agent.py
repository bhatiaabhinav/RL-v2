import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim

import RL
from RL.agents.exp_buff_agent import ExperienceBuffer
from RL.utils.standard_models import FFModel
from RL.utils.util_fns import toNpFloat32

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, state_shape, action_n, convs, hiddens, action_low=-1, action_high=1, target_deviation=0.2, adaptation_factor=1.01):
        super().__init__()
        self.a_model = FFModel(
            state_shape, convs, hiddens + [action_n], ln=True)
        self.noisy_a_model = FFModel(
            state_shape, convs, hiddens + [action_n], ln=True)
        logger.info(f'Action Space low={action_low}, high={action_high}')
        self.action_activation = nn.Tanh()
        self.action_shift = torch.Tensor((action_low + action_high) / 2)
        self.action_scale = torch.Tensor((action_high - action_low) / 2)
        self.action_shift.requires_grad = False
        self.action_scale.requires_grad = False
        self.sigma = target_deviation
        self.target_deviation = target_deviation
        self.adaptation_factor = adaptation_factor
        logger.info(f'Target deviation is {target_deviation}')
        logger.info(
            f'Action Scale={self.action_scale}, Shift={self.action_shift}')

    def forward(self, s, noisy=False):
        a = self.noisy_a_model(s) if noisy else self.a_model(s)
        a = self.action_activation(a)
        a = a * self.action_scale + self.action_shift
        return a

    def reset_noisy_actor(self):
        for p, noisy_p in zip(self.a_model.parameters(), self.noisy_a_model.parameters()):
            noisy_p.data.copy_(p.data + self.sigma * torch.randn_like(p))

    def adapt_sigma(self, s):
        a = self.a_model(s)
        noisy_a = self.noisy_a_model(s)
        diff = torch.sqrt(F.mse_loss(a, noisy_a))
        if diff < self.target_deviation:
            self.sigma *= self.adaptation_factor
        else:
            self.sigma /= self.adaptation_factor
        return diff


class Critic(nn.Module):
    def __init__(self, state_shape, action_n, convs, hiddens):
        super().__init__()
        self.conv_model = FFModel(state_shape, convs, [], flatten=True)
        inp_n = self.conv_model.output_shape[0] + action_n
        self.q_model = FFModel([inp_n], [], hiddens + [1])
        self.q_model.linear_layers[-1].bias.data.fill_(0)

    def forward(self, s, a):
        s = self.conv_model(s)
        s_a = torch.cat((s, a), dim=-1)
        q = self.q_model(s_a)[:, 0]  # batch_size
        return q


class DDPGAgent(RL.Agent):
    def __init__(self, name, algo, convs, hidden_layers, train_freq, mb_size, gamma, nsteps, td_clip, grad_clip, lr, a_lr, eval_mode, no_train_for_steps, exp_buffer: ExperienceBuffer, noise):
        super().__init__(name, algo, supports_multiple_envs=False)
        self.convs = convs
        self.hidden_layers = hidden_layers
        self.train_freq = train_freq
        self.mb_size = mb_size
        self.gamma = gamma
        self.nsteps = nsteps
        self.td_clip = td_clip
        self.grad_clip = grad_clip
        self.lr = lr
        self.a_lr = a_lr
        self.no_train_for_steps = no_train_for_steps
        self.exp_buffer = exp_buffer
        self._loss, self._mb_v, self._noise = 0, 0, 0

        obs_space = self.env.observation_space
        ac_space = self.env.action_space

        logger.info(f'Creating Actor on device {device}')
        self.a = Actor(
            obs_space.shape, ac_space.shape[0], convs, hidden_layers, ac_space.low, ac_space.high, target_deviation=noise).to(device)
        self.a.reset_noisy_actor()
        logger.info(str(self.a))
        logger.info(f'Creating Critics on device {device}')
        self.q = Critic(obs_space.shape,
                        ac_space.shape[0], convs, hidden_layers).to(device)
        logger.info(str(self.q))
        if logger.isEnabledFor(logging.DEBUG):
            wandb.watch([self.a, self.q], log='all',
                        log_freq=1000 // train_freq)

        if not eval_mode:
            logger.info(f'Creating target critics on device {device}')
            self.target_q = Critic(
                obs_space.shape, ac_space.shape[0], convs, hidden_layers).to(device)
            logger.info(f'Creating optimizer actor (lr={a_lr})')
            self.optim_a = optim.Adam(self.a.parameters(), lr=self.a_lr)
            logger.info(f'Creating optimizer critic (lr={lr})')
            self.optim_q = optim.Adam(self.q.parameters(), lr=self.lr)

    def act(self):
        if self.manager.episode_type > 0:
            logger.debug('Exploit Action')
            with torch.no_grad():
                obs = torch.from_numpy(toNpFloat32(
                    self.manager.obs, True)).to(device)
                a = self.a(obs).cpu().detach().numpy()[0]
                return a, {}
        else:
            if self.manager.num_steps < self.no_train_for_steps:
                logger.debug('Random Action')
                return self.env.action_space.sample()
            else:
                logger.debug('Noisy Action')
                with torch.no_grad():
                    obs = torch.from_numpy(toNpFloat32(
                        self.manager.obs, True)).to(device)
                    a = self.a(obs, noisy=True).cpu().detach().numpy()[0]
                    return a, {}

    def post_act(self):
        if self.manager.num_steps > self.no_train_for_steps and self.manager.step_id % self.train_freq == 0:
            logger.debug('Training')
            with torch.no_grad():
                states, actions, rewards, dones, info, next_states = self.exp_buffer.random_experiences_unzipped(
                    self.mb_size)
                states = torch.from_numpy(toNpFloat32(states)).to(device)
                actions = torch.from_numpy(toNpFloat32(actions)).to(device)
                next_states = torch.from_numpy(
                    toNpFloat32(next_states)).to(device)
                next_actions = self.a(next_states)
                next_target_q = self.target_q(next_states, next_actions)
                next_target_v = next_target_q.cpu().detach().numpy()
                desired_q = rewards + \
                    (1 - dones.astype(np.int)) * \
                    (self.gamma ** self.nsteps) * next_target_v

            # Optimize Critic
            self.optim_q.zero_grad()
            desired_q = torch.from_numpy(toNpFloat32(desired_q)).to(device)
            # TODO: TD Error Clip
            critics_loss = F.mse_loss(self.q(states, actions), desired_q)
            critics_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
            self.optim_q.step()

            # Optimize Actor
            self.optim_a.zero_grad()
            actions = self.a(states)
            q = self.q(states, actions)
            mean_v = torch.mean(q)
            actor_loss = -mean_v
            actor_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.a.parameters(), self.grad_clip)
            self.optim_a.step()

            self._loss = critics_loss.cpu().detach().item()
            self._mb_v = mean_v.cpu().detach().item()

        if self.manager.step_id % 1000 == 0:
            wandb.log({
                'DDPG/Loss': self._loss,
                'DDPG/Value': self._mb_v,
                'DDPG/Noise': self._noise,
                'DDPG/Sigma': self.a.sigma
            }, step=self.manager.step_id)

    def post_episode(self):
        if self.manager.num_steps > self.no_train_for_steps:
            with torch.no_grad():
                states = torch.from_numpy(toNpFloat32(
                    self.exp_buffer.random_states(self.mb_size))).to(device)
                self._noise = self.a.adapt_sigma(states).cpu().detach().item()
                self.a.reset_noisy_actor()

        wandb.log({
            'DDPG/Loss': self._loss,
            'DDPG/Value': self._mb_v,
            'DDPG/Noise': self._noise,
            'DDPG/Sigma': self.a.sigma
        }, step=self.manager.step_id)
