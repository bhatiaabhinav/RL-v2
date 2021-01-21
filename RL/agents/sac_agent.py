import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.distributions.normal import Normal

import RL
from RL.agents.exp_buff_agent import ExperienceBuffer
from RL.utils.standard_models import FFModel
from RL.utils.util_fns import toNpFloat32

logger = logging.getLogger(__name__)
ldebug = logger.isEnabledFor(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, state_shape, action_n, convs, hiddens, action_low=-1, action_high=1):
        super().__init__()
        logger.info(f'Action Space low={action_low}, high={action_high}')
        self.conv_model = FFModel(state_shape, convs, [])
        self.mu_model = FFModel(
            self.conv_model.output_shape, [], hiddens + [action_n])
        self.action_activation = nn.Tanh()
        self.logstd_model = FFModel(
            self.conv_model.output_shape, [], hiddens + [action_n])
        self.logstd_model.linear_layers[-1].bias.data.fill_(0)
        self.action_shift = torch.Tensor((action_low + action_high) / 2)
        self.action_scale = torch.Tensor((action_high - action_low) / 2)
        self.action_shift.requires_grad = False
        self.action_scale.requires_grad = False
        logger.info(
            f'Action Scale={self.action_scale}, Shift={self.action_shift}')

    def forward(self, s, deterministic=False, return_logpi=False, rsample=False):
        s = self.conv_model(s)
        mu = self.mu_model(s)
        policy = None
        if not deterministic or return_logpi:
            logstd = torch.clamp(self.logstd_model(s), -20, 2)
            policy = Normal(mu, torch.exp(logstd))

        if deterministic:
            a = mu
        else:
            a = policy.rsample() if rsample else policy.sample()

        if not return_logpi:
            a = self.action_activation(a)
            a = a * self.action_scale + self.action_shift
            return a
        else:
            logpi = torch.sum(policy.log_prob(a), dim=-1)
            a = self.action_activation(a)
            logpi_paper = logpi - torch.sum(torch.log(torch.clamp(1 -  # noqa
                                                     a**2, 0, 1) + 1e-6), dim=-1)
            logpi_spinup = logpi - (2 * (np.log(2) - a -  # noqa
                                  F.softplus(-2 * a))).sum(axis=1)  # noqa
            logpi = logpi_paper
            a = a * self.action_scale + self.action_shift
            return a, logpi


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


class SACAgent(RL.Agent):
    def __init__(self, name, algo, convs, hidden_layers, train_freq, mb_size, gamma, nsteps, td_clip, grad_clip, lr, a_lr, eval_mode, no_train_for_steps, exp_buffer: ExperienceBuffer, alpha=0.2, fix_alpha=False):
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
        assert alpha >= 0, "alpha must be >= 0"
        self.logalpha = torch.log(torch.tensor(alpha + 1e-10)).to(device)
        self.fix_alpha = fix_alpha
        if not fix_alpha:
            self.logalpha.requires_grad = True
        self._loss, self._a_loss, self._mb_v, self._mb_ent = 0, 0, 0, 0

        obs_space = self.env.observation_space
        ac_space = self.env.action_space

        self.desired_ent = -ac_space.shape[0]
        wandb.config.update({'SAC/Desired Entropy': self.desired_ent})
        logger.info(f'Desired entropy is {self.desired_ent}')

        logger.info(f'Creating Actor on device {device}')
        self.a = Actor(
            obs_space.shape, ac_space.shape[0], convs, hidden_layers, ac_space.low, ac_space.high).to(device)
        logger.info(str(self.a))
        logger.info(f'Creating Critics on device {device}')
        self.q1 = Critic(
            obs_space.shape, ac_space.shape[0], convs, hidden_layers).to(device)
        self.q2 = Critic(
            obs_space.shape, ac_space.shape[0], convs, hidden_layers).to(device)
        logger.info(str(self.q1))
        if ldebug:
            wandb.watch([self.a, self.q1], log='all',
                        log_freq=1000 // train_freq)

        if not eval_mode:
            logger.info(f'Creating target critics on device {device}')
            self.target_q1 = Critic(
                obs_space.shape, ac_space.shape[0], convs, hidden_layers).to(device)
            self.target_q2 = Critic(
                obs_space.shape, ac_space.shape[0], convs, hidden_layers).to(device)

            logger.info(f'Creating optimizer actor (lr={a_lr})')
            self.optim_a = optim.Adam(self.a.parameters(), lr=self.a_lr)
            logger.info(f'Creating optimizer critics (lr={lr})')
            self.q_params = list(self.q1.parameters()) + \
                list(self.q2.parameters())
            self.optim_q = optim.Adam(self.q_params, lr=self.lr)
            if not fix_alpha:
                logger.info(f'Creating optimizer for alpha (lr={lr})')
                self.optim_alpha = optim.Adam([self.logalpha], lr=lr)

    @property
    def alpha(self):
        return torch.exp(self.logalpha)

    def act(self):
        if self.manager.episode_type > 0:
            ldebug and logger.debug('Exploit Action')
            with torch.no_grad():
                obs = torch.from_numpy(toNpFloat32(
                    self.manager.obs, True)).to(device)
                a = self.a(obs, deterministic=True).cpu().detach().numpy()[0]
                return a, {}
        else:
            if self.manager.num_steps < self.no_train_for_steps:
                ldebug and logger.debug('Random Action')
                return self.env.action_space.sample(), {}
            else:
                ldebug and logger.debug('Stochastic Action')
                with torch.no_grad():
                    obs = torch.from_numpy(toNpFloat32(
                        self.manager.obs, True)).to(device)
                    a = self.a(obs).cpu().detach().numpy()[0]
                    return a, {}

    def post_act(self):
        if self.manager.num_steps > self.no_train_for_steps and (self.manager.num_steps - 1) % self.train_freq == 0:
            ldebug and logger.debug('Training')
            with torch.no_grad():
                states, actions, rewards, dones, info, next_states = self.exp_buffer.random_experiences_unzipped(
                    self.mb_size)
                states = torch.from_numpy(toNpFloat32(states)).to(device)
                actions = torch.from_numpy(toNpFloat32(actions)).to(device)
                rewards = torch.from_numpy(toNpFloat32(rewards)).to(device)
                dones = torch.from_numpy(toNpFloat32(dones)).to(device)
                next_states = torch.from_numpy(
                    toNpFloat32(next_states)).to(device)
                next_actions, next_logpis = self.a(
                    next_states, return_logpi=True)
                next_target_q = torch.min(
                    self.target_q1(next_states, next_actions), self.target_q2(next_states, next_actions))
                next_target_v = (next_target_q - self.alpha * next_logpis)
                desired_q = rewards + \
                    (1 - dones) * (self.gamma ** self.nsteps) * next_target_v

            # Optimize Critic
            self.optim_q.zero_grad()
            # TODO: TD Error Clip
            critics_loss = 0.5 * (F.mse_loss(self.q1(
                states, actions), desired_q) + F.mse_loss(self.q2(states, actions), desired_q))
            critics_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.q_params, self.grad_clip)
            self.optim_q.step()

            # Optimize Actor
            self.optim_a.zero_grad()
            actions, logpis = self.a(states, return_logpi=True, rsample=True)
            entropy = -torch.mean(logpis)
            q = torch.min(self.q1(states, actions), self.q2(states, actions))
            mean_v = torch.mean(q) + self.alpha * entropy
            actor_loss = -mean_v
            actor_loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.a.parameters(), self.grad_clip)
            self.optim_a.step()

            # Optimize Alpha
            if not self.fix_alpha:
                self.optim_alpha.zero_grad()
                alpha_loss = self.alpha * (entropy.detach() - self.desired_ent)
                alpha_loss.backward()
                self.optim_alpha.step()

            self._loss = critics_loss.item()
            self._a_loss = actor_loss.item()
            self._mb_v = mean_v.item()
            self._mb_ent = entropy.item()

        if (self.manager.num_steps - 1) % 1000 == 0:
            wandb.log({
                'SAC/Loss': self._loss,
                'SAC/A_Loss': self._a_loss,
                'SAC/Value': self._mb_v,
                'SAC/Entropy': self._mb_ent,
                'SAC/Alpha': self.alpha.item()
            }, step=self.manager.num_steps - 1)

    def post_episode(self):
        wandb.log({
            'SAC/Loss': self._loss,
            'SAC/A_Loss': self._a_loss,
            'SAC/Value': self._mb_v,
            'SAC/Entropy': self._mb_ent,
            'SAC/Alpha': self.alpha.item()
        }, step=self.manager.num_steps - 1)
