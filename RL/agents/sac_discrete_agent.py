import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.distributions.categorical import Categorical

import RL
from RL.agents.exp_buff_agent import ExperienceBuffer
from RL.utils.standard_models import FFModel
from RL.utils.util_fns import toNpFloat32

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Actor(FFModel):
    '''Holds logits for every action'''

    def __init__(self, input_shape, convs, hidden_layers, n_actions):
        super().__init__(input_shape, convs, list(hidden_layers) + [n_actions])
        self.linear_layers[-1].bias.data.fill_(0)

    def forward(self, x, deterministic=False, return_pi=False):
        logits = super().forward(x).clamp(-20, 0)
        if deterministic and not return_pi:
            return logits.argmax(dim=-1)
        pis = torch.exp(logits)
        pis = pis / pis.sum(dim=-1, keepdim=True)
        policy = Categorical(logits=logits)
        if deterministic:
            a = pis.argmax(dim=-1)
        else:
            a = policy.sample()

        if return_pi:
            return a, pis
        else:
            return a

    def pi(self, x):
        logpis = super().forward(x).clamp(-20, 0)
        pis = torch.exp(logpis)
        pis = pis / pis.sum(dim=-1, keepdim=True)
        return pis


class Critic(FFModel):
    '''Holds q value for every action'''

    def __init__(self, input_shape, convs, hidden_layers, n_actions):
        super().__init__(input_shape, convs, list(hidden_layers) + [n_actions])
        self.linear_layers[-1].bias.data.fill_(0)


class SACDiscreteAgent(RL.Agent):
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
        self._loss, self._mb_v, self._mb_ent = 0, 0, 0

        obs_space = self.env.observation_space
        ac_space = self.env.action_space

        self.desired_ent = np.log(1 + 0.1 * ac_space.n)
        wandb.config.update({'SAC/Desired Entropy': self.desired_ent})
        logger.info(f'Desired entropy is {self.desired_ent}')

        logger.info(f'Creating Actor on device {device}')
        self.a = Actor(obs_space.shape, convs,
                       hidden_layers, ac_space.n).to(device)
        logger.info(str(self.a))

        logger.info(f'Creating Critics on device {device}')
        self.q1 = Critic(
            obs_space.shape, convs, hidden_layers, ac_space.n).to(device)
        self.q2 = Critic(
            obs_space.shape, convs, hidden_layers, ac_space.n).to(device)
        logger.info(str(self.q1))
        wandb.watch([self.a, self.q1], log='all', log_freq=1000 // train_freq)

        if not eval_mode:
            logger.info(f'Creating target critics on device {device}')
            self.target_q1 = self.q1 = Critic(
                obs_space.shape, convs, hidden_layers, ac_space.n).to(device)
            self.target_q2 = self.q1 = Critic(
                obs_space.shape, convs, hidden_layers, ac_space.n).to(device)
            logger.info(f'Creating optimizer for actor (lr={a_lr})')
            self.optim_a = optim.Adam(self.a.parameters(), lr=a_lr)
            logger.info(f'Creating optimizer for critics (lr={lr})')
            self.q_params = list(self.q1.parameters()) + \
                list(self.q2.parameters())
            self.optim_q = optim.Adam(self.q_params, lr=self.lr)
            if not fix_alpha:
                logger.info(f'Creating optimizer for alpha (lr={lr})')
                self.optim_alpha = optim.Adam([self.logalpha], lr=lr)

    @property
    def alpha(self):
        return torch.exp(self.logalpha)

    def pi(self, q, alpha, return_q_argmax=False):
        q_scaled = q / alpha
        q_scaled_max, q_scaled_argmax = torch.max(
            q_scaled, dim=-1, keepdim=True)
        q_scaled_shifted = q_scaled - q_scaled_max
        pi = torch.exp(q_scaled_shifted)
        pi.clamp_(1e-9, 1)  # don't allow too low probabilities.
        pi = pi / torch.sum(pi, dim=-1, keepdim=True)   # normalize
        if return_q_argmax:
            return pi, q_scaled_argmax
        else:
            return pi

    def act(self):
        if self.manager.episode_type > 0:
            logger.debug('Exploit Action')
            with torch.no_grad():
                obs = torch.from_numpy(toNpFloat32(
                    self.manager.obs, True)).to(device)
                a = self.a(obs, deterministic=True).item()
                return a, {}
        else:
            if self.manager.num_steps < self.no_train_for_steps:
                logger.debug('Random Action')
                return self.env.action_space.sample(), {}
            else:
                logger.debug('Stochastic Action')
                with torch.no_grad():
                    obs = torch.from_numpy(toNpFloat32(
                        self.manager.obs, True)).to(device)
                    a = self.a(obs).item()
                    return a, {}

    def post_act(self):
        if self.manager.num_steps > self.no_train_for_steps and self.manager.step_id % self.train_freq == 0:
            logger.debug('Training')
            with torch.no_grad():
                states, actions, rewards, dones, info, next_states = self.exp_buffer.random_experiences_unzipped(
                    self.mb_size)
                states = torch.from_numpy(toNpFloat32(states)).to(device)
                actions = torch.from_numpy(actions).to(device)
                rewards = torch.from_numpy(toNpFloat32(rewards)).to(device)
                dones = torch.from_numpy(toNpFloat32(dones)).to(device)
                next_states = torch.from_numpy(
                    toNpFloat32(next_states)).to(device)
                next_pis = self.a.pi(next_states)
                next_logpis = torch.log(next_pis)
                next_target_q = torch.min(
                    self.target_q1(next_states), self.target_q2(next_states))
                next_target_v = torch.sum(
                    next_pis * (next_target_q - self.alpha * next_logpis), dim=-1)
                desired_q = rewards + \
                    (1 - dones) * (self.gamma ** self.nsteps) * next_target_v

                q1, q2 = self.q1(states), self.q2(states)
                all_states_idx = torch.arange(self.mb_size).to(device)
                td1 = desired_q - q1[all_states_idx, actions]
                td2 = desired_q - q2[all_states_idx, actions]
                if self.td_clip is not None:
                    logger.debug('Doing TD Clipping')
                    td1 = torch.clamp(td1, -self.td_clip, self.td_clip)
                    td2 = torch.clamp(td2, -self.td_clip, self.td_clip)
                q1[all_states_idx, actions] = q1[all_states_idx, actions] + td1
                q2[all_states_idx, actions] = q2[all_states_idx, actions] + td2

            # Optimize Critic
            self.optim_q.zero_grad()
            critics_loss = 0.5 * \
                (F.smooth_l1_loss(self.q1(states), q1) +  # noqa
                 F.smooth_l1_loss(self.q2(states), q2))
            critics_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.q_params, self.grad_clip)
            self.optim_q.step()

            # Optimize Actor
            self.optim_a.zero_grad()
            q = torch.min(q1, q2)  # type: torch.Tensor
            pis = self.a.pi(states)
            logpis = torch.log(pis)
            entropy = (-pis * logpis).sum(dim=-1).mean()
            mean_v = (pis * q).sum(dim=-1).mean() + self.alpha * entropy
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
            self._mb_v = mean_v.item()
            self._mb_ent = entropy.item()

        if self.manager.step_id % 1000 == 0:
            wandb.log({
                'SAC/Loss': self._loss,
                'SAC/Value': self._mb_v,
                'SAC/Entropy': self._mb_ent,
                'SAC/Alpha': self.alpha.item()
            }, step=self.manager.step_id)

    def post_episode(self):
        wandb.log({
            'SAC/Loss': self._loss,
            'SAC/Value': self._mb_v,
            'SAC/Entropy': self._mb_ent,
            'SAC/Alpha': self.alpha.item()
        }, step=self.manager.step_id)
