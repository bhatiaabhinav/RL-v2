import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from torch.distributions.categorical import Categorical
import wandb
from torch import nn, optim

import RL
from RL.agents.exp_buff_agent import ExperienceBuffer
from RL.utils.standard_models import FFModel
from RL.utils.util_fns import toNpFloat32

logger = logging.getLogger(__name__)
ldebug = logger.isEnabledFor(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, input_shape, convs, hidden_layers, n_actions, ln=False):
        super().__init__()
        self.logits = FFModel(input_shape, convs, list(
            hidden_layers) + [n_actions], ln=ln)
        self.logits.linear_layers[-1].bias.data.fill_(0)

    def forward(self, state):
        logits = Categorical(logits=self.logits(state)).logits
        return logits

    def sample_one_hot(self, state, return_logpi=False):
        act_one_hot = gumbel_softmax(self.logits(state), tau=1)
        if return_logpi:
            dist = Categorical(logits=self.logits(state))
            act = torch.argmax(act_one_hot, dim=-1)  # [n]
            logpi_act = dist.logits[torch.arange(len(state)), act]  # [batch]
            return act_one_hot, logpi_act
        return act_one_hot

    def sample(self, state):
        dist = Categorical(logits=self.logits(state))
        action = dist.sample()
        return action

    def entropy(self, state):
        dist = Categorical(logits=self.logits(state))
        return dist.entropy()


# class Critic(nn.Module):
#     def __init__(self, input_shape, convs, hidden_layers, n_actions, ln=False):
#         super().__init__()
#         self.convs = FFModel(input_shape, convs, [],
#                              flatten=True, apply_act_output=True, ln=ln)
#         self.q = FFModel(
#             (self.convs.output_shape[0] + n_actions,), [], linears=list(hidden_layers) + [1], ln=ln)
#         self.q.linear_layers[-1].bias.data.fill_(0)

#     def forward(self, state, action_one_hot):
#         convs = self.convs(state)
#         s_a_concat = torch.cat((convs, action_one_hot), dim=1)
#         q = self.q(s_a_concat)[:, 0]
#         return q


class CriticDQN(FFModel):
    '''Holds q value for every action'''

    def __init__(self, input_shape, convs, hidden_layers, n_actions, ln=False):
        super().__init__(input_shape, convs, list(
            hidden_layers) + [n_actions], ln=ln)
        self.linear_layers[-1].bias.data.fill_(0)


class SACDiscreteAgent(RL.Agent):
    def __init__(self, name: str, algo: RL.Algorithm, convs: List, hidden_layers: List, train_freq: int, sgd_steps: int, mb_size: int, dqn_mse_loss: bool, gamma: float, nsteps: int, td_clip: float, grad_clip: float, lr: float, a_lr: float, eval_mode: bool, no_train_for_steps: int, exp_buffer: ExperienceBuffer, alpha=0.2, fix_alpha=False):
        super().__init__(name, algo, supports_multiple_envs=False)
        self.convs = convs
        self.hidden_layers = hidden_layers
        self.train_freq = train_freq
        self.sgd_steps = sgd_steps
        self.mb_size = mb_size
        self.dqn_mse_loss = dqn_mse_loss
        self.gamma = gamma
        self.nsteps = nsteps
        self.td_clip = td_clip
        self.grad_clip = grad_clip
        self.lr = lr
        self.a_lr = a_lr
        self.no_train_for_steps = no_train_for_steps
        self.exp_buffer = exp_buffer
        assert alpha >= 0, "alpha must be >= 0"
        self.alpha = alpha
        self.fix_alpha = fix_alpha
        self.min_alpha = alpha
        self.derired_ent = np.log(self.env.action_space.n / 2)
        self.eval_mode = eval_mode
        self._loss, self._a_loss, self._mb_v, self._mb_ent = 0, 0, 0, 0

        logger.info(f'Creating main actor critic networks on device {device}')
        self.critic1 = CriticDQN(self.env.observation_space.shape,
                                 convs, hidden_layers, self.env.action_space.n)
        self.critic1.to(device)
        self.critic2 = CriticDQN(self.env.observation_space.shape,
                                 convs, hidden_layers, self.env.action_space.n)
        self.critic2.to(device)
        logger.info(str(self.critic1))
        self.actor = Actor(self.env.observation_space.shape,
                           convs, hidden_layers, self.env.action_space.n)
        self.actor.to(device)
        logger.info(str(self.actor))
        if ldebug:
            wandb.watch([self.critic1], log='all', log_freq=1000 // train_freq)
            wandb.watch([self.actor], log='all', log_freq=1000 // train_freq)
        if not eval_mode:
            logger.info(f'Creating target critic networks on device {device}')
            self.target_critic1 = CriticDQN(
                self.env.observation_space.shape, convs, hidden_layers, self.env.action_space.n)
            self.target_critic1.to(device)
            self.target_critic2 = CriticDQN(
                self.env.observation_space.shape, convs, hidden_layers, self.env.action_space.n)
            self.target_critic2.to(device)
            logger.info(f'Creating critic optimizer (lr={lr})')
            self.optim_critic = optim.Adam(
                list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.lr)
            logger.info(f'Creating actor optimizer (lr={a_lr})')
            self.optim_actor = optim.Adam(
                self.actor.parameters(), lr=self.a_lr)

    def act(self):
        if self.manager.episode_type > 0:
            with torch.no_grad():
                obs = torch.from_numpy(toNpFloat32(
                    self.manager.obs, True)).to(device)
                action = self.actor.sample(obs).item()
            return action, {}
        else:
            ldebug and logger.debug('Behavior Policy')
            if self.manager.num_steps < self.no_train_for_steps:
                ldebug and logger.debug('Random action')
                return self.env.action_space.sample(), {}
            else:
                ldebug and logger.debug('Stochastic action')
                with torch.no_grad():
                    obs = torch.from_numpy(toNpFloat32(
                        self.manager.obs, True)).to(device)
                    action = self.actor.sample(obs).item()
                return action, {}

    def sgd_update(self):
        ldebug and logger.debug('Training')
        with torch.no_grad():
            states, actions, rewards, dones, info, next_states = self.exp_buffer.random_experiences_unzipped(
                self.mb_size)
            states = torch.from_numpy(toNpFloat32(states)).to(device)
            rewards = torch.from_numpy(toNpFloat32(rewards)).to(device)
            dones = torch.from_numpy(toNpFloat32(dones)).to(device)
            next_states = torch.from_numpy(
                toNpFloat32(next_states)).to(device)
            # next_actions, next_logpis = self.actor.sample_one_hot(
            #     next_states, return_logpi=True)
            # next_target_q = torch.min(self.target_critic1(
            #     next_states, next_actions), self.target_critic2(next_states, next_actions))
            # next_target_v = (next_target_q - self.alpha * next_logpis)

            # desired_q = rewards + (1 - dones) * \
            #     (self.gamma ** self.nsteps) * next_target_v
            next_logpis = self.actor(next_states)
            next_pis = torch.exp(next_logpis)
            next_target_q = torch.min(self.target_critic1(
                next_states), self.target_critic2(next_states))
            next_target_v = torch.sum(
                next_pis * (next_target_q - self.alpha * next_logpis), dim=-1)
            desired_q = (rewards + (1 - dones) * (self.gamma **
                                                  self.nsteps) * next_target_v)

        '''-------------------begin optimization----------------'''
        all_states_idx = torch.arange(self.mb_size).to(device)

        # optimize critic
        # self.optim_critic.zero_grad()
        # actions_one_hot = np.zeros((self.mb_size, self.env.action_space.n))
        # actions_one_hot[all_states_idx, actions] = 1
        # actions_one_hot = torch.from_numpy(
        #     toNpFloat32(actions_one_hot)).to(device)
        # critic_loss = 0.5 * (F.mse_loss(self.critic1(states, actions_one_hot),
        #                                 desired_q) + F.mse_loss(self.critic2(states, actions_one_hot), desired_q))
        # critic_loss.backward()
        # if self.grad_clip:
        #     nn.utils.clip_grad_norm_(list(self.critic1.parameters(
        #     )) + list(self.critic2.parameters()), self.grad_clip)
        # self.optim_critic.step()

        q1, q2 = self.critic1(states), self.critic2(states)
        q1_desired, q2_desired = q1.detach().clone(), q2.detach().clone()
        td1 = desired_q - q1_desired[all_states_idx, actions]
        td2 = desired_q - q2_desired[all_states_idx, actions]
        if self.td_clip is not None:
            ldebug and logger.debug('Doing TD Clipping')
            td1 = torch.clamp(td1, -self.td_clip, self.td_clip)
            td2 = torch.clamp(td2, -self.td_clip, self.td_clip)
        q1_desired[all_states_idx,
                   actions] = q1_desired[all_states_idx, actions] + td1
        q2_desired[all_states_idx,
                   actions] = q2_desired[all_states_idx, actions] + td2

        self.optim_critic.zero_grad()
        if self.dqn_mse_loss:
            critic_loss = 0.5 * (F.mse_loss(q1, q1_desired) +  # noqa
                                F.mse_loss(q2, q2_desired))
        else:
            critic_loss = 0.5 * (F.smooth_l1_loss(q1, q1_desired) +  # noqa
                                F.smooth_l1_loss(q2, q2_desired))
        critic_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(list(self.critic1.parameters(
            )) + list(self.critic2.parameters()), self.grad_clip)
        self.optim_critic.step()

        # optimize actor
        # self.optim_actor.zero_grad()
        # actions_one_hot = self.actor.sample_one_hot(states)
        # q = torch.min(self.critic1(states, actions_one_hot),
        #               self.critic2(states, actions_one_hot))
        # entropy = torch.mean(self.actor.entropy(states))
        # mean_v = torch.mean(q) + self.alpha * entropy
        # actor_loss = -mean_v
        # actor_loss.backward()
        # if self.grad_clip:
        #     nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        # self.optim_actor.step()

        self.optim_actor.zero_grad()
        logpis = self.actor(states)
        pis = torch.exp(logpis)
        q = torch.min(q1_desired, q2_desired)
        entropy = torch.mean(-torch.sum(pis * logpis, dim=-1))
        mean_v = torch.mean(torch.sum(pis * q, dim=-1)) + self.alpha * entropy
        actor_loss = -mean_v
        actor_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.optim_actor.step()

        entropy = entropy.item()
        if not self.fix_alpha:
            logalpha = np.log(self.alpha)
            logalpha -= 0.001 * (entropy - self.derired_ent)
            self.alpha = max(np.exp(logalpha), self.min_alpha)

        '''====================optimization done================'''

        self._loss = critic_loss.item()
        self._a_loss = actor_loss.item()
        self._mb_v = mean_v.item()
        self._mb_ent = entropy

    def post_act(self):
        if self.manager.episode_type < 2 and self.manager.num_steps > self.no_train_for_steps and (self.manager.num_steps - 1) % self.train_freq == 0:
            for sgd_step in range(self.sgd_steps):
                self.sgd_update()

        if (self.manager.num_steps - 1) % 1000 == 0:
            wandb.log({
                'SAC/Loss': self._loss,
                'SAC/A_Loss': self._a_loss,
                'SAC/Value': self._mb_v,
                'SAC/Entropy': self._mb_ent,
                'SAC/Alpha': self.alpha
            }, step=self.manager.num_steps - 1)

    def post_episode(self):
        wandb.log({
            'SAC/Loss': self._loss,
            'SAC/A_Loss': self._a_loss,
            'SAC/Value': self._mb_v,
            'SAC/Entropy': self._mb_ent,
            'SAC/Alpha': self.alpha
        }, step=self.manager.num_steps - 1)
