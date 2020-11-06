import logging
from typing import List

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
ldebug = logger.isEnabledFor(logging.DEBUG)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DQNModel(nn.Module):
    def __init__(self, input_shape, convs, hidden_layers, n_actions, dueling=False, noisy=False, ln=False):
        super().__init__()
        self.noisy = noisy
        self.dueling = dueling

        self.convs = FFModel(
            input_shape, convs, [], flatten=True, ln=ln, apply_act_output=True)

        if dueling:
            halved_hiddens = list(np.ceil(
                (np.asarray(hidden_layers) / 2)).astype(np.int)) if len(hidden_layers) > 0 else []
            self.v = FFModel(self.convs.output_shape, [],
                             halved_hiddens + [1], ln=ln)
            self.v.linear_layers[-1].bias.data.fill_(0)
            self.a = FFModel(self.convs.output_shape, [],
                             halved_hiddens + [n_actions], ln=ln)
            self.a.linear_layers[-1].bias.data.fill_(0)
        else:
            self.q = FFModel(self.convs.output_shape, [], list(
                hidden_layers) + [n_actions], ln=ln)
            self.q.linear_layers[-1].bias.data.fill_(0)

        if self.noisy:
            self.logstd = FFModel(
                self.convs.output_shape, [], hidden_layers + [n_actions], ln=ln)
            self.logstd.linear_layers[-1].bias.data.fill_(0)

    def forward(self, x: torch.Tensor, noisy=False):
        convs = self.convs(x)  # [batch, hidden[-2]]

        if self.dueling:
            v = self.v(convs)  # [batch, 1]
            a = self.a(convs)
            '''combine value and advantage theoretically correctly:'''
            # q = v + a - torch.max(a, 1, keepdim=True)[0]  # [batch, n]
            '''combine value and advantage in more stable way:'''
            q = v + a - torch.mean(a, 1, keepdim=True)
        else:
            q = self.q(convs)

        noisy = noisy and self.noisy
        if noisy:
            logstd = torch.clamp(self.logstd(convs), -20, 2)
            with torch.no_grad():
                av_a = torch.mean(
                    q - torch.max(q, axis=-1, keepdims=True)[0]).detach()
            std = 0.5 * abs(av_a) * torch.exp(logstd)
            noise = std * torch.normal(0, torch.ones_like(std))
            q_noisy = q + noise
        else:
            std = torch.zeros(1)
            noise = torch.zeros(1)
            q_noisy = q
        return (q_noisy, q, std)


class DQNCoreAgent(RL.Agent):
    def __init__(self, name: str, algo: RL.Algorithm, convs: List, hidden_layers: List, train_freq: int, sgd_steps: int, mb_size: int, double_dqn: bool, dueling_dqn: bool, gamma: float, nsteps: int, td_clip: float, grad_clip: float, lr: float, epsilon: float, noisy_explore: bool, eval_mode: bool, no_train_for_steps: int, exp_buffer: ExperienceBuffer, policy_temperature: float = 0):
        super().__init__(name, algo, supports_multiple_envs=False)
        self.convs = convs
        self.hidden_layers = hidden_layers
        self.train_freq = train_freq
        self.sgd_steps = sgd_steps
        self.mb_size = mb_size
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.noisy_explore = noisy_explore
        self.gamma = gamma
        self.nsteps = nsteps
        self.td_clip = td_clip
        self.grad_clip = grad_clip
        self.lr = lr
        self.no_train_for_steps = no_train_for_steps
        self.exp_buffer = exp_buffer
        self.epsilon = epsilon
        self.noisy_explore = noisy_explore
        self.eval_mode = eval_mode
        self.ptemp = policy_temperature
        self._loss, self._mb_v, self._mb_q_std = 0, 0, 0

        logger.info(f'Creating main network on device {device}')
        self.q = DQNModel(self.env.observation_space.shape,
                          convs, hidden_layers, self.env.action_space.n, dueling=dueling_dqn, noisy=noisy_explore)
        self.q.to(device)
        logger.info(str(self.q))
        if ldebug:
            wandb.watch([self.q], log='all', log_freq=1000 // train_freq)
        if not eval_mode:
            logger.info(f'Creating target network on device {device}')
            self.target_q = DQNModel(self.env.observation_space.shape, convs, hidden_layers,
                                     self.env.action_space.n, dueling=dueling_dqn, noisy=noisy_explore)
            self.target_q.to(device)
            logger.info(f'Creating optimizer (lr={lr})')
            self.optim = optim.Adam(self.q.parameters(), lr=self.lr)

    def pi(self, q, alpha=0):
        if alpha == 0:
            p = torch.zeros_like(q)
            q_argmax = torch.argmax(q, dim=-1)
            p[torch.arange(len(q)), q_argmax] = 1
        else:
            q_max, q_argmax = torch.max(q, dim=-1, keepdim=True)
            q = q - q_max
            q /= alpha
            exp_q = torch.exp(q)
            sum_exp_q = torch.sum(exp_q, dim=-1, keepdim=True)
            p = exp_q / sum_exp_q
        return p

    def v(self, q, pi=None, alpha=0):
        if alpha == 0:
            if pi is None:
                v, action = torch.max(q, dim=-1)
            else:
                v = torch.sum(pi * q, dim=-1)
        else:
            if pi is None:
                pi = self.pi(q, alpha)
            logpi = torch.log(pi + 1e-20)
            v = torch.sum(pi * (q - alpha * logpi), dim=-1)
        return v

    def action(self, q, alpha=0):
        if alpha == 0:
            a = torch.argmax(q, dim=-1)
        else:
            p = self.pi(q, alpha=alpha)
            a = torch.multinomial(p, num_samples=1, replacement=True)[:, 0]
        return a

    def act(self):
        if self.manager.episode_type > 0:
            # print('Exploit action')
            ldebug and logger.debug('Exploit action')
            # if not self.eval_mode:
            with torch.no_grad():
                # print()
                obs = torch.from_numpy(toNpFloat32(
                    self.manager.obs, True)).to(device)
                # last_16 = obs[0][-16:]
                # print(last_16)
                # print(self.q.q.linear_layers[0].weight.data[0][-16:] /
                #       torch.max(torch.abs(self.q.q.linear_layers[0].weight.data[0][-16:])))
                greedy_a = self.action(
                    self.q(obs, noisy=False)[0], alpha=0).cpu().detach().numpy()[0]
            # if self.eval_mode:
            #     self.optim.zero_grad
            #     obs = torch.from_numpy(toNpFloat32(
            #         self.manager.obs, True)).to(device)
            #     q = self.q(obs)  # [1 x n]
            #     av_val = torch.mean(q)

            #     greedy_a = self.action(q, alpha=0).cpu().detach().numpy()[0]
            return greedy_a, {}
        else:
            ldebug and logger.debug('Behavior Policy')
            if np.random.rand() < self.epsilon:
                ldebug and logger.debug('Random action')
                return self.env.action_space.sample(), {}
            else:
                ldebug and logger.debug('greedy action')
                with torch.no_grad():
                    obs = torch.from_numpy(toNpFloat32(
                        self.manager.obs, True)).to(device)
                    a = self.action(
                        self.q(obs, noisy=True)[0], alpha=self.ptemp).cpu().detach().numpy()[0]
                return a, {}

    def sgd_update(self):
        ldebug and logger.debug('Training')
        with torch.no_grad():
            states, actions, rewards, dones, info, next_states = self.exp_buffer.random_experiences_unzipped(
                self.mb_size)
            states = torch.from_numpy(toNpFloat32(states)).to(device)
            next_states = torch.from_numpy(
                toNpFloat32(next_states)).to(device)
            next_target_q = self.target_q(next_states, noisy=True)[0]
            if self.double_dqn:
                next_q = self.q(next_states, noisy=True)[0]
                next_pi = self.pi(next_q, self.ptemp)
                next_target_v = self.v(
                    next_target_q, next_pi, self.ptemp).cpu().detach().numpy()
            else:
                next_target_v = self.v(
                    next_target_q, None, self.ptemp).cpu().detach().numpy()
            desired_q = rewards + \
                (1 - dones.astype(np.int)) * \
                (self.gamma ** self.nsteps) * next_target_v

        '''-------------------begin optimization----------------'''
        self.optim.zero_grad()

        '''current q'''
        q, q_mean, q_std = self.q(states, noisy=True)
        # print(q_mean)

        '''to calculate desired q in shape of q:'''
        q_detached = np.copy(q.cpu().detach().numpy())
        all_states_idx = np.arange(self.mb_size)
        td_errors = desired_q - q_detached[all_states_idx, actions]
        if self.td_clip is not None:
            ldebug and logger.debug('Doing TD error clipping')
            td_errors = np.clip(td_errors, -self.td_clip, self.td_clip)
            # print(td_errors)
        q_detached[all_states_idx, actions] = q_detached[all_states_idx, actions] + \
            td_errors  # this is now desired_q
        desired_q = q_detached

        '''loss and gradient clip'''
        loss = F.smooth_l1_loss(q, torch.from_numpy(desired_q).to(device))
        # loss = F.mse_loss(q, torch.from_numpy(desired_q).to(device))
        loss.backward()
        if self.grad_clip is not None:
            ldebug and logger.debug('Doing grad clipping')
            torch.nn.utils.clip_grad_norm_(
                self.q.parameters(), self.grad_clip)

        self.optim.step()
        '''====================optimization done================'''

        v = self.v(q_mean, None, self.ptemp)
        self._loss, self._mb_v, self._mb_q_std = loss.cpu().detach(
        ).item(), torch.mean(v).item(), torch.mean(q_std).item()

    def post_act(self):
        if self.manager.episode_type < 2 and self.manager.num_steps > self.no_train_for_steps and (self.manager.num_steps - 1) % self.train_freq == 0:
            for sgd_step in range(self.sgd_steps):
                self.sgd_update()

        if (self.manager.num_steps - 1) % 1000 == 0:
            wandb.log({
                'DQN/Loss': self._loss,
                'DQN/Value': self._mb_v,
                'DQN/Q_Std': self._mb_q_std,
                'DQN/Epsilon': self.epsilon
            }, step=self.manager.num_steps - 1)

    def post_episode(self):
        wandb.log({
            'DQN/Loss': self._loss,
            'DQN/Value': self._mb_v,
            'DQN/Q_Std': self._mb_q_std,
            'DQN/Epsilon': self.epsilon
        }, step=self.manager.num_steps - 1)
