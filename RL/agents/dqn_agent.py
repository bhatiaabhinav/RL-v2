import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import RL
from RL.agents.exp_buff_agent import ExperienceBuffer
from RL.utils.standard_models import FFModel
from RL.utils.util_fns import toNpFloat32

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DQNModel(FFModel):
    def __init__(self, input_shape, convs, hidden_layers, n_actions):
        super().__init__(input_shape, convs, list(hidden_layers) + [n_actions])


class DQNCoreAgent(RL.Agent):
    def __init__(self, name, algo, convs, hidden_layers, train_freq, mb_size, double_dqn, gamma, nsteps, td_clip, grad_clip, lr, epsilon, should_exploit_fn, eval_mode, no_train_for_steps, exp_buffer: ExperienceBuffer, policy_temperature=0, death_cost=0):
        super().__init__(name, algo, supports_multiple_envs=False)
        self.convs = convs
        self.hidden_layers = hidden_layers
        self.train_freq = train_freq
        self.mb_size = mb_size
        self.double_dqn = double_dqn
        self.gamma = gamma
        self.nsteps = nsteps
        self.td_clip = td_clip
        self.grad_clip = grad_clip
        self.lr = lr
        self.no_train_for_steps = no_train_for_steps
        self.exp_buffer = exp_buffer
        self.epsilon = epsilon
        self.should_exploit_fn = should_exploit_fn
        self.death_cost = death_cost
        self.ptemp = policy_temperature

        logger.info(f'Creating main network on device {device}')
        self.q = DQNModel(self.env.observation_space.shape,
                          convs, hidden_layers, self.env.action_space.n)
        self.q.to(device)
        logger.info(str(self.q))
        if not eval_mode:
            logger.info(f'Creating target network on device {device}')
            self.target_q = DQNModel(
                self.env.observation_space.shape, convs, hidden_layers, self.env.action_space.n)
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
            v, a = torch.max(q, dim=-1)
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
        if self.should_exploit_fn():
            logger.debug('Exploit mode action')
            with torch.no_grad():
                obs = torch.from_numpy(toNpFloat32(
                    self.manager.obs, True)).to(device)
                greedy_a = self.action(
                    self.q(obs), alpha=0).cpu().detach().numpy()[0]
            return greedy_a
        else:
            if np.random.rand() < self.epsilon:
                logger.debug('Random action')
                return self.env.action_space.sample()
            else:
                logger.debug('Soft greedy action')
                with torch.no_grad():
                    obs = torch.from_numpy(toNpFloat32(
                        self.manager.obs, True)).to(device)
                    a = self.action(
                        self.q(obs), alpha=self.ptemp).cpu().detach().numpy()[0]
                return a

    def loss(self, states, desired_q):
        return F.smooth_l1_loss(self.q(states), desired_q)

    def post_act(self):
        if self.manager.num_steps > self.no_train_for_steps and (self.manager.num_steps - 1) % self.train_freq == 0:
            logger.debug('Training')
            with torch.no_grad():
                states, actions, rewards, dones, info, next_states = self.exp_buffer.random_experiences_unzipped(
                    self.mb_size)
                states = torch.from_numpy(toNpFloat32(states)).to(device)
                next_states = torch.from_numpy(
                    toNpFloat32(next_states)).to(device)
                next_target_q = self.target_q(next_states)
                if self.double_dqn:
                    next_q = self.q(next_states)
                    next_pi = self.pi(next_q, self.ptemp)
                    next_target_v = self.v(
                        next_target_q, next_pi, self.ptemp).cpu().detach().numpy()
                else:
                    next_target_v = self.v(
                        next_target_q, None, self.ptemp).cpu().detach().numpy()
                desired_v = rewards + (1 - dones.astype(np.int)) * (self.gamma ** self.nsteps) * \
                    next_target_v - \
                    dones.astype(np.int) * (self.death_cost)  # penalize death
                q = self.q(states)
                v = self.v(q, None, self.ptemp).cpu().detach().numpy()
                q = q.cpu().detach().numpy()
                all_states_idx = np.arange(self.mb_size)
                td_errors = desired_v - q[all_states_idx, actions]
                if self.td_clip is not None:
                    logger.debug('Doing TD error clipping')
                    td_errors = np.clip(td_errors, -self.td_clip, self.td_clip)
                q[all_states_idx, actions] = q[all_states_idx, actions] + \
                    td_errors  # this is now desired_q

            self.optim.zero_grad()
            loss = self.loss(states, torch.from_numpy(q).to(device))
            loss.backward()
            if self.grad_clip is not None:
                logger.debug('Doing grad clipping')
                torch.nn.utils.clip_grad_norm_(
                    self.q.parameters(), self.grad_clip)
            self.optim.step()
            loss = loss.cpu().detach().item()

            logger.debug(f'Stepped. Loss={loss}')
            RL.stats.record_kvstats(loss=loss, mb_v=np.mean(v))

    def post_episode(self):
        loss = RL.stats.kvstats.get('loss', 0)
        mb_v = RL.stats.kvstats.get('mb_v', 0)
        RL.stats.record_stats(loss=loss, mb_v=mb_v)
