import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import RL
from RL.agents.exp_buff_agent import ExperienceBuffer
from RL.agents.stats_recording_agent import StatsRecordingAgent
from RL.utils.standard_models import FFModel
from RL.utils.util_fns import toNpFloat32

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DQNModel(FFModel):
    def __init__(self, input_shape, convs, hidden_layers, n_actions):
        super().__init__(input_shape, convs, list(hidden_layers) + [n_actions])


class DQNCoreAgent(RL.Agent):
    def __init__(self, name, algo, convs, hidden_layers, train_freq, mb_size, double_dqn, gamma, nsteps, td_clip, grad_clip, lr, epsilon, should_exploit_fn, eval_mode, no_train_for_steps, exp_buffer: ExperienceBuffer, death_cost=1):
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

        logger.info(f'Creating main network on device {device}')
        self.q = DQNModel(self.env.observation_space.shape,
                          convs, hidden_layers, self.env.action_space.n)
        self.q.to(device)
        if not eval_mode:
            logger.info(f'Creating target network on device {device}')
            self.target_q = DQNModel(
                self.env.observation_space.shape, convs, hidden_layers, self.env.action_space.n)
            self.target_q.to(device)
            logger.info(f'Creating optimizer (lr={lr})')
            self.optim = optim.Adam(self.q.parameters(), lr=self.lr)

    def act(self):
        with torch.no_grad():
            obs = torch.from_numpy(toNpFloat32(
                self.manager.obs, True)).to(device)
            q = self.q(obs).cpu().detach().numpy()[0]
        logger.debug(f'q_values: {q}')
        logger.debug(f'highest_q: {np.max(q)}')
        greedy_a = np.argmax(q)

        if self.should_exploit_fn():
            logger.debug('Exploit mode action')
            return greedy_a
        else:
            if np.random.rand() < self.epsilon:
                logger.debug('Random action')
                return self.env.action_space.sample()
            else:
                logger.debug('Greedy action')
                return greedy_a

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
                all_rows = np.arange(self.mb_size)
                next_target_q = self.target_q(
                    next_states).cpu().detach().numpy()
                if self.double_dqn:
                    next_actions = np.argmax(
                        self.q(next_states).cpu().detach().numpy(), axis=1)
                    next_target_v = next_target_q[all_rows, next_actions]
                else:
                    next_target_v = np.max(next_target_q, axis=1)
                desired_v = rewards + \
                    (1 - dones.astype(np.int)) * \
                    (self.gamma ** self.nsteps) * next_target_v - \
                    dones.astype(np.int) * (self.death_cost)  # penalize death
                q = self.q(states).cpu().detach().numpy()
                td_errors = desired_v - q[all_rows, actions]
                if self.td_clip is not None:
                    logger.debug('Doing TD error clipping')
                    td_errors = np.clip(td_errors, -self.td_clip, self.td_clip)
                q[all_rows, actions] = q[all_rows, actions] + \
                    td_errors  # this is now desired_q

            self.optim.zero_grad()
            loss = self.loss(states, torch.from_numpy(q).to(device))
            loss.backward()
            if self.grad_clip is not None:
                logger.debug('Doing grad clipping')
                for p in self.q.parameters():
                    p.grad.data.clamp_(-self.grad_clip, self.grad_clip)
            self.optim.step()
            loss = loss.cpu().detach().item()

            logger.debug(f'Stepped. Loss={loss}')
            recorder = self.algo.get_agent_by_type(StatsRecordingAgent)
            recorder.record_kvstat('loss', loss)
            recorder.record_kvstat('mb_v', np.mean(q[all_rows, actions]))
