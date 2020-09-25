import os
from time import sleep

import torch
from six import MAXSIZE
from torch import nn

import RL


class ModelLoadAgent(RL.Agent):
    def __init__(self, name, algo, model: nn.Module, checkpoints_folder, in_sequence=False, wait_for_new=False):
        '''The fn should take as args'''
        super().__init__(name, algo, False)
        self.in_sequence = in_sequence
        assert wait_for_new, "Non wait not supported yet"
        self.wait_for_new = wait_for_new
        self.checkpoints_folder = checkpoints_folder
        self.model = model
        self.consumed_paths = set()
        self.model_step_id = None
        self.model_episode_id = None

    def start(self):
        self.consumed_paths.clear()
        self.update_model()

    def next_model_stepId_epId(self):
        all = os.listdir(self.checkpoints_folder)
        # print(self.consumed_paths)
        new_files = set(all) - self.consumed_paths
        # print(new_files)
        least_step_id = MAXSIZE
        least_ep_id = MAXSIZE
        next_chkpt_model = None
        for f in new_files:
            if f.endswith('.model') and f.startswith('step'):
                step_id = int(f[f.find('-') + 1: f.find('-ep')])
                ep_id = int(f[f.find('+') + 1:f.find('.')])
                if step_id < least_step_id:
                    least_step_id = step_id
                    least_ep_id = ep_id
                    next_chkpt_model = f
        if next_chkpt_model is not None:
            # print(next_chkpt_model)
            self.consumed_paths.add(next_chkpt_model)
            next_chkpt_model = os.path.join(
                self.checkpoints_folder, next_chkpt_model)
        return next_chkpt_model, least_step_id, least_ep_id

    def latest_model_stepId_epId(self):
        all = os.listdir(self.checkpoints_folder)
        new_files = set(all) - self.consumed_paths
        max_step_id = -1
        max_ep_id = -1
        latest_model = None
        if len(new_files) > 0:
            path = os.path.join(self.checkpoints_folder, 'latest.model')
            if not os.path.exists(path):
                latest_model = None
            else:
                latest_model = path
                for f in new_files:
                    if f.endswith('.model') and f.startswith('step'):
                        step_id = int(f[f.find('-') + 1: f.find('-ep')])
                        ep_id = int(f[f.find('+') + 1:f.find('.')])
                        if step_id > max_step_id:
                            max_step_id = step_id
                            max_ep_id = ep_id
                self.consumed_paths.add(all)

        return latest_model, max_step_id, max_ep_id

    def update_model(self):
        while True:
            if self.in_sequence:
                path, step_id, ep_id = self.next_model_stepId_epId()
            else:
                path, step_id, ep_id = self.latest_model_stepId_epId()

            if path is not None:
                print('Loadng path', path)
                self.model.load_state_dict(torch.load(path))
                self.model_step_id = step_id
                self.model_episode_id = ep_id
                break
            else:
                if self.wait_for_new:
                    # print('Waiting for new')
                    sleep(1)
                else:
                    break

    def pre_episode(self):
        self.manager.step_id = self.model_step_id
        self.manager.episode_id = self.model_episode_id

    def pre_act(self):
        self.manager.step_id = self.model_step_id
        self.manager.episode_id = self.model_episode_id

    def post_act(self):
        self.manager.step_id = self.model_step_id
        self.manager.episode_id = self.model_episode_id

    def post_episode(self):
        self.manager.step_id = self.model_step_id
        self.manager.episode_id = self.model_episode_id

        self.update_model()

    def pre_close(self):
        self.manager.step_id = self.model_step_id
        self.manager.episode_id = self.model_episode_id
