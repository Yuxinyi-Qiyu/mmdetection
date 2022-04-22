# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from mmcv.runner import EpochBasedRunner, save_checkpoint, get_host_info, RUNNERS
import numpy as np
from random import choice
import random

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)


@RUNNERS.register_module()
class EpochBasedRunner_tfs(EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def __init__(self,
                 panas_type=0,
                 panas_c_range=[128, 256],
                 panas_d_range=[3, 12],
                 head_d_range=[1, 3],
                 widen_factor_range=[0, 1],
                 deepen_factor_range=[0, 1],
                 cb_type=3,
                 cb_step=4,
                 c_interval=16,
                 w_interval=16,  # @todo
                 base_channel=64,
                 search_backbone=True,
                 search_neck=True,
                 search_head=False,
                 sandwich=False,
                 **kwargs):
        self.panas_type = panas_type
        self.step = 0
        self.panas_c_range = panas_c_range
        self.panas_d_range = panas_d_range
        self.head_d_range = head_d_range
        self.widen_factor_range = widen_factor_range,
        self.deepen_factor_range = deepen_factor_range,
        self.cb_type = cb_type
        self.cb_step = cb_step
        self.c_interval = c_interval
        self.base_channel = base_channel
        self.w_interval = w_interval
        self.sandwich = sandwich

        self.search_backbone = search_backbone
        self.search_neck = search_neck
        self.search_head = search_head

        self.arch = None

        super(EpochBasedRunner_tfs, self).__init__(**kwargs)

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:

            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')

        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def set_grad_none(self, **kwargs):
        self.model.module.set_grad_none(**kwargs)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1


