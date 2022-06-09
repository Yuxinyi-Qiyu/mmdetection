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
class EpochBasedRunnerSuper(EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def __init__(self,
                 widen_factor_backbone_range,
                 deepen_factor_backbone_range,
                 widen_factor_neck_range,
                 widen_factor_head_range,
                 search=False,#?
                 # search_backbone=False,
                 # search_neck=False,
                 # search_head=False,
                 sandwich=False,
                 **kwargs):
        self.widen_factor_backbone_range = widen_factor_backbone_range
        self.deepen_factor_backbone_range = deepen_factor_backbone_range
        self.widen_factor_neck_range = widen_factor_neck_range
        self.widen_factor_head_range = widen_factor_head_range
        self.search = search
        # self.search_backbone = search_backbone
        # self.search_neck = search_neck
        # self.search_head = search_head
        self.sandwich = sandwich

        self.arch = None

        super(EpochBasedRunnerSuper, self).__init__(**kwargs)

    # def run_iter(self, data_batch, train_mode, **kwargs):
    #     if self.batch_processor is not None:
    #         outputs = self.batch_processor(
    #             self.model, data_batch, train_mode=train_mode, **kwargs)
    #     elif train_mode:
    #         outputs = self.model.train_step(data_batch, self.optimizer,
    #                                         **kwargs)
    #     else:
    #         outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
    #     if not isinstance(outputs, dict):
    #         raise TypeError('"batch_processor()" or "model.train_step()"'
    #                         'and "model.val_step()" must return a dict')
    #
    #     if 'log_vars' in outputs:
    #         self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
    #     self.outputs = outputs

    def get_cand_arch(self, max_arch=False, min_arch=False):
        arch = {}
        if max_arch or min_arch: # study!
            assert not max_arch or not min_arch
            fn = max if max_arch else min
            arch['widen_factor_backbone'] = tuple([fn(self.widen_factor_backbone_range)]*5)
            arch['deepen_factor_backbone'] = tuple([fn(self.deepen_factor_backbone_range)]*4)
            arch['widen_factor_neck'] = tuple([fn(self.widen_factor_neck_range)]*8)
            arch['widen_factor_head'] = fn(self.widen_factor_head_range)
        else:
            arch['widen_factor_backbone'] = tuple(random.choices(self.widen_factor_backbone_range, k=5))
            arch['deepen_factor_backbone'] = tuple(random.choices(self.deepen_factor_backbone_range, k=4))
            arch['widen_factor_neck'] = tuple(random.choices(self.widen_factor_neck_range, k=8))
            arch['widen_factor_head'] = random.choice(self.widen_factor_head_range)
        return arch

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

            # if self.search_backbone or self.search_neck or self.search_head:
            if self.search:
                self.arch = self.get_cand_arch()

                if self.sandwich:
                    self.archs = []
                    self.archs.append(self.get_cand_arch(max_arch=True))
                    self.archs.append(self.get_cand_arch(min_arch=True))
                    self.archs.append(self.get_cand_arch())
                    self.archs.append(self.arch)
                    print("runner_arch")
                    print(self.archs)
                    # self.model.module-->YOLOX_Searchable
                    self.model.module.set_archs(self.archs, **kwargs)
                else:
                    self.model.module.set_arch(self.arch, **kwargs)

            if i % 50 == 0:
                self.logger.info(f'arch: {self.archs}')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1


