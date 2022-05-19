# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import init_random_seed, set_random_seed, train_detector

from utils import get_broadcast_cand, dict_to_tuple, tuple_to_dict, get_test_data, check_cand
from trainer import parse_args,get_train_data, train_model, get_model, get_cfg
from tester import get_cand_map, get_cand_map_new, forward_model
import time
import logging
import numpy as np
from random import choice
import functools
import random

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()



class EvolutionSearcher1(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs  # 20
        # self.max_epochs = 5 # 20
        self.select_num = args.select_num  # 10
        # self.select_num = 2 # 10
        # self.population_num = args.population_num # 50
        self.population_num = 4
        self.m_prob = args.m_prob  # 0.1
        self.crossover_num = args.crossover_num  # 25
        # self.crossover_num = 2
        self.mutation_num = args.mutation_num  # 25
        # self.mutation_num = 2
        self.flops_limit = args.flops_limit  # None (float) # 17.651 M 122.988 GFLOPS
        self.params_limit = args.params_limit
        self.input_shape = (3,) + tuple(args.shape)  # default=[1280, 800]  [3,1280,800] todo?
        self.cfg, self.meta = get_cfg(self.args)  # 获取cfg文件所有内容到meta上
        self.cfg = self.cfg.copy()

        self.model, self.distributed = get_model(self.cfg, self.args)
        print("after get model")

        # dataset
        self.train_dataset = build_dataset(self.cfg.data.train_val)
        self.train_data_loader = build_dataloader(
            self.train_dataset,
            samples_per_gpu=self.cfg.data.samples_per_gpu,  # !
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=self.distributed,
            shuffle=False)
        self.test_dataset = build_dataset(self.cfg.data.test)
        self.test_data_loader = build_dataloader(
            self.test_dataset,
            samples_per_gpu=self.cfg.data.samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=self.distributed,
            shuffle=False)
        self.log_dir = '.workdir/summary'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.checkpoint_name = os.path.join(self.log_dir, 'ea_'
                                                          'checkpoint.pth.tar')
        loaded_checkpoint = os.path.split(args.checkpoint)[1].split('.')[0]
        times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # default 1600*2*4
        print(loaded_checkpoint, self.log_dir)
        self.logfile = os.path.join(self.log_dir, '{}_ea_dis{}_{}.log'.format(loaded_checkpoint, self.distributed,
                                                                                 times))

    def get_param(self, cand): # todo
        arch = cand
        cfg = Config.fromfile('configs/yolox/yolox_s_8x8_300e_voc_tfs.py')

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # cfg.model.backbone.deepen_factor = arch['deepen_factor']
        cfg.model.backbone.widen_factor = arch['widen_factor_backbone']

        in_channels = [256, 512, 1024]
        for i in range(len(in_channels)):
            in_channels[i] = int(in_channels[i] * arch['widen_factor_backbone'][i + 2])
        cfg.model.neck.in_channels = in_channels

        cfg.model.neck.widen_factor = arch['widen_factor_neck']
        cfg.model.neck.widen_factor_out = arch['widen_factor_neck_out']
        cfg.model.bbox_head.widen_factor_neck = arch['widen_factor_neck_out']

        model = build_detector(
            cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        if torch.cuda.is_available():
            model.cuda()

        model.eval()

        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                    format(model.__class__.__name__))
        # flops_, params_ = get_model_complexity_info(model, self.input_shape)
        flops, params = get_model_complexity_info(model, self.input_shape, as_strings=False, print_per_layer_stat=False) # todo：如何在这里根据深度变化改变获取复杂度的结果？
        flops = round(flops / 10. ** 9, 2)
        params = round(params / 10 ** 6, 2)
        print("flops,params")
        print(flops,params)

        torch.cuda.empty_cache()
        del model
        return params, flops, cfg

    def is_legal(self, arch, **kwargs):
        size, fp, cfg = self.get_param(arch)

        rank, world_size = get_dist_info() # 0,1
        cand = dict_to_tuple(arch) # (0.625, 0.5, 0.375, 0.125, 0.125, 0.33, 0.33, 1, 1)

        self.model.set_arch(arch, **kwargs) # 这里set_ARCH,修改了模型宽度

        print("after model set arch")

        map = get_cand_map_new(self.model,
                           self.args,
                           self.distributed,
                           self.cfg,
                           self.train_data_loader,
                           self.train_dataset,
                           self.test_data_loader,
                           self.test_dataset)
        # 获得当前模型的map
        if not isinstance(map, tuple):
            if self.args.eval[0] == "bbox":
                map = tuple([0.] * 6)
            else:
                map = tuple([0.])
        if not map:
            map = tuple([0.] * 6)
        map = get_broadcast_cand(map, self.distributed, rank)
        torch.cuda.empty_cache()

        if map:
            return map

        return False


    def search(self):
        rank, _ = get_dist_info()
        if rank == 0:
            print("rank==0")
            logging.basicConfig(filename=self.logfile, level=logging.INFO)
            print(self.logfile)
            logging.info(self.cfg)

        # cand = {'widen_factor_backbone': (0.25, 0.25, 0.25, 0.125, 0.5),
        #         'deepen_factor': (0.33, 0.33, 0.33, 0.33),
        #         'widen_factor_neck':(0.125, 0.375, 0.5, 0.375, 0.25, 0.375, 0.5, 0.25),
        #         'widen_factor_neck_out':0.25}
        # cand = {'widen_factor_backbone': (0.375, 0.5, 0.125, 0.5, 0.25),
        #         'deepen_factor': (0.33, 0.33, 0.33, 0.33),
        #         'widen_factor_neck':(0.5, 0.125, 0.125, 0.375, 0.375, 0.25, 0.25, 0.25),
        #         'widen_factor_neck_out':0.375}
        cand = {'widen_factor_backbone': (0.375, 0.125, 0.125, 0.5, 0.5),
                'deepen_factor': (0.33, 0.33, 0.33, 0.33),
                'widen_factor_neck':(0.125, 0.5, 0.25, 0.5, 0.375, 0.125, 0.25, 0.5),
                'widen_factor_neck_out':0.5}
        map = self.is_legal(cand)
        print("cand:" + str(cand))
        print("map:" + str(map))




if __name__ == '__main__':
    args = parse_args()
    searcher = EvolutionSearcher1(args)
    print("fin")
    searcher.search()

    # python tools/search/search.py 运行
