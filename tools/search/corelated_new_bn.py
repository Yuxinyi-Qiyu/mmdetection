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
from trainer import parse_args, get_train_data, train_model, get_model, get_cfg
from tester import get_cand_map, get_cand_map_new, forward_model
import time
import logging
import numpy as np
from random import choice
import functools
import random
import copy

from torch import nn
from thop.vision.basic_hooks import count_parameters
from thop.profile import prRed, register_hooks
from mmdet.models import build_detector
from mmdet.models.utils.usconv import USConv2d, USBatchNorm2d, \
    count_usconvNd_flops, count_usconvNd_params, count_usbn_flops, count_usbn_params

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

def profile(model: nn.Module, inputs, custom_ops=None, verbose=True):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))

        m_type = type(m)

        fn = None
        params_fn = count_parameters
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if len(fn) == 2:
                fn, params_fn = fn
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and verbose:
                prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if fn is not None:
            handler_collection[m] = (m.register_forward_hook(fn), m.register_forward_hook(params_fn))
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params = 0, 0
        for m in module.children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
            total_params += m_params
        #  print(prefix, module._get_name(), (total_ops.item(), total_params.item()))
        return total_ops, total_params

    total_ops, total_params = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return total_ops, total_params

def my_get_model_complexity_info(model, input_shape, custom_ops=None):
    assert type(input_shape) is tuple
    assert len(input_shape) >= 1
    assert isinstance(model, nn.Module)
    try:
        batch = torch.ones(()).new_empty(
            (1, *input_shape),
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device)
    except StopIteration:
        # Avoid StopIteration for models which have no parameters,
        # like `nn.Relu()`, `nn.AvgPool2d`, etc.
        batch = torch.ones(()).new_empty((1, *input_shape))
    flops_count, params_count = profile(model, (batch,), verbose=False, custom_ops=custom_ops)
    return flops_count, params_count

class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs # 20
        # self.max_epochs = 5 # 20
        self.select_num = args.select_num # 10
        # self.select_num = 2 # 10
        self.population_num = args.population_num # 50
        # self.population_num = 4
        self.m_prob = args.m_prob # 0.1
        self.crossover_num = args.crossover_num # 25
        # self.crossover_num = 2
        self.mutation_num = args.mutation_num # 25
        # self.mutation_num = 2
        self.map_limit = 0.738 # "tiny_map"
        self.flops_limit = args.flops_limit # None (float) # 17.651 M 122.988 GFLOPS
        self.params_limit = args.params_limit
        self.input_shape = (3,) + tuple(args.shape) # default=[1280, 800]  [3,1280,800] todo?
        self.cfg, self.meta = get_cfg(self.args) # 获取cfg文件所有内容到meta上
        self.cfg = self.cfg.copy()

        self.model, self.distributed = get_model(self.cfg, self.args)

        # dataset
        self.train_dataset = build_dataset(self.cfg.data.train_val)
        self.train_data_loader = build_dataloader(
            self.train_dataset,
            samples_per_gpu=self.cfg.data.samples_per_gpu, # !
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

        self.widen_factor_range = self.cfg.get('widen_factor_range', None)
        self.deepen_factor_range = self.cfg.get('deepen_factor_range', None)
        # self.search_backbone = self.cfg.model.get('search_backbone', None)
        # self.search_neck = self.cfg.model.get('search_neck', None)
        # self.search_head = self.cfg.model.get('search_head', None)

        self.log_dir = './summary'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.checkpoint_name = os.path.join(self.log_dir, 'ea_'
                                                          'checkpoint.pth.tar')
        loaded_checkpoint = os.path.split(args.checkpoint)[1].split('.')[0]
        times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # default 1600*2*4
        self.logfile = os.path.join(self.log_dir, '{}_ea_dis{}_map_{}_{}.log'.format(loaded_checkpoint, self.distributed, self.map_limit, times))

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

    def idx_to_arch(self, cand): # cand: idx->factor
        widen_factor_backbone = []
        for i in range(len(cand['widen_factor_backbone_idx'])):  # !!!here 修改cand传递形式，可能是因为broad cast无法传递0。33
            widen_factor_backbone.append(self.widen_factor_range[cand['widen_factor_backbone_idx'][i]])
        deepen_factor = []
        for i in range(len(cand['deepen_factor_idx'])):
            deepen_factor.append(self.deepen_factor_range[cand['deepen_factor_idx'][i]])
        widen_factor_neck = []
        for i in range(len(cand['widen_factor_neck_idx'])):
            widen_factor_neck.append(self.widen_factor_range[cand['widen_factor_neck_idx'][i]])
        widen_factor_neck_out = self.widen_factor_range[cand['widen_factor_neck_out_idx']]

        arch = {
            'widen_factor_backbone': tuple(widen_factor_backbone),
            'deepen_factor': tuple(deepen_factor),
            'widen_factor_neck': tuple(widen_factor_neck),
            'widen_factor_neck_out': widen_factor_neck_out,
        }
        return arch

    def get_param(self, cand):

        arch = self.idx_to_arch(cand)
        print(arch)
        # cfg = Config.fromfile('configs/yolox/yolox_s_8x8_300e_voc_searchable.py')
        cfg = Config.fromfile('configs/yolox/yolox_s_8x8_300e_voc_tfs.py')

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # cfg.model.backbone.deepen_factor = arch['deepen_factor']
        cfg.model.backbone.widen_factor = arch['widen_factor_backbone']

        in_channels = [256, 512, 1024]
        for i in range(len(in_channels)):
            in_channels[i] = int(in_channels[i] * arch['widen_factor_backbone'][i + 2])
        cfg.model.neck.in_channels = in_channels
        print(in_channels)

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
        flops, params = my_get_model_complexity_info(model, self.input_shape,
                                                     # as_strings=False,
                                                     # print_per_layer_stat=False
                                                     )
        # Warning: variables __flops__ or __params__ are already defined for the moduleMaxPool2d ptflops can affect your code!
        # todo: params可变，但是flops不变，FLOPs与卷积核通道有无关系？
        flops = round(flops / 10. ** 9, 2)
        params = round(params / 10 ** 6, 2)

        torch.cuda.empty_cache()
        del model
        return params, flops, cfg

    def is_legal(self, arch, **kwargs):
        rank, world_size = get_dist_info()
        cand = dict_to_tuple(arch)
        with open(self.log_dir + '_gpu_{}.txt'.format(rank), 'a') as f:
            f.write(str(cand) + '\n')

        if cand not in self.vis_dict: # 记录每种cand的map、model size、flops
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        size, fp, cfg = self.get_param(arch)  # 获得子网的参数量和flops
        print(size, fp)
        rank, _ = get_dist_info()

        size = get_broadcast_cand(size, self.distributed, rank)
        fp = get_broadcast_cand(fp, self.distributed, rank)

        info['fp'] = fp
        info['size'] = size
        info['cfg'] = cfg
        del size, fp, cfg
        self.model.set_arch(self.idx_to_arch(arch)) # 这里set_ARCH,修改了模型参数

        # 获得当前模型的map
        map = get_cand_map_new(self.model,
                           self.args,
                           self.distributed,
                           self.cfg,
                           self.train_data_loader,
                           self.train_dataset,
                           self.test_data_loader,
                           self.test_dataset) #  # (0.6599323749542236,)

        # map = []
        # map.append(round(random.uniform(0, 1),2))
        # map = tuple(map)

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
            info['map'] = map[0]
            info['map_list'] = map
            info['visited'] = True # 标记
            del map
            return True

        return False

    def get_random(self, num): # num=population_num
        rank, _ = get_dist_info()
        if rank == 0:
            print('random select ........')
        # 生成子网结构
        print('test_tiny')
        cand = {
            'widen_factor_backbone_idx': tuple([2, 2, 2, 2, 2]),
            'deepen_factor_idx': tuple([0, 0, 0, 0]),
            'widen_factor_neck_idx': tuple([2, 2, 2, 2, 2, 2, 2, 2]),
            'widen_factor_neck_out_idx': 2,
        }
        rank, world_size = get_dist_info() # 0, 1
        cand = get_broadcast_cand(cand, self.distributed, rank)
        cand_tuple = dict_to_tuple(cand) # cand是dict，转换成数组
        # (0, 3, 2, 0, 0, 0, 0, 0, 0)
        self.is_legal(cand) # 是否符合约束，不符合就重新生成子网架构

    def search(self):
        rank, _ = get_dist_info()
        if rank == 0:
            print(
                'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                    self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                    self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.get_random(self.population_num)


if __name__ == '__main__':
    args = parse_args()
    searcher = EvolutionSearcher(args)
    print("!!!!!!!!!!!!!!")
    searcher.search()
