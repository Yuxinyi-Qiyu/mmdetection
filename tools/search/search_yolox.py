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
        self.select_num = args.select_num # 10
        self.population_num = args.population_num # 50
        self.m_prob = args.m_prob # 0.1
        self.crossover_num = args.crossover_num # 25
        self.mutation_num = args.mutation_num # 25
        self.flops_limit = args.flops_limit # None (float) # 17.651 M 122.988 GFLOPS
        self.params_limit = args.params_limit
        self.input_shape = (3,) + tuple(args.shape) # default=[1280, 800]  [3,1280,800] todo?
        self.cfg, self.meta = get_cfg(self.args) # ??????cfg?????????????????????meta???
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
        self.search_channel = self.cfg.get('search_channel', None)
        # self.random_func = self.cfg.get('random_func', None)
        self.search_backbone = self.cfg.model.get('search_backbone', None)
        self.search_neck = self.cfg.model.get('search_neck', None)
        self.search_head = self.cfg.model.get('search_head', None)

        self.log_dir = './summary'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.checkpoint_name = os.path.join(self.log_dir, 'ea_'
                                                          'checkpoint.pth.tar')
        loaded_checkpoint = os.path.split(args.checkpoint)[1].split('.')[0]
        times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # default 1600*2*4
        self.logfile = os.path.join(self.log_dir, '{}_ea_dis{}_fp_{}_{}.log'.format(loaded_checkpoint, self.distributed, self.flops_limit, times))

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

    def save_checkpoint(self):
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        print('save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_name)
        return True

    def idx_to_arch(self, cand): # cand: idx->factor
        widen_factor_backbone = []
        deepen_factor = []
        widen_factor_neck = []
        widen_factor_neck_out = None
        if self.search_backbone and self.search_channel:
            for i in range(len(cand['widen_factor_backbone_idx'])):  # !!!here ??????cand??????????????????????????????broad cast????????????0???33
                widen_factor_backbone.append(self.widen_factor_range[cand['widen_factor_backbone_idx'][i]])
        # if self.search_backbone and self.search_depth: # todo
        for i in range(len(cand['deepen_factor_idx'])):
            deepen_factor.append(self.deepen_factor_range[cand['deepen_factor_idx'][i]])
        if self.search_neck and self.search_channel:
            for i in range(len(cand['widen_factor_neck_idx'])):
                widen_factor_neck.append(self.widen_factor_range[cand['widen_factor_neck_idx'][i]])
            widen_factor_neck_out = self.widen_factor_range[cand['widen_factor_neck_out_idx']]

        arch = {}
        if widen_factor_backbone != []:
            arch['widen_factor_backbone'] = tuple(widen_factor_backbone)
        if deepen_factor != []:
            arch['deepen_factor'] = tuple(deepen_factor)
        if widen_factor_neck != []:
            arch['widen_factor_neck'] = tuple(widen_factor_neck)
        if widen_factor_neck_out != None:
            arch['widen_factor_neck_out'] = widen_factor_neck_out

        return arch

    def get_param(self, cand):

        arch = self.idx_to_arch(cand)
        cfg = Config.fromfile('configs/yolox/yolox_s_8x8_300e_voc_tfs.py')

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        if 'widen_factor_backbone' in arch:
            # cfg.model.backbone.deepen_factor = arch['deepen_factor']
            cfg.model.backbone.widen_factor = arch['widen_factor_backbone']

            in_channels = [256, 512, 1024]
            for i in range(len(in_channels)):
                in_channels[i] = int(in_channels[i] * arch['widen_factor_backbone'][i + 2])
            cfg.model.neck.in_channels = in_channels

        if 'widen_factor_neck' in arch:
            cfg.model.neck.widen_factor = arch['widen_factor_neck']
        if 'widen_factor_neck_out' in arch:
            cfg.model.neck.widen_factor_out = arch['widen_factor_neck_out']
        if 'widen_factor_neck_out' in arch:
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
        # todo: params???????????????flops?????????FLOPs?????????????????????????????????
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

        if cand not in self.vis_dict: # ????????????cand???map???model size???flops
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        size, fp, cfg = self.get_param(arch)  # ???????????????????????????flops

        rank, _ = get_dist_info()
        if rank == 0:
            print(size, fp) # 46.01 50.05 Config # 8.65 49.73

        size = get_broadcast_cand(size, self.distributed, rank)
        fp = get_broadcast_cand(fp, self.distributed, rank)

        if (self.flops_limit and fp > self.flops_limit)\
                or (self.params_limit and size > self.params_limit): # ??????????????????
            del size, fp, cfg
            return False

        info['fp'] = fp
        info['size'] = size
        info['cfg'] = cfg
        del size, fp, cfg
        print(arch)
        print(self.idx_to_arch(arch))
        self.model.set_arch(self.idx_to_arch(arch)) # ??????set_ARCH,?????????????????????

        # ?????????????????????map
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
            info['visited'] = True # ??????
            del map
            if rank == 0:
                print("self.vis_dict")
                print(self.vis_dict)
            return True

        return False

    def update_top_k(self, candidates, *, k, key, reverse=True):
        # ??????key??????k=select_num?????????
        # ???candidates??????????????????map????????????????????????k???
        # self.keep_top_k????????????????????????candidates?????????????????????????????????self.keep_top_k???????????????????????????
        assert k in self.keep_top_k
        rank, _ = get_dist_info()
        if rank == 0:
            print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            # cands = ??????????????????
            for cand in cands:
                if isinstance(cand, dict):
                    cand = dict_to_tuple(cand)
                if cand not in self.vis_dict: # vis_dict ???????????????info ??????
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            # self.vis_dict: {"(0.125,0.5,0.5,0.5,0.5,0.33,...)":{},"":{},...}
            for cand in cands:
                yield cand

    def get_cand_iter(self):
        arch = {}
        if self.search_backbone and self.search_channel:
            arch['widen_factor_backbone_idx'] = tuple([np.random.randint(0, len(self.widen_factor_range)) for i in range(5)])
        # if self.search_backbone and self.search_depth: # todo
        arch['deepen_factor_idx'] = tuple([np.random.randint(0, len(self.deepen_factor_range)) for i in range(4)])
        if self.search_neck and self.search_channel:
            arch['widen_factor_neck_idx'] = tuple([np.random.randint(0, len(self.widen_factor_range)) for i in range(8)])
            arch['widen_factor_neck_out_idx'] = np.random.randint(0, len(self.widen_factor_range))
        return self.stack_random_cand(lambda: arch)

    def get_random(self, num): # num=population_num
        rank, _ = get_dist_info()
        if rank == 0:
            print('random select ........')
        # ??????????????????
        cand_iter = self.get_cand_iter()
        while len(self.candidates) < num: # ??????????????????population num
            cand = next(cand_iter) # ???????????????arch
            print(cand)
            # {'widen_factor_idx': (0, 3, 2, 0, 0), 'deepen_factor_idx': (0, 0, 0, 0)}
            rank, world_size = get_dist_info() # 0, 1
            cand = get_broadcast_cand(cand, self.distributed, rank)
            cand_tuple = dict_to_tuple(cand) # cand???dict??????????????????
            if not self.is_legal(cand): # ?????????????????????????????????????????????????????????
                continue

            self.candidates.append(cand_tuple) # cand_tuple??????[]
            if rank == 0:
                print('random {}/{}'.format(len(self.candidates), num))
                logging.info(
                    'random {}/{}, arch: {}, AP {}, {} M, {} GFLOPS'.format(
                        len(self.candidates), num, cand,
                        self.vis_dict[cand_tuple]['map_list'],
                        self.vis_dict[cand_tuple]['size'],
                        self.vis_dict[cand_tuple]['fp']
                    ))

        if rank == 0:
            print('random_num = {}'.format(len(self.candidates)))
            print(self.candidates)

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        rank, _ = get_dist_info()
        if rank == 0:
            print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10 # 20

        def random_func(): # ?????????prob??????????????????stage???channel???depth
            cand = list(choice(self.keep_top_k[k]))
            idx = 0
            if self.search_backbone and self.search_channel:
                for i in range(5): # ????????????????????????????????? # todo ?????????
                    if np.random.random_sample() < m_prob:
                        cand[idx + i] = np.random.randint(0, len(self.widen_factor_range))
                idx += 5
            # if self.search_backbone and self.search_depth:
            for i in range(4):
                if np.random.random_sample() < m_prob:
                    cand[idx + i] = np.random.randint(0, len(self.deepen_factor_range))
            idx += 4
            if self.search_neck and self.search_channel:
                for i in range(8):
                    if np.random.random_sample() < m_prob:
                        cand[idx + i] = np.random.randint(0, len(self.widen_factor_range))
                idx += 8
                if np.random.random_sample() < m_prob:
                    cand[idx] = np.random.randint(0, len(self.widen_factor_range))
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func) # ??????mutation???????????????????????????
        while len(res) < mutation_num and max_iters > 0:
            # res????????????mutation num????????????????????????????????????
            max_iters -= 1
            cand_tuple = next(cand_iter)
            print(cand_tuple)
            rank, world_size = get_dist_info()
            cand_tuple = get_broadcast_cand(cand_tuple, self.distributed, rank)
            cand = tuple_to_dict(self, cand_tuple)
            cand = get_broadcast_cand(cand, self.distributed, rank)
            if not self.is_legal(cand):
                continue
            res.append(cand_tuple)
            print('mutation {}/{}'.format(len(res), mutation_num))
            logging.info(
                'mutation {}/{}, arch: {}, AP {}, {} M, {} GFLOPS'.format(len(res), mutation_num, cand,
                                                                              self.vis_dict[cand_tuple][
                                                                                  'map_list'],
                                                                              self.vis_dict[cand_tuple]['size'],
                                                                              self.vis_dict[cand_tuple]['fp']))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num): # ??????
        # ???candidate??????p1???p2
        assert k in self.keep_top_k
        print('crossover ......')
        res = [] # ???????????????????????????
        iter = 0
        max_iters = 10 * crossover_num # 20

        def random_func():
            # print(self.keep_top_k)
            # {2: [(2, 1, 1, 3, 1, 0, 0, 0, 0), (0, 3, 2, 0, 0, 0, 0, 0, 0)],
            # 50: [(2, 1, 1, 3, 1, 0, 0, 0, 0), (0, 3, 2, 0, 0, 0, 0, 0, 0),
            # (2, 2, 1, 3, 3, 0, 0, 0, 0), (3, 1, 3, 3, 1, 0, 0, 0, 0)]}
            p1 = choice(self.keep_top_k[k]) # ?????????parent
            p2 = choice(self.keep_top_k[k])
            cand = []
            # ?????????????????????????????????
            # factor = [3, 9, 9, 3]
            # len1 = 0
            # len2 = 0
            # for i in range(4):
            #     len1 += int(factor[i] * self.deepen_factor_range[p1[i + 5]])
            #     len2 += int(factor[i] * self.deepen_factor_range[p2[i + 5]])
            # len1 = p1[self.panas_layer + 1] # parent1?????????
            # len2 = p2[self.panas_layer + 1] # parent2?????????
            # max_len = len1 if len1 > len2 else len2
            # min_len = len2 if len1 > len2 else len1
            # max_p = p1 if len1 > len2 else p2 # max parent
            for i in range(len(p1)): # p1?????????
                rand = np.random.randint(2) # ???0.5??????????????????parent???????????????????????????
                if rand:
                    cand.append(p2[i])
                else:
                    cand.append(p1[i])

            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0: # res??????????????? ??????2
            max_iters -= 1
            cand_tuple = next(cand_iter)
            rank, world_size = get_dist_info()
            cand_tuple = get_broadcast_cand(cand_tuple, self.distributed, rank)
            # (0, 1, 2, 0, 0, 0, 0, 0, 0)
            cand = tuple_to_dict(cand_tuple)
            cand = get_broadcast_cand(cand, self.distributed, rank)
            # {'widen_factor_idx': (0, 1, 2, 0, 0), 'deepen_factor_idx': (0, 0, 0, 0)}

            if not self.is_legal(cand): # ??????map?????????
                continue
            res.append(cand_tuple)
            # [(0, 1, 2, 0, 0, 0, 0, 0, 0), (2, 1, 1, 0, 0, 0, 0, 0, 0)]
            print('crossover {}/{}'.format(len(res), crossover_num))

            logging.info(
                'crossover {}/{}, arch: {}, AP {}, {} M, {} GFLOPS'.format(len(res), crossover_num, cand,
                                                                                self.vis_dict[cand_tuple][
                                                                                   'map_list'],
                                                                               self.vis_dict[cand_tuple]['size'],
                                                                               self.vis_dict[cand_tuple]['fp']))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        rank, _ = get_dist_info()
        if rank == 0:
            print(
                'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                    self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                    self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
            logging.basicConfig(filename=self.logfile, level=logging.INFO)
            print(self.logfile) # .workdir/summary/latest_ea_disFalse_fp_None_20220322_171439.log
            logging.info(self.cfg)

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            if rank == 0:
                print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand) # ?????????epoch?????????cand?????????
                # self.candidates:[[(0, 3, 2, 0, 0, 0, 0, 0, 0), (2, 2, 1, 3, 3, 0, 0, 0, 0), (3, 1, 3, 3, 1, 0, 0, 0, 0), (2, 1, 1, 3, 1, 0, 0, 0, 0)]]
            # ?????????
            self.update_top_k( # ?????? ??? ????????? self.candidates
                self.candidates,
                k=self.select_num,
                key=lambda x: self.vis_dict[x]['map'])
            # self.keep_top_k:
            # {2: [(2, 1, 1, 3, 1, 0, 0, 0, 0), (0, 3, 2, 0, 0, 0, 0, 0, 0)],
            # 50: []}

            # self.candidates # 4???????????????
            # [(0, 3, 2, 0, 0, 0, 0, 0, 0), (2, 2, 1, 3, 3, 0, 0, 0, 0), (3, 1, 3, 3, 1, 0, 0, 0, 0), (2, 1, 1, 3, 1, 0, 0, 0, 0)]
            self.update_top_k(# ??????50?????? ???????????????????????? ??? ??????50??????????????????50????????????
                self.candidates,
                k=50,
                key=lambda x: self.vis_dict[x]['map'])
            # self.keep_top_k
            # {2: [(2, 1, 1, 3, 1, 0, 0, 0, 0), (0, 3, 2, 0, 0, 0, 0, 0, 0)],
            # 50: [(2, 1, 1, 3, 1, 0, 0, 0, 0), (0, 3, 2, 0, 0, 0, 0, 0, 0),
            # (2, 2, 1, 3, 3, 0, 0, 0, 0), (3, 1, 3, 3, 1, 0, 0, 0, 0)]}

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 map = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['map']))
                cand_dict = tuple_to_dict(cand)

                arch = [cand]
                # , AP {}, {} M, {} GFLOPS
                logging.info(
                    'No.{} arch: {}, {}, AP = {},  {} M, {} GLOPS'.format(i + 1, cand, arch,
                                                                          self.vis_dict[cand]['map_list'],
                                                                          self.vis_dict[cand]['size'],
                                                                          self.vis_dict[cand]['fp']
                                                                          ))

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
            if rank == 0:
                print("mutation over")

            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            print("self.candidates")
            print(self.candidates)

            self.get_random(self.population_num) # ???????????????????????????candidates

            self.epoch += 1
            self.save_checkpoint()

            # torch.cuda.empty_cache()

        self.save_checkpoint()


if __name__ == '__main__':
    args = parse_args()
    searcher = EvolutionSearcher(args)
    searcher.search()

    # python tools/search/search.py ??????
