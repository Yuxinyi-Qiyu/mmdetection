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
from tester import get_cand_map, forward_model
import time
import logging
import numpy as np
from random import choice
import functools
import random
import copy

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs # 20
        # self.max_epochs = 5 # 20
        self.select_num = args.select_num # 10
        # self.select_num = 2 # 10
        # self.population_num = args.population_num # 50
        self.population_num = 4
        self.m_prob = args.m_prob # 0.1
        self.crossover_num = args.crossover_num # 25
        # self.crossover_num = 2
        self.mutation_num = args.mutation_num # 25
        # self.mutation_num = 2
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

    def get_depth_cand(self, cand, depth): # todo 没用上
        cand = list(cand)
        for i in range(depth, self.panas_layer):
            cand[i] = -1
        cand = tuple(cand)
        return cand

    def get_param(self, cand):

        arch = self.idx_to_arch(cand)
        # cfg = Config.fromfile('configs/yolox/yolox_s_8x8_300e_voc_searchable.py')
        cfg = Config.fromfile('configs/yolox/yolox_s_8x8_300e_voc.py')

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
        flops, params = get_model_complexity_info(model, self.input_shape, as_strings=False, print_per_layer_stat=False)
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

        rank, _ = get_dist_info()
        if rank == 0:
            print(size, fp) # 46.01 50.05 Config # 8.65 49.73

        size = get_broadcast_cand(size, self.distributed, rank)
        fp = get_broadcast_cand(fp, self.distributed, rank)

        if (self.flops_limit and fp > self.flops_limit)\
                or (self.params_limit and size > self.params_limit): # 硬件约束筛选
            del size, fp, cfg
            return False

        info['fp'] = fp
        info['size'] = size
        info['cfg'] = cfg
        del size, fp, cfg
        self.model.set_arch(self.idx_to_arch(arch)) # 这里set_ARCH,修改了模型参数

        # 获得当前模型的map
        map = get_cand_map(self.model,
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
            if rank == 0:
                print("self.vis_dict")
                print(self.vis_dict)
            return True

        return False

    def update_top_k(self, candidates, *, k, key, reverse=True):
        # 筛选key排前k=select_num个结构
        # 对candidates里的结构按照map进行排序，选取前k个
        # self.keep_top_k数组里，加入新的candidates中的子网架构，更新后对self.keep_top_k中的子网再进行排序
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
            # cands = 十个子网结构
            for cand in cands:
                if isinstance(cand, dict):
                    cand = dict_to_tuple(cand)
                if cand not in self.vis_dict: # vis_dict 架构对应的info 词典
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            # self.vis_dict: {"(0.125,0.5,0.5,0.5,0.5,0.33,...)":{},"":{},...}
            for cand in cands:
                yield cand

    def get_random(self, num): # num=population_num
        rank, _ = get_dist_info()
        if rank == 0:
            print('random select ........')
        # 生成子网结构
        # cand_iter = self.stack_random_cand(
        #     lambda: {
        #              'widen_factor': tuple([self.widen_factor_range[np.random.randint(0, len(self.widen_factor_range))] for i in range(5)]),
        #              'deepen_factor': tuple([self.deepen_factor_range[np.random.randint(0, len(self.deepen_factor_range))] for i in range(4)]) # tuple-->(1,2,3,4)/no tuple-->[1,2,3,4]
        #             })
        cand_iter = self.stack_random_cand(
            lambda: {
                'widen_factor_backbone_idx': tuple([np.random.randint(0, len(self.widen_factor_range)) for i in range(5)]),
                'deepen_factor_idx': tuple([np.random.randint(0, len(self.deepen_factor_range)) for i in range(4)]),
                'widen_factor_neck_idx': tuple([np.random.randint(0, len(self.widen_factor_range)) for i in range(8)]),
                'widen_factor_neck_out_idx': np.random.randint(0, len(self.widen_factor_range)),
            })
        while len(self.candidates) < num: # 候选子网少于population num
            cand = next(cand_iter) # 再生成一个arch
            # {'widen_factor_idx': (0, 3, 2, 0, 0), 'deepen_factor_idx': (0, 0, 0, 0)}
            rank, world_size = get_dist_info() # 0, 1
            cand = get_broadcast_cand(cand, self.distributed, rank)
            cand_tuple = dict_to_tuple(cand) # cand是dict，转换成数组
            # (0, 3, 2, 0, 0, 0, 0, 0, 0)
            if not self.is_legal(cand): # 是否符合约束，不符合就重新生成子网架构
                continue

            self.candidates.append(cand_tuple) # cand_tuple集合[]
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

        def random_func(): # 以概率prob随机更改每个stage的channel和depth
            cand = list(choice(self.keep_top_k[k]))
            for i in range(5): # 变结构在层数里进行变异 # todo 怎么改
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(0, len(self.widen_factor_range))
            for i in range(4):
                if np.random.random_sample() < m_prob:
                    cand[i + 5] = np.random.randint(0, len(self.deepen_factor_range))
            for i in range(8):
                if np.random.random_sample() < m_prob:
                    cand[i + 9] = np.random.randint(0, len(self.widen_factor_range))
            if np.random.random_sample() < m_prob:
                cand[17] = np.random.randint(0, len(self.widen_factor_range))
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func) # 根据mutation函数，随机生成结构
        while len(res) < mutation_num and max_iters > 0:
            # res个数少于mutation num，且没有达到最大迭代次数
            max_iters -= 1
            cand_tuple = next(cand_iter)
            rank, world_size = get_dist_info()
            cand_tuple = get_broadcast_cand(cand_tuple, self.distributed, rank)
            cand = tuple_to_dict(cand_tuple)
            cand = get_broadcast_cand(cand, self.distributed, rank)
            print(cand)
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

    def get_crossover(self, k, crossover_num): # 交叉
        # 从candidate选择p1、p2
        assert k in self.keep_top_k
        print('crossover ......')
        res = [] # 盛放随机交叉的子网
        iter = 0
        max_iters = 10 * crossover_num # 20

        def random_func():
            # print(self.keep_top_k)
            # {2: [(2, 1, 1, 3, 1, 0, 0, 0, 0), (0, 3, 2, 0, 0, 0, 0, 0, 0)],
            # 50: [(2, 1, 1, 3, 1, 0, 0, 0, 0), (0, 3, 2, 0, 0, 0, 0, 0, 0),
            # (2, 2, 1, 3, 3, 0, 0, 0, 0), (3, 1, 3, 3, 1, 0, 0, 0, 0)]}
            p1 = choice(self.keep_top_k[k]) # 选两个parent
            p2 = choice(self.keep_top_k[k])
            cand = []
            # 目前深度一样，先不考虑
            # factor = [3, 9, 9, 3]
            # len1 = 0
            # len2 = 0
            # for i in range(4):
            #     len1 += int(factor[i] * self.deepen_factor_range[p1[i + 5]])
            #     len2 += int(factor[i] * self.deepen_factor_range[p2[i + 5]])
            # len1 = p1[self.panas_layer + 1] # parent1的深度
            # len2 = p2[self.panas_layer + 1] # parent2的深度
            # max_len = len1 if len1 > len2 else len2
            # min_len = len2 if len1 > len2 else len1
            # max_p = p1 if len1 > len2 else p2 # max parent
            for i in range(len(p1)): # p1的每层
                rand = np.random.randint(2) # 以0.5的概率取两个parent的一个加入新的结构
                if rand:
                    cand.append(p2[i])
                else:
                    cand.append(p1[i])

            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0: # res中的子网数 少于2
            max_iters -= 1
            cand_tuple = next(cand_iter)
            rank, world_size = get_dist_info()
            cand_tuple = get_broadcast_cand(cand_tuple, self.distributed, rank)
            # (0, 1, 2, 0, 0, 0, 0, 0, 0)
            cand = tuple_to_dict(cand_tuple)
            cand = get_broadcast_cand(cand, self.distributed, rank)
            # {'widen_factor_idx': (0, 1, 2, 0, 0), 'deepen_factor_idx': (0, 0, 0, 0)}

            if not self.is_legal(cand): # 计算map，排序
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
                self.memory[-1].append(cand) # 在本次epoch里记录cand的数组
                # self.candidates:[[(0, 3, 2, 0, 0, 0, 0, 0, 0), (2, 2, 1, 3, 3, 0, 0, 0, 0), (3, 1, 3, 3, 1, 0, 0, 0, 0), (2, 1, 1, 3, 1, 0, 0, 0, 0)]]
            # 意义？
            self.update_top_k( # 排序 ， 不改动 self.candidates
                self.candidates,
                k=self.select_num,
                key=lambda x: self.vis_dict[x]['map'])
            # self.keep_top_k:
            # {2: [(2, 1, 1, 3, 1, 0, 0, 0, 0), (0, 3, 2, 0, 0, 0, 0, 0, 0)],
            # 50: []}

            # self.candidates # 4个候选结构
            # [(0, 3, 2, 0, 0, 0, 0, 0, 0), (2, 2, 1, 3, 3, 0, 0, 0, 0), (3, 1, 3, 3, 1, 0, 0, 0, 0), (2, 1, 1, 3, 1, 0, 0, 0, 0)]
            self.update_top_k(# 选前50个？ 为什么又重复一次 ： 更新50对应的数组？50的意义？
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

            self.get_random(self.population_num) # 随机生成子网加入到candidates

            self.epoch += 1
            self.save_checkpoint()

            # torch.cuda.empty_cache()

        self.save_checkpoint()


if __name__ == '__main__':
    args = parse_args()
    searcher = EvolutionSearcher(args)
    searcher.search()

    # python tools/search/search.py 运行
