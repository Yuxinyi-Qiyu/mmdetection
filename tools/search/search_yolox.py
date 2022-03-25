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

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs # 20
        self.select_num = args.select_num # 10
        # self.population_num = args.population_num # 50
        self.population_num = 10
        self.m_prob = args.m_prob # 0.1
        # self.crossover_num = args.crossover_num # 25
        self.crossover_num = 5
        # self.mutation_num = args.mutation_num # 25
        self.mutation_num = 5
        self.flops_limit = args.flops_limit # None (float) # 17.651 M 122.988 GFLOPS
        self.input_shape = (3,) + tuple(args.shape) # default=[1280, 800]  [3,1280,800] todo?

        self.cfg, self.meta = get_cfg(self.args)
        # 获取cfg文件所有内容到meta上

        # self.cfg.model['search_backbone'] = True
        # self.cfg.model['search_neck'] = True
        # self.cfg.model['search_head'] = True
        self.train_dataset = get_train_data(self.cfg)
        self.cfg = self.cfg.copy()

        self.widen_factor_range = self.cfg.get('widen_factor_range', None)
        self.deepen_factor_range = self.cfg.get('deepen_factor_range', None))

        print("before get model")
        # bug!! The model and loaded state dict do not match exactly # todo
        self.model, self.distributed = get_model(self.cfg, self.args)
        print("after get model")

        # dataset
        self.test_dataset, self.test_data_loader = get_test_data(self.cfg.data.test, self.distributed, self.args)
        # self.train_dataset, self.train_data_loader = get_test_data(self.cfg.data.train, self.distributed, self.args)

        self.panas_c_range = self.cfg.get('panas_c_range', None) # panas_c_range = [16, 64]
        self.panas_d_range = self.cfg.get('panas_d_range', None) # None
        self.head_d_range = self.cfg.get('head_d_range', None)
        self.panas_layer = self.panas_d_range[1]
        self.panas_state = self.cfg.get('panas_type', None)
        self.cb_step = self.cfg.get('cb_step', None)
        self.cb_type = self.cfg.get('cb_type', None)
        self.primitives = self.cfg.get('primitives', None)

        # self.search_backbone = self.cfg.model.get('search_backbone', None)
        # self.search_neck = self.cfg.model.get('search_neck', None)
        # self.search_head = self.cfg.model.get('search_head', None)

        self.search_backbone = True
        self.search_neck = True
        self.search_head = False
        # self.log_dir = os.path.join(os.path.split(args.checkpoint)[0], 'ea')
        self.log_dir = '.workdir/summary'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.checkpoint_name = os.path.join(self.log_dir, 'ea_'
                                                          'checkpoint.pth.tar')
        loaded_checkpoint = os.path.split(args.checkpoint)[1].split('.')[0]
        times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # default 1600*2*4
        print(loaded_checkpoint, self.log_dir)
        self.logfile = os.path.join(self.log_dir, '{}_ea_dis{}_fp_{}_{}.log'.format(loaded_checkpoint, self.distributed,
                                                                                    self.flops_limit, times))

        # 这些是啥 checkpoint 相关
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

    def get_depth_cand(self, cand, depth):
        cand = list(cand)
        for i in range(depth, self.panas_layer):
            cand[i] = -1
        cand = tuple(cand)
        return cand

    def get_param(self, cand): # todo
        # panas_arch = [self.primitives[i] for i in cand['panas_arch'][:cand['panas_d']]]
        panas_arch = cand
        cfg = Config.fromfile('configs/yolox/yolox_s_8x8_300e_coco_searchable.py')

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)


        # cfg.model.neck.panas_arch = panas_arch
        # channel = cand['panas_c']
        # cfg.model.neck.out_channels = channel
        # cfg.model.rpn_head.in_channels, cfg.model.rpn_head.feat_channels = channel, channel
        # cfg.model.roi_head.bbox_roi_extractor.out_channels = channel
        # if 'cascade' in self.args.config:
        #     head_step = cand['head_step']
        #     cfg.model.roi_head.num_stages = head_step
        #     cfg.model.roi_head.bbox_head = cfg.model.roi_head.bbox_head[:head_step]
        #     for i in range(head_step):
        #         cfg.model.roi_head.bbox_head[i].in_channels = channel
        # if 'mask' in self.args.config:
        #     #cfg.model.roi_head.bbox_head.in_channels = channel
            # cfg.model.roi_head.mask_roi_extractor.out_channels = channel
            # cfg.model.roi_head.mask_head.in_channels = channel
            # cfg.model.roi_head.mask_head.conv_out_channels = channel

        # cfg.model.module.set_arch(panas_arch, **kwargs)
        # todo 如何修改网络宽度、深度

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
        flops, params = get_model_complexity_info(model, self.input_shape, as_strings=False, print_per_layer_stat=False)
        flops = round(flops / 10. ** 9, 2)
        params = round(params / 10 ** 6, 2)

        print(params, flops, cand, panas_arch) #46.01 50.05 {'widen_factor': 0.4911868497913349} {'widen_factor': 0.4911868497913349}

        torch.cuda.empty_cache()
        del model
        return params, flops, cfg

    def is_legal(self, arch, **kwargs):
        # cand = {'widen_factor':cand_tuple[0]}
        rank, world_size = get_dist_info() # 0,1
        cand = dict_to_tuple(arch) # (0.4911868497913349,)

        with open(self.log_dir + '_gpu_{}.txt'.format(rank), 'a') as f:
            f.write(str(cand) + '\n')

        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        print(arch) # {'widen_factor': 0.4911868497913349}
        size, fp, cfg = self.get_param(arch)
        print(size, fp) # 46.01 50.05 Config
        # size, fp, cfg = 0, 0, 0
        size = get_broadcast_cand(size, self.distributed, rank)
        fp = get_broadcast_cand(fp, self.distributed, rank)

        if self.flops_limit and fp > self.flops_limit:
            del size, fp, cfg
            return False
        info['fp'] = fp
        info['size'] = size
        info['cfg'] = cfg
        del size, fp, cfg
        self.model.set_arch(arch, **kwargs) # 这里set_ARCH,修改了模型宽度

        print("after model set arch")

        map = get_cand_map(self.model, self.args, self.distributed, self.cfg, self.test_data_loader, self.test_dataset)
        # map = tuple([0.]*6)
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
            info['map'] = map[0]
            info['map_list'] = map
            info['visited'] = True
            del map
            return True

        return False

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]
        print(self.keep_top_k)

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if isinstance(cand, dict):
                    cand = dict_to_tuple(cand)
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand
        print("cand")
        print(cand)
        print("info")
        print(info)

    def get_random(self, num): # num=population_num
        print('random select ........')
        # lambda: {'panas_arch': tuple([np.random.randint(self.panas_state) for i in range(self.panas_layer)]),
        # AttributeError: 'EvolutionSearcher' object has no attribute 'panas_layer'
        cand_iter = self.stack_random_cand(
            lambda: {
                     # 'widen_factor': np.random.uniform(self.widen_factor_range[0] + 0.01, self.widen_factor_range[1])
                     'widen_factor': tuple([self.widen_factor_range[np.random.randint(0, len(self.widen_factor_range))] for i in range(5)]),
                     'deepen_factor': tuple([self.deepen_factor_range[np.random.randint(0, len(self.deepen_factor_range))] for i in range(4)]) # tuple-->(1,2,3,4)/no tuple-->[1,2,3,4]
                    })
        while len(self.candidates) < num: # 候选子网少于population num
            cand = next(cand_iter) # 再生成一个arch
            rank, world_size = get_dist_info()
            cand = get_broadcast_cand(cand, self.distributed, rank)
            cand_tuple = dict_to_tuple(cand) # (0.4911868497913349,) // (0.4911868497913349, 2, 0, 0, 2, 2)
            # (0.4911868497913349, 2, 0, 0, 2, 2, 96, 4, 0, 2, 0)
            # cand_tuple = check_cand(cand_tuple, self.search_head, self.search_neck, self.search_backbone,
            #                         self.panas_layer)
            # (0.4911868497913349, 2, 0, 0, 2, [-1], 96, 4, 0, 2, 0)
            # cand = tuple_to_dict(cand_tuple, self.panas_layer)
            # {'panas_arch': (0.4911868497913349, 2, 0, 0, 2), 'panas_c': [-1], 'panas_d': 96, 'cb_type': 4, 'cb_step': 0, 'head_step': 2}
            cand = {'widen_factor':cand_tuple[0]}
            if not self.is_legal(cand):
                continue

            self.candidates.append(cand_tuple)
            print('random {}/{}'.format(len(self.candidates), num))
            panas_arch = [cand]
            # , AP {}, {} M, {} GFLOPS
            logging.info(
                'random {}/{}, arch: {}, {}'.format(len(self.candidates), num, cand, panas_arch,
                                                                            # self.vis_dict[cand_tuple]['map_list'],
                                                                            # self.vis_dict[cand_tuple]['size'],
                                                                            # self.vis_dict[cand_tuple]['fp']
                                                                            ))

        print('random_num = {}'.format(len(self.candidates)))
        print(self.candidates)

    def get_mutation(self, k, mutation_num, m_prob): # self.select_num, self.mutation_num, self.m_prob
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))

            for i in range(self.panas_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.panas_state)
            if np.random.random_sample() < m_prob:
                cand[self.panas_layer] = np.random.randint(self.panas_c_range[0], self.panas_c_range[1]) // 16 * 16
            depth_now = cand[self.panas_layer + 1]
            if np.random.random_sample() < m_prob:
                depth = np.random.randint(self.panas_d_range[0], self.panas_d_range[1] + 1)
                if depth < depth_now:
                    cand[self.panas_layer + 1] = depth
                elif depth > depth_now:
                    for i in range(depth_now, depth):
                        cand[i] = np.random.randint(self.panas_state)

            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand_tuple = next(cand_iter)
            rank, world_size = get_dist_info()
            cand_tuple = get_broadcast_cand(cand_tuple, self.distributed, rank)
            cand_tuple = check_cand(cand_tuple, self.search_head, self.search_neck, self.search_backbone,
                                    self.panas_layer)
            cand = tuple_to_dict(cand_tuple, self.panas_layer)
            cand = get_broadcast_cand(cand, self.distributed, rank)

            if not self.is_legal(cand):
                continue
            res.append(cand_tuple)
            print('mutation {}/{}'.format(len(res), mutation_num))
            panas_arch = [self.primitives[i] for i in cand['panas_arch'][:cand['panas_d']]]
            logging.info(
                'mutation {}/{}, arch: {}, {}, AP {}, {} M, {} GFLOPS'.format(len(res), mutation_num, cand, panas_arch,
                                                                              self.vis_dict[cand_tuple][
                                                                                  'map_list'],
                                                                              self.vis_dict[cand_tuple]['size'],
                                                                              self.vis_dict[cand_tuple]['fp']))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num): # 交叉
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            cand = []
            len1 = p1[self.panas_layer + 1] # parent1的深度
            len2 = p2[self.panas_layer + 1] # parent2的深度
            max_len = len1 if len1 > len2 else len2
            min_len = len2 if len1 > len2 else len1
            max_p = p1 if len1 > len2 else p2 # max parent
            for i in range(len(p1)):
                rand = np.random.randint(2)
                if rand:
                    cand.append(p2[i])
                else:
                    cand.append(p1[i])

            if cand[self.panas_layer + 1] == max_len:
                for i in range(min_len, max_len):
                    cand[i] = max_p[i]

            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand_tuple = next(cand_iter)
            rank, world_size = get_dist_info()
            cand_tuple = get_broadcast_cand(cand_tuple, self.distributed, rank)
            cand_tuple = check_cand(cand_tuple, self.search_head, self.search_neck, self.search_backbone,
                                    self.panas_layer)
            cand = tuple_to_dict(cand_tuple, self.panas_layer)
            cand = get_broadcast_cand(cand, self.distributed, rank)

            if not self.is_legal(cand):
                continue
            res.append(cand_tuple)
            print('crossover {}/{}'.format(len(res), crossover_num))
            panas_arch = [self.primitives[i] for i in cand['panas_arch'][:cand['panas_d']]]

            logging.info(
                'crossover {}/{}, arch: {}, {}, AP {}, {} M, {} GFLOPS'.format(len(res), crossover_num, cand,
                                                                               panas_arch, self.vis_dict[cand_tuple][
                                                                                   'map_list'],
                                                                               self.vis_dict[cand_tuple]['size'],
                                                                               self.vis_dict[cand_tuple]['fp']))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        rank, _ = get_dist_info()
        print(rank) # 0
        if rank == 0:
            print(
                'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                    self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                    self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
            logging.basicConfig(filename=self.logfile, level=logging.INFO)
            print(self.logfile) # .workdir/summary/latest_ea_disFalse_fp_None_20220322_171439.log
            logging.info(self.cfg)

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:  # 0,20
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates: # ?? # todo 应该在这里就把factor进行统一，是几个选项中的随机
                self.memory[-1].append(cand)
            print("self.memory")
            print(self.memory) # 候选子网
            # [[(0.4911868497913349,), (0.6888551943359122,), (0.9857202901549829,), (0.2036404460606003,), (0.5875775567487878,), (0.9807763553283119,), (0.18867549441605322,), (0.7044539487313805,), (0.8478464501767815,), (0.7987422023007866,)]]
            # 这里就已经把map求出来了，如何求的？
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['map'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['map'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 map = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['map']))
                # cand_dict = tuple_to_dict(cand, self.panas_layer)
                cand_dict = {'widen_factor':cand[0]}

                panas_arch = [cand]
                # , AP {}, {} M, {} GFLOPS
                logging.info(
                    'No.{} arch: {}, {}, AP = {},  {} M, {} GLOPS'.format(i + 1, cand, panas_arch,
                                                                          self.vis_dict[cand]['map_list'],
                                                                          self.vis_dict[cand]['size'],
                                                                          self.vis_dict[cand]['fp']
                                                                          ))

                # panas_arch = [self.primitives[i] for i in cand_dict['panas_arch'][:cand_dict['panas_d']]]
                # logging.info(
                #     'No.{} arch: {}, {}, AP = {},  {} M, {} GLOPS'.format(i + 1, cand_dict, panas_arch,
                #                                                           self.vis_dict[cand]['map_list'],
                #                                                           self.vis_dict[cand]['size'],
                #                                                           self.vis_dict[cand]['fp']))

            # channel太少，搜索空间太小，不用mutation、crossover

            # mutation = self.get_mutation(
            #     self.select_num, self.mutation_num, self.m_prob)
            # crossover = self.get_crossover(self.select_num, self.crossover_num)
            #
            # self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1
            self.save_checkpoint()

            # torch.cuda.empty_cache()

        self.save_checkpoint()


if __name__ == '__main__':
    args = parse_args()
    searcher = EvolutionSearcher(args)
    print("fin")
    searcher.search()

    # python tools/search/search.py 运行
