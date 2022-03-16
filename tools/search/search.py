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
        self.population_num = args.population_num # 50
        self.m_prob = args.m_prob # 0.1
        self.crossover_num = args.crossover_num # 25
        self.mutation_num = args.mutation_num # 25
        self.flops_limit = args.flops_limit # None (float) # 17.651 M 122.988 GFLOPS
        self.input_shape = (3,) + tuple(args.shape) # default=[1280, 800]  [3,1280,800]?

        self.cfg, self.meta = get_cfg(self.args)
        # 获取cfg文件所有内容道meta上

        # self.cfg.model['search_backbone'] = True
        # self.cfg.model['search_neck'] = True
        # self.cfg.model['search_head'] = True
        self.train_dataset = get_train_data(self.cfg)
        self.cfg = self.cfg.copy()
        print("before get model")
        # bug!! The model and loaded state dict do not match exactly #todo
        self.model, self.distributed = get_model(self.cfg, self.args)
        print("after get model")

        # dataset
        self.test_dataset, self.test_data_loader = get_test_data(self.cfg.data.test, self.distributed, self.args)
        # self.train_dataset, self.train_data_loader = get_test_data(self.cfg.data.train, self.distributed, self.args)

        self.panas_c_range = self.cfg.get('panas_c_range', None) # panas_c_range = [16, 64]
        # self.panas_d_range = self.cfg.get('panas_d_range', None) # None
        # self.panas_d_range = [1,1]
        # self.head_d_range = self.cfg.get('head_d_range', None)
        # self.panas_layer = self.panas_d_range[1]
        # self.panas_state = self.cfg.get('panas_type', None)
        # self.cb_step = self.cfg.get('cb_step', None)
        # self.cb_type = self.cfg.get('cb_type', None)
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

    def get_param(self, cand):
        panas_arch = [self.primitives[i] for i in cand['panas_arch'][:cand['panas_d']]]
        if 'r18' in self.args.config:
            cfg = Config.fromfile('configs/panasv2_fpn/cascade_rcnn_r18_panasv2_fp_1x_coco.py')
        if '2r50' in self.args.config:
            if cand['cb_step'] == 1:
                cfg = Config.fromfile('configs/panasv2_fpn/cascade_rcnn_2r50dcn_panasv2_fp_1x_coco.py')
            else:
                cfg = Config.fromfile('configs/panasv2_fpn/cascade_rcnn_db2r50dcn_panasv2_fp_1x_coco.py')
        if '2r101' in self.args.config:
            if cand['cb_step'] == 1:
                cfg = Config.fromfile('configs/panasv2_fpn/cascade_rcnn_2r101dcn_panasv2_fp_1x_coco.py')
            else:
                cfg = Config.fromfile('configs/panasv2_fpn/cascade_rcnn_db2r101dcn_panasv2_fp_1x_coco.py')
        elif 'swin-t' in self.args.config:
            if cand['cb_step'] == 1:
                cfg = Config.fromfile('configs/panasv2_fpn/cascade_mask_rcnn_swin-t-p4-w7_PANASv2_fp_1x_coco.py')
            else:
                cfg = Config.fromfile('configs/panasv2_fpn/cascade_mask_rcnn_dbswin-t-p4-w7_PANASv2_fp_1x_coco.py')
        elif 'swin-s' in self.args.config:
            if cand['cb_step'] == 1:
                cfg = Config.fromfile('configs/panasv2_fpn/mask_rcnn_swin-s-p4-w7_PANASv2_fp_1x_coco.py')
            else:
                cfg = Config.fromfile('configs/panasv2_fpn/mask_rcnn_dbswin-s-p4-w7_PANASv2_fp_1x_coco.py')

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        cfg.model.neck.panas_arch = panas_arch
        channel = cand['panas_c']
        cfg.model.neck.out_channels = channel
        cfg.model.rpn_head.in_channels, cfg.model.rpn_head.feat_channels = channel, channel
        cfg.model.roi_head.bbox_roi_extractor.out_channels = channel
        if 'cascade' in self.args.config:
            head_step = cand['head_step']
            cfg.model.roi_head.num_stages = head_step
            cfg.model.roi_head.bbox_head = cfg.model.roi_head.bbox_head[:head_step]
            for i in range(head_step):
                cfg.model.roi_head.bbox_head[i].in_channels = channel
        if 'mask' in self.args.config:
            # cfg.model.roi_head.bbox_head.in_channels = channel
            cfg.model.roi_head.mask_roi_extractor.out_channels = channel
            cfg.model.roi_head.mask_head.in_channels = channel
            cfg.model.roi_head.mask_head.conv_out_channels = channel

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

        print(params, flops, cand, panas_arch)
        torch.cuda.empty_cache()
        del model
        return params, flops, cfg

    def is_legal(self, arch, **kwargs):
        rank, world_size = get_dist_info()
        cand = dict_to_tuple(arch)
        with open(self.log_dir + '_gpu_{}.txt'.format(rank), 'a') as f:
            f.write(str(cand) + '\n')

        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        print(arch)
        size, fp, cfg = self.get_param(arch)
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
        self.model.set_arch(arch, **kwargs)

        map = get_cand_map(self.model, self.args, self.distributed, self.cfg, self.test_data_loader, self.test_dataset)
        # map = tuple([0.]*6)
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

    def get_random(self, num):
        print('random select ........')
        #lambda: {'panas_arch': tuple([np.random.randint(self.panas_state) for i in range(self.panas_layer)]),
        # AttributeError: 'EvolutionSearcher' object has no attribute 'panas_layer'
        cand_iter = self.stack_random_cand(
            lambda: {
                     # 'panas_arch': tuple([np.random.randint(self.panas_state) for i in range(self.panas_layer)]),
                     'panas_c': np.random.randint(self.panas_c_range[0], self.panas_c_range[1]) // 16 * 16,
                     # 'panas_d': np.random.randint(self.panas_d_range[0], self.panas_d_range[1] + 1),
                     # 'cb_type': np.random.randint(self.cb_type),
                     # 'cb_step': np.random.randint(1, self.cb_step + 1),
                     # 'head_step': np.random.randint(self.head_d_range[0],
                     #                                self.head_d_range[1] + 1) if 'cascade' in self.args.config else 0,

                     })
        while len(self.candidates) < num:
            cand = next(cand_iter)
            rank, world_size = get_dist_info()
            cand = get_broadcast_cand(cand, self.distributed, rank)
            cand_tuple = dict_to_tuple(cand)
            cand_tuple = check_cand(cand_tuple, self.search_head, self.search_neck, self.search_backbone,
                                    self.panas_layer)
            cand = tuple_to_dict(cand_tuple, self.panas_layer)
            if not self.is_legal(cand):
                continue

            self.candidates.append(cand_tuple)
            print('random {}/{}'.format(len(self.candidates), num))
            panas_arch = [self.primitives[i] for i in cand['panas_arch'][:cand['panas_d']]]
            logging.info(
                'random {}/{}, arch: {}, {}, AP {}, {} M, {} GFLOPS'.format(len(self.candidates), num, cand, panas_arch,
                                                                            self.vis_dict[cand_tuple]['map_list'],
                                                                            self.vis_dict[cand_tuple]['size'],
                                                                            self.vis_dict[cand_tuple]['fp']))

        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
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

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            cand = []
            len1 = p1[self.panas_layer + 1]
            len2 = p2[self.panas_layer + 1]
            max_len = len1 if len1 > len2 else len2
            min_len = len2 if len1 > len2 else len1
            max_p = p1 if len1 > len2 else p2
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
        if rank == 0:
            print(
                'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                    self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                    self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
            logging.basicConfig(filename=self.logfile, level=logging.INFO)
            print(self.logfile)
            logging.info(self.cfg)

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

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
                cand_dict = tuple_to_dict(cand, self.panas_layer)
                panas_arch = [self.primitives[i] for i in cand_dict['panas_arch'][:cand_dict['panas_d']]]
                logging.info(
                    'No.{} arch: {}, {}, AP = {},  {} M, {} GLOPS'.format(i + 1, cand_dict, panas_arch,
                                                                          self.vis_dict[cand]['map_list'],
                                                                          self.vis_dict[cand]['size'],
                                                                          self.vis_dict[cand]['fp']))

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1
            self.save_checkpoint()

            # torch.cuda.empty_cache()

        self.save_checkpoint()


if __name__ == '__main__':
    args = parse_args()
    searcher = EvolutionSearcher(args)
    searcher.search()

    # python tools/search/search.py 运行？
