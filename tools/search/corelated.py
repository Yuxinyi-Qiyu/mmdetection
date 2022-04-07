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
from trainer import get_train_data, train_model, get_model, get_cfg
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
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()



class EvolutionSearcher1(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs # 20
        # self.select_num = args.select_num # 10
        self.select_num = 1 # 10
        # self.population_num = args.population_num # 50
        self.population_num = 2
        self.m_prob = args.m_prob # 0.1
        # self.crossover_num = args.crossover_num # 25
        self.crossover_num = 2
        # self.mutation_num = args.mutation_num # 25
        self.mutation_num = 2
        self.flops_limit = args.flops_limit # None (float) # 17.651 M 122.988 GFLOPS
        self.input_shape = (3,) + tuple(args.shape) # default=[1280, 800]  [3,1280,800] todo?

        self.cfg, self.meta = get_cfg(self.args)
        # 获取cfg文件所有内容到meta上

        self.train_dataset = get_train_data(self.cfg)
        self.cfg = self.cfg.copy()

        print("before get model")
        # bug!! The model and loaded state dict do not match exactly # todo
        self.model, self.distributed = get_model(self.cfg, self.args)
        print("after get model")

        # dataset
        self.test_dataset, self.test_data_loader = get_test_data(self.cfg.data.test, self.distributed, self.args)
        # self.train_dataset, self.train_data_loader = get_test_data(self.cfg.data.train, self.distributed, self.args)

        self.widen_factor_range = self.cfg.get('widen_factor_range', None)
        self.deepen_factor_range = self.cfg.get('deepen_factor_range', None)
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

    def get_param(self, cand): # todo
        panas_arch = cand # {'widen_factor': (0.625, 0.5, 0.375, 0.125, 0.125), 'deepen_factor': (0.33, 0.33, 1, 1)}
        cfg = Config.fromfile('configs/yolox/yolox_s_8x8_300e_coco_searchable.py')

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

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

        map = get_cand_map(self.model, self.args, self.distributed, self.cfg, self.test_data_loader, self.test_dataset)
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

        cand = {'widen_factor': (0.875, 0.375, 0.75, 0.5, 0.625), 'deepen_factor': (1, 0.33, 0.33, 1)}
        map = self.is_legal(cand)
        print("cand:" + str(cand))
        print("map:" + str(map))

        # # self.load_checkpoint()
        # for k in [0, 1, 2, 3, 4]:
        #     for i in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
        #         logging.info('widen: index: {}, factor:{}'.format(k, i))
        #         l_wide = [1.0, 1.0, 1.0, 1.0, 1.0]
        #         l_wide[k] = i
        #         cand =  {'widen_factor': tuple(l_wide), 'deepen_factor': (1.0, 1.0, 1.0, 1.0)}
        #         map = self.is_legal(cand)
        #         print("cand:"+str(cand))
        #         print("map:"+str(map))
        #         logging.info('arch: {}, map:{}'.format(cand, map))
        # for k in [0, 1, 2, 3]:
        #     for i in [0.33, 1.0]:
        #         logging.info('deepen: index: {}, factor:{}'.format(k, i))
        #         l_deep = [1.0, 1.0, 1.0, 1.0]
        #         l_deep[k] = i
        #         cand =  {'widen_factor': (1.0, 1.0, 1.0, 1.0), 'deepen_factor': tuple(l_deep)}
        #         map = self.is_legal(cand)
        #         print("cand:"+str(cand))
        #         print("map:"+str(map))
        #         logging.info('arch: {}, map:{}'.format(cand, map))



if __name__ == '__main__':
    args = parse_args()
    searcher = EvolutionSearcher1(args)
    print("fin")
    searcher.search()

    # python tools/search/search.py 运行
