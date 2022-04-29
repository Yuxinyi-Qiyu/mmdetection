# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
# python tools/search/search.py
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--nproc_per_node', type=int, help='the dir to save logs and models')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_iters', type=int, default=600)
    parser.add_argument('--samples_per_gpu', type=int, default=2)

    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        default='mAP',
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
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')

    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--flops-limit', type=float, default=None)  # 17.651 M 122.988 GFLOPS
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=[1280, 800])
                        # default=[1080, 720])

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.launcher: # default none
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text #就是cfg文件的内容
    # print("cfg.pretty_text"+str(meta['config']))
    # log some basic info
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    # logger.info(f'Set random seed to {seed}, '
    #             f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # cfg.runner = dict(type='EpochBasedRunner_nolog', max_epochs=1)
    # cfg.total_epochs = 1
    cfg.checkpoint_config = dict(type='CheckpointHook_nolog', interval=50)
    cfg.runner = dict(type='IterBasedRunner_nolog', max_iters=args.max_iters)
    cfg.lr_config = None

    return cfg, meta

def get_train_data(cfg):

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    return datasets

def get_model(cfg, args):
    # model get error?
    # print("cfg.model")
    # print(cfg.model)

# cfg.model
# {'type': 'YOLOX_Searchable', 'input_size': (640, 640), 'random_size_range': (15, 25), 'random_size_interval': 10, 'backbone': {'type': 'CSPDarknet_Searchable', 'conv_cfg': {'type': 'USConv2d'}, 'norm_cfg': {'type': 'USBN2d'}, 'deepen_factor': 1.0, 'widen_factor': 1.0}, 'neck': {'type': 'YOLOXPAFPN_Searchable', 'conv_cfg': {'type': 'USConv2d'}, 'norm_cfg': {'type': 'USBN2d'}, 'in_channels': [256, 512, 1024], 'out_channels': 256, 'num_csp_blocks': 1}, 'bbox_head': {'type': 'YOLOXHead', 'num_classes': 80, 'in_channels': 256, 'feat_channels': 256}, 'train_cfg': {'assigner': {'type': 'SimOTAAssigner', 'center_radius': 2.5}}, 'test_cfg': {'score_thr': 0.01, 'nms': {'type': 'nms', 'iou_threshold': 0.65}}}

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # checkpoint里有两份权重，yolox是会额外存一个ema的权重
    # ema problem：https://githubhot.com/repo/open-mmlab/mmdetection/issues/6156
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    return model, distributed

def train_model(model, datasets, cfg, distributed, meta):
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=False,
        # timestamp=timestamp,
        meta=meta)


if __name__== '__main__':
    args = parse_args()
    model, datasets, cfg, distributed, meta = get_train_data(args)
    get_cand_map(model, datasets, cfg, distributed, meta)




