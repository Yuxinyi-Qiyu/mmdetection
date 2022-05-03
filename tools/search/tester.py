# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import numpy as np
import pickle
import shutil
import tempfile

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.utils import setup_multi_processes
import logging
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmdet.core import encode_mask_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

@no_grad_wrapper
def get_cand_map(model, args, distributed, cfg, train_data_loader, train_dataset, test_data_loader, test_dataset):
    #todo：map=0？而且第一次test后不计算map？
    model.train()
    if not distributed:
        model_P = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model_P, train_data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model_P = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

        outputs = multi_gpu_test(model_P, train_data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            train_dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)

            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print("eval_kwargs")
            print(eval_kwargs)
            metric = train_dataset.evaluate(outputs, **eval_kwargs)
            print(metric)

    model.eval()
    if not distributed: # False
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, test_data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

        outputs = multi_gpu_test(model, test_data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info() # rank = 0
    if rank == 0:
        if args.out: # None
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            test_dataset.format_results(outputs, **kwargs)
        if args.eval: # bbox
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = test_dataset.evaluate(outputs, **eval_kwargs)

            metric_dict = dict(config=args.config, metric=metric)
            print(metric)

            #OrderedDict([('bbox_mAP', 0.0), ('bbox_mAP_50', 0.0), ('bbox_mAP_75', 0.0), ('bbox_mAP_s', 0.0), ('bbox_mAP_m', 0.0), ('bbox_mAP_l', 0.0), ('bbox_mAP_copypaste', '0.000 0.000 0.000 0.000 0.000 0.000')])
            map = []
            for key in metric:
                map.append(metric[key])
            return tuple(map[:-1]) # (0.6599323749542236,)

    return None

@no_grad_wrapper
def forward_model(model, distributed, data_loader, max_iters=0):
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.train()
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            if i > max_iters:
                break
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            batch_size = len(result)
            del result
            for _ in range(batch_size):
                prog_bar.update()
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        model.train()
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        time.sleep(2)  # This line can prevent deadlock problem in some cases.
        for i, data in enumerate(data_loader):
            if i > max_iters:
                break
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            if rank == 0:
                batch_size = len(result)
                for _ in range(batch_size * world_size):
                    prog_bar.update()
                del result


if __name__ == '__main__':
    main()
