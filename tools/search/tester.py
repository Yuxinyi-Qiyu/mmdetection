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
from mmdet.utils import collect_env, get_root_logger
from mmdet.apis import init_random_seed, set_random_seed, train_detector



def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

@no_grad_wrapper
def get_cand_map(model, args, distributed, cfg, data_loader, dataset):

    dataset_test = dataset
    data_loader_test = data_loader

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader_test, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader_test, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset_test.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset_test.evaluate(outputs, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            print(metric)
            map = []
            for key in metric:
                map.append(metric[key])
            return tuple(map[:-1])

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
