# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
from torch import nn
from mmcv import Config, DictAction
from thop.vision.basic_hooks import count_parameters
from thop.profile import prRed, register_hooks
from mmdet.models import build_detector

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
from mmdet.models.utils.usconv import USConv2d, USBatchNorm2d, \
    count_usconvNd_flops, count_usconvNd_params, count_usbn_flops, count_usbn_params

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
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
        '--size-divisor',
        type=int,
        default=32,
        help='Pad the input image, the minimum size that is divisible '
        'by size_divisor, -1 means do not pad the image.')
    args = parser.parse_args()
    return args

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

def main():

    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    orig_shape = (3, h, w)
    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor

    input_shape = (3, h, w)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    custom_ops = {
        # custom
        USConv2d: (count_usconvNd_flops, count_usconvNd_params),
        USBatchNorm2d: (count_usbn_flops, count_usbn_params),
        nn.Conv2d: (count_usconvNd_flops, count_usconvNd_params),
        nn.BatchNorm2d: (count_usbn_flops, count_usbn_params)
    }

    flops, params = my_get_model_complexity_info(model, input_shape, custom_ops=custom_ops)

    # flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30

    if divisor > 0 and \
            input_shape != orig_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {orig_shape} to {input_shape}\n')
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
