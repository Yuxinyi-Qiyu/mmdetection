from types import MethodType
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.models.utils import make_divisible

def add_prefix(inputs, prefix): # 前缀
    """Add prefix for dict.
    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.
    Returns:
        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs

class SearchBase:
    def __init__(self, bn_training_mode=True, num_sample_training=4, divisor=8, retraining=False) -> None: #
        self.bn_training_mode = bn_training_mode
        self.num_sample_training = num_sample_training
        self.divisor = divisor
        self.retraining = retraining

        for name, module in self.named_modules():
            self.add_nas_attrs(module)

    @staticmethod
    def modify_conv_forward(module): # conv类型变化
        """Modify the forward method of a conv layer."""
        def modified_forward(self, feature):
            # print('here')   #? 怎么调用的
            assert self.groups == 1
            # print(self.in_channels,self.out_channels)
            # print(feature.size())
            weight = self.weight[:self.out_channels, :self.in_channels, :, :]
            if self.bias is not None:
                bias = self.bias[:self.out_channels]
            else:
                bias = self.bias
            return self._conv_forward(feature, weight) # , bias

        return MethodType(modified_forward, module)
        # 将modified forward方法绑定到module实例上

    @staticmethod
    def modify_fc_forward(module): # fc类型变化
        """Modify the forward method of a linear layer."""
        def modified_forward(self, feature):
            weight = self.weight[:self.out_features, :self.in_features]
            if self.bias is not None:
                bias = self.bias[:self.out_features]
            else:
                bias = self.bias
            return F.linear(feature, weight, bias)

        return MethodType(modified_forward, module)

    @staticmethod
    def modify_seq_forward(module): # deep变化
        """Modify the forward method of a sequential container."""
        def modified_forward(self, input):
            for module in self[:self.num_layers]:
                input = module(input)
            return input

        return MethodType(modified_forward, module)

    @staticmethod
    def modify_bn_forward(module): # @todo:unknown
        """Modify the forward method of a linear layer."""
        def modified_forward(self, feature):
            self._check_input_dim(feature)
            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that it gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:  # type: ignore[has-type]
                    self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            r"""
            Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
            passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
            used for normalization (i.e. in eval mode when buffers are not None).
            """
            return F.batch_norm(
                feature,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean[:self.num_features]
                if not self.training or self.track_running_stats
                else None,
                self.running_var[:self.num_features] if not self.training or self.track_running_stats else None,
                self.weight[:self.num_features],
                self.bias[:self.num_features],
                bn_training,
                exponential_average_factor,
                self.eps,
            )

        return MethodType(modified_forward, module)

    def add_nas_attrs(self, module):
        """Add masks to a ``nn.Module``."""
        if isinstance(module, nn.Conv2d):
            module.forward = self.modify_conv_forward(module)
        if isinstance(module, nn.Linear):
            module.forward = self.modify_fc_forward(module)
        if isinstance(module, _BatchNorm):
            module.forward = self.modify_bn_forward(module)
        if isinstance(module, nn.Sequential):
            module.num_layers = len(module)
            module.forward = self.modify_seq_forward(module)

    def set_archs(self, archs, **kwargs):
        # print("searchbase_set_archs")
        self.archs = archs
        # print(self.archs)

    def set_arch(self, arch, **kwargs):
        # print("searchbase_set_arch")
        self.arch = arch
        self.backbone.set_arch(self.arch)
        self.neck.set_arch(self.arch)
        self.bbox_head.set_arch(self.arch)
        # print(self.arch)

    def train(self, mode=True):
        """Overwrite the train method in `nn.Module` to set `nn.BatchNorm` to
        training mode when model is set to eval mode when
        `self.bn_training_mode` is `True`.
        Args:
            mode (bool): whether to set training mode (`True`) or evaluation
                mode (`False`). Default: `True`.
        """
        super().train(mode)
        if not mode and self.bn_training_mode:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True

    def train_step(self, data, optimizer):
        """Train step function.
        This function implements the standard training iteration for
        autoslim pretraining and retraining.
        Args:
            data (dict): Input data from dataloader.
            optimizer (:obj:`torch.optim.Optimizer`): The optimizer to
                accumulate gradient
        """
        losses = dict()
        if not self.retraining:
            # assert self.pruner is not None

            arch_dict = self.sample_arch(mode='max')
            self.set_arch(arch_dict)
            max_model_losses = self(**data)
            losses.update(add_prefix(max_model_losses, 'max_model'))

            arch_dict = self.sample_arch(mode='min')
            self.set_arch(arch_dict)
            min_model_losses = self(**data)
            losses.update(add_prefix(min_model_losses, 'min_model'))

            for i in range(self.num_sample_training - 2):
                arch_dict = self.sample_arch(mode='random')
                self.set_arch(arch_dict)
                model_losses = self(**data)
                losses.update(
                    add_prefix(model_losses,
                               'sample_model{}'.format(i + 1)))
        else:
            model_losses = self(**data)
            losses.update(add_prefix(model_losses, 'retrain_model'))

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs