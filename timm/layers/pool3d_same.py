"""AvgPool3d w/ Same Padding

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from ._fx import register_notrace_module
from .helpers import to_3tuple
from .padding import pad_same, get_padding_value


def avg_pool3d_same(
    x: torch.Tensor,
    kernel_size: List[int],
    stride: List[int],
    ceil_mode: bool = False,
    count_include_pad: bool = True,
):
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool3d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


@register_notrace_module
class AvgPool3dSame(nn.AvgPool3d):
    """Tensorflow like 'SAME' wrapper for 3D average pooling"""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        super().__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool3d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


def max_pool3d_same(
    x: torch.Tensor,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int] = (1, 1, 1),
    ceil_mode: bool = False,
):
    x = pad_same(x, kernel_size, stride, value=-float("inf"))
    return F.max_pool3d(x, kernel_size, stride, (0, 0), dilation, ceil_mode)


@register_notrace_module
class MaxPool3dSame(nn.MaxPool3d):
    """Tensorflow like 'SAME' wrapper for 3D max pooling"""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        ceil_mode: bool = False,
    ):
        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)
        dilation = to_3tuple(dilation)
        super().__init__(kernel_size, stride, (0, 0), dilation, ceil_mode)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, value=-float("inf"))
        return F.max_pool3d(
            x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode
        )


def create_pool3d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(
        padding, kernel_size, stride=stride, **kwargs
    )
    if is_dynamic:
        if pool_type == "avg":
            return AvgPool3dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == "max":
            return MaxPool3dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    else:
        if pool_type == "avg":
            return nn.AvgPool3d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == "max":
            return nn.MaxPool3d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
