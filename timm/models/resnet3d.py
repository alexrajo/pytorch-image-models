"""PyTorch ResNet3D

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    LayerType,
    get_act_layer,
    get_norm_layer,
    create_classifier,
    to_ntuple,
)
from timm.layers.pool3d_same import AvgPool3dSame
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import (
    register_model,
    generate_default_cfgs,
    register_model_deprecations,
)

GROUP_NORM_NUM_GROUPS = 8

__all__ = [
    "ResNet3D",
    "BasicBlock3D",
    "Bottleneck3D",
]  # model_registry will add each entrypoint fn to this


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BasicBlock3D(nn.Module):
    """Basic residual block for ResNet3D.

    This is the standard residual block used in ResNet3D-18 and ResNet3D-34.
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.GroupNorm,
        device=None,
        dtype=None,
        kernel_size=3,
    ) -> None:
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer class.
            norm_layer: Normalization layer class.
            kernel_size: The kernel size of the convolution layers.
        """
        dd = {"device": device, "dtype": dtype}
        super().__init__()

        assert cardinality == 1, "BasicBlock3D only supports cardinality of 1"
        assert base_width == 64, "BasicBlock3D does not support changing base width"
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv3d(
            inplanes,
            first_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation,
            bias=False,
            **dd,
        )
        self.bn1 = norm_layer(
            num_groups=GROUP_NORM_NUM_GROUPS, num_channels=first_planes, **dd
        )
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv3d(
            first_planes,
            outplanes,
            kernel_size=kernel_size,
            padding=dilation,
            dilation=dilation,
            bias=False,
            **dd,
        )
        self.bn2 = norm_layer(
            num_groups=GROUP_NORM_NUM_GROUPS, num_channels=outplanes, **dd
        )

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def zero_init_last(self) -> None:
        """Initialize the last batch norm layer weights to zero for better convergence."""
        if getattr(self.bn2, "weight", None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck3D(nn.Module):
    """Bottleneck3D residual block for ResNet3D.

    This is the bottleneck block used in ResNet3D-50, ResNet3D-101, and ResNet3D-152.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.GroupNorm,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer class.
            norm_layer: Normalization layer class.
        """
        dd = {"device": device, "dtype": dtype}
        super().__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv3d(inplanes, first_planes, kernel_size=1, bias=False, **dd)
        self.bn1 = norm_layer(
            num_groups=GROUP_NORM_NUM_GROUPS, num_channels=first_planes, **dd
        )
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv3d(
            first_planes,
            width,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False,
            **dd,
        )
        self.bn2 = norm_layer(
            num_groups=GROUP_NORM_NUM_GROUPS, num_channels=width, **dd
        )
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv3d(width, outplanes, kernel_size=1, bias=False, **dd)
        self.bn3 = norm_layer(
            num_groups=GROUP_NORM_NUM_GROUPS, num_channels=outplanes, **dd
        )

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def zero_init_last(self) -> None:
        """Initialize the last batch norm layer weights to zero for better convergence."""
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    first_dilation: Optional[int] = None,
    norm_layer: Optional[Type[nn.Module]] = None,
    device=None,
    dtype=None,
) -> nn.Module:
    dd = {"device": device, "dtype": dtype}
    norm_layer = norm_layer or nn.GroupNorm
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(
        *[
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=p,
                dilation=first_dilation,
                bias=False,
                **dd,
            ),
            norm_layer(
                num_groups=GROUP_NORM_NUM_GROUPS, num_channels=out_channels, **dd
            ),
        ]
    )


def downsample_avg(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    dilation: int = 1,
    norm_layer: Optional[Type[nn.Module]] = None,
    device=None,
    dtype=None,
) -> nn.Module:
    dd = {"device": device, "dtype": dtype}
    norm_layer = norm_layer or nn.GroupNorm
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = (
            AvgPool3dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool3d
        )
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(
        *[
            pool,
            nn.Conv3d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False, **dd
            ),
            norm_layer(
                num_groups=GROUP_NORM_NUM_GROUPS, num_channels=out_channels, **dd
            ),
        ]
    )


def make_blocks(
    block_fns: Tuple[Union[Type[BasicBlock3D], Type[Bottleneck3D]], ...],
    channels: Tuple[int, ...],
    block_repeats: Tuple[int, ...],
    inplanes: int,
    reduce_first: int = 1,
    output_stride: int = 32,
    down_kernel_size: int = 1,
    avg_down: bool = False,
    device=None,
    dtype=None,
    **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    """Create ResNet3D stages with specified block configurations.

    Args:
        block_fns: Block class to use for each stage.
        channels: Number of channels for each stage.
        block_repeats: Number of blocks to repeat for each stage.
        inplanes: Number of input channels.
        reduce_first: Reduction factor for first convolution in each stage.
        output_stride: Target output stride of network.
        down_kernel_size: Kernel size for downsample layers.
        avg_down: Use average pooling for downsample.
        **kwargs: Additional arguments passed to block constructors.

    Returns:
        Tuple of stage modules list and feature info list.
    """
    dd = {"device": device, "dtype": dtype}
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (block_fn, planes, num_blocks) in enumerate(
        zip(block_fns, channels, block_repeats)
    ):
        stage_name = f"layer{stage_idx + 1}"  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get("norm_layer"),
                **dd,
            )
            downsample = (
                downsample_avg(**down_kwargs)
                if avg_down
                else downsample_conv(**down_kwargs)
            )

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            kernel_size = (1, 3, 3) if block_idx == 0 or block_idx == 1 else 3
            blocks.append(
                block_fn(
                    inplanes,
                    planes,
                    stride,
                    downsample,
                    first_dilation=prev_dilation,
                    kernel_size=kernel_size,
                    **block_kwargs,
                    **dd,
                )
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(
            dict(num_chs=inplanes, reduction=net_stride, module=stage_name)
        )

    return stages, feature_info


class ResNet3D(nn.Module):
    """ResNet3D / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet3D, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet3D impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
        self,
        block: Union[BasicBlock3D, Bottleneck3D],
        layers: Tuple[int, ...],
        num_classes: int = 1000,
        in_chans: int = 3,
        output_stride: int = 32,
        global_pool: str = "avg",
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        replace_stem_pool: bool = False,
        block_reduce_first: int = 1,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        channels: Optional[Tuple[int, ...]] = (64, 128, 256, 512),
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.GroupNorm,
        drop_rate: float = 0.0,
        zero_init_last: bool = True,
        block_args: Optional[Dict[str, Any]] = None,
        device=None,
        dtype=None,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock3D, Bottleneck3D.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck3D. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = "deep" in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if "tiered" in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv3d(
                        in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False, **dd
                    ),
                    norm_layer(
                        num_groups=GROUP_NORM_NUM_GROUPS, num_channels=stem_chs[0], **dd
                    ),
                    act_layer(inplace=True),
                    nn.Conv3d(
                        stem_chs[0],
                        stem_chs[1],
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                        **dd,
                    ),
                    norm_layer(
                        num_groups=GROUP_NORM_NUM_GROUPS, num_channels=stem_chs[1], **dd
                    ),
                    act_layer(inplace=True),
                    nn.Conv3d(
                        stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False, **dd
                    ),
                ]
            )
        else:
            self.conv1 = nn.Conv3d(
                in_chans,
                inplanes,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=3,
                bias=False,
                **dd,
            )
        self.bn1 = norm_layer(
            num_groups=GROUP_NORM_NUM_GROUPS, num_channels=inplanes, **dd
        )
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module="act1")]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(
                *filter(
                    None,
                    [
                        nn.Conv3d(
                            inplanes,
                            inplanes,
                            3,
                            stride=2,
                            padding=1,
                            bias=False,
                            **dd,
                        ),
                        norm_layer(
                            num_groups=GROUP_NORM_NUM_GROUPS,
                            num_channels=inplanes,
                            **dd,
                        ),
                        act_layer(inplace=True),
                    ],
                )
            )
        else:
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        block_fns = to_ntuple(len(channels))(block)
        stage_modules, stage_feature_info = make_blocks(
            block_fns,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **block_args,
            **dd,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = self.head_hidden_size = (
            channels[-1] * block_fns[-1].expansion
        )
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, **dd
        )

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True) -> None:
        """Initialize model weights.

        Args:
            zero_init_last: Zero-initialize the last BN in each residual branch.
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, "zero_init_last"):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, str]:
        """Create regex patterns for parameter grouping.

        Args:
            coarse: Use coarse (stage-level) or fine (block-level) grouping.

        Returns:
            Dictionary mapping group names to regex patterns.
        """
        matcher = dict(
            stem=r"^conv1|bn1|maxpool",
            blocks=r"^layer(\d+)" if coarse else r"^layer(\d+)\.(\d+)",
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing.
        """
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False) -> Union[str, nn.Module]:
        """Get the classifier module.

        Args:
            name_only: Return classifier module name instead of module.

        Returns:
            Classifier module or name.
        """
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes: int, global_pool: str = "avg") -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor.
            indices: Take last n blocks if int, all if None, select matching indices if sequence.
            norm: Apply norm layer to compatible intermediates.
            stop_early: Stop iterating over blocks when last desired intermediate hit.
            output_fmt: Shape of intermediate feature outputs.
            intermediates_only: Only return intermediate features.

        Returns:
            Features and list of intermediate features or just intermediate features.
        """
        assert output_fmt in ("NCHW",), "Output shape must be NCHW."
        intermediates = []
        take_indices, max_index = feature_take_indices(5, indices)

        # forward pass
        feat_idx = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if feat_idx in take_indices:
            intermediates.append(x)
        x = self.maxpool(x)

        layer_names = ("layer1", "layer2", "layer3", "layer4")
        if stop_early:
            layer_names = layer_names[:max_index]
        for n in layer_names:
            feat_idx += 1
            x = getattr(self, n)(
                x
            )  # won't work with torchscript, but keeps code reasonable, FML
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
        self,
        indices: Union[int, List[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ) -> List[int]:
        """Prune layers not required for specified intermediates.

        Args:
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layers.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(5, indices)
        layer_names = ("layer1", "layer2", "layer3", "layer4")
        layer_names = layer_names[max_index:]
        for n in layer_names:
            setattr(self, n, nn.Identity())
        if prune_head:
            self.reset_classifier(0, "")
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(
                [self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True
            )
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return features before final classifier layer.

        Returns:
            Output tensor.
        """
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout3d(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(variant: str, pretrained: bool = False, **kwargs) -> ResNet3D:
    """Create a ResNet3D model.

    Args:
        variant: Model variant name.
        pretrained: Load pretrained weights.
        **kwargs: Additional model arguments.

    Returns:
        ResNet3D model instance.
    """
    return build_model_with_cfg(ResNet3D, variant, pretrained, **kwargs)


def _cfg(url: str = "", **kwargs) -> Dict[str, Any]:
    """Create a default configuration for ResNet3D models."""
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (7, 7),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "conv1",
        "classifier": "fc",
        "license": "apache-2.0",
        **kwargs,
    }


default_cfgs = generate_default_cfgs({})


@register_model
def resnet3d_18(pretrained: bool = False, **kwargs) -> ResNet3D:
    """Constructs a ResNet3D-18 model."""
    model_args = dict(
        block=BasicBlock3D, layers=(2, 2, 2, 2), channels=(16, 64, 128, 128)
    )
    return _create_resnet("resnet3d_18", pretrained, **dict(model_args, **kwargs))


@register_model
def test_resnet3d(pretrained: bool = False, **kwargs) -> ResNet3D:
    """Constructs a tiny ResNet3D test model."""
    model_args = dict(
        block=[BasicBlock3D, BasicBlock3D, Bottleneck3D, BasicBlock3D],
        layers=(1, 1, 1, 1),
        stem_width=16,
        stem_type="deep",
        avg_down=True,
        channels=(32, 48, 48, 96),
    )
    return _create_resnet("test_resnet3d", pretrained, **dict(model_args, **kwargs))


register_model_deprecations(
    __name__,
    {},
)
