# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import, division, print_function

import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

try:
    from torch.amp import custom_bwd, custom_fwd
except ImportError:
    from torch.cuda.amp import custom_bwd, custom_fwd

from torch_cpp import _C


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(
    dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6
):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(
        self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias
    ):
        center_feature_scale = F.linear(
            query,
            weight=center_feature_scale_proj_weight,
            bias=center_feature_scale_proj_bias,
        ).sigmoid()
        return center_feature_scale


class DCNv3(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer="GELU",
        norm_layer="LN",
        center_feature_scale=False,
        remove_center=False,
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f"channels must be divisible by group, but got {channels} and {group}"
            )
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError("remove_center is only compatible with odd kernel size.")

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels,
            ),
            build_norm_layer(channels, norm_layer, "channels_first", "channels_last"),
            build_act_layer(act_layer),
        )
        self.offset = nn.Linear(
            channels, group * (kernel_size * kernel_size - remove_center) * 2
        )
        self.mask = nn.Linear(
            channels, group * (kernel_size * kernel_size - remove_center)
        )
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float)
            )
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float)
                .view((1,))
                .repeat(
                    group,
                )
            )
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)
        constant_(self.mask.weight.data, 0.0)
        constant_(self.mask.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)

        x = DCNv3Function.apply(
            x,
            offset,
            mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
            256,
            self.remove_center,
        )

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1,
                self.center_feature_scale_proj_weight,
                self.center_feature_scale_proj_bias,
            )
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = (
                center_feature_scale[..., None]
                .repeat(1, 1, 1, 1, self.channels // self.group)
                .flatten(-2)
            )
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


class DCNv3Function(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        offset,
        mask,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        group_channels,
        offset_scale,
        im2col_step,
        remove_center,
    ):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center

        args = [
            input,
            offset,
            mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_channels,
            offset_scale,
            ctx.im2col_step,
            remove_center,
        ]

        output = _C.dcnv3_forward(*args)
        ctx.save_for_backward(input, offset, mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors

        args = [
            input,
            offset,
            mask,
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride_h,
            ctx.stride_w,
            ctx.pad_h,
            ctx.pad_w,
            ctx.dilation_h,
            ctx.dilation_w,
            ctx.group,
            ctx.group_channels,
            ctx.offset_scale,
            grad_output.contiguous(),
            ctx.im2col_step,
            ctx.remove_center,
        ]

        grad_input, grad_offset, grad_mask = _C.dcnv3_backward(*args)

        return (
            grad_input,
            grad_offset,
            grad_mask,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def symbolic(
        g,
        input,
        offset,
        mask,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        group_channels,
        offset_scale,
        im2col_step,
        remove_center,
    ):
        """Symbolic function for mmdeploy::DCNv3.

        Returns:
            DCNv3 op for onnx.
        """
        return g.op(
            "mmdeploy::TRTDCNv3",
            input,
            offset,
            mask,
            kernel_h_i=int(kernel_h),
            kernel_w_i=int(kernel_w),
            stride_h_i=int(stride_h),
            stride_w_i=int(stride_w),
            pad_h_i=int(pad_h),
            pad_w_i=int(pad_w),
            dilation_h_i=int(dilation_h),
            dilation_w_i=int(dilation_w),
            group_i=int(group),
            group_channels_i=int(group_channels),
            offset_scale_f=float(offset_scale),
            im2col_step_i=int(im2col_step),
            remove_center=int(remove_center),
        )
