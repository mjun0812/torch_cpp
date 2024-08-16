import torch
import torch.nn.functional as F

from .dcn_v3 import DCNv3


class DCNv3_pytorch(DCNv3):
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

        x = dcnv3_core_pytorch(
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


def dcnv3_core_pytorch(
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
    # for debug and test only,
    # need to use cuda version instead

    if remove_center and (
        kernel_h % 2 == 0 or kernel_w % 2 == 0 or kernel_w != kernel_h
    ):
        raise ValueError(
            "remove_center is only compatible with square odd kernel size."
        )

    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape,
        input.device,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device
    )
    spatial_norm = (
        torch.tensor([W_in, H_in])
        .reshape(1, 1, 1, 2)
        .repeat(1, 1, 1, group * (kernel_h * kernel_w - remove_center))
        .to(input.device)
    )

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1)
    if remove_center:
        sampling_locations = remove_center_sampling_locations(
            sampling_locations, kernel_w=kernel_w, kernel_h=kernel_h
        )
    sampling_locations = sampling_locations.flatten(3, 4)
    sampling_locations = sampling_locations + offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w - remove_center
    sampling_grids = 2 * sampling_locations - 1
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = (
        input.view(N_, H_in * W_in, group * group_channels)
        .transpose(1, 2)
        .reshape(N_ * group, group_channels, H_in, W_in)
    )
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = (
        sampling_grids.view(N_, H_out * W_out, group, P_, 2)
        .transpose(1, 2)
        .flatten(0, 1)
    )
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_,
        sampling_grid_,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = (
        mask.view(N_, H_out * W_out, group, P_)
        .transpose(1, 2)
        .reshape(N_ * group, 1, H_out * W_out, P_)
    )
    output = (
        (sampling_input_ * mask).sum(-1).view(N_, group * group_channels, H_out * W_out)
    )

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()


def _get_reference_points(
    spatial_shapes,
    device,
    kernel_h,
    kernel_w,
    dilation_h,
    dilation_w,
    pad_h=0,
    pad_w=0,
    stride_h=1,
    stride_w=1,
):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device,
        ),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device,
        ),
        indexing="ij",
    )
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)

    return ref


def _generate_dilation_grids(
    spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device
):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w,
            kernel_w,
            dtype=torch.float32,
            device=device,
        ),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h,
            kernel_h,
            dtype=torch.float32,
            device=device,
        ),
        indexing="ij",
    )

    points_list.extend([x / W_, y / H_])
    grid = (
        torch.stack(points_list, -1)
        .reshape(-1, 1, 2)
        .repeat(1, group, 1)
        .permute(1, 0, 2)
    )
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid


def remove_center_sampling_locations(sampling_locations, kernel_w, kernel_h):
    idx = list(range(sampling_locations.shape[-2]))
    C = (kernel_w * kernel_h - 1) // 2
    idx = [i for i in idx if i != C and (i - C) % (C * 2 + 1) != 0]
    sampling_locations = sampling_locations[:, :, :, idx, :]
    return sampling_locations
