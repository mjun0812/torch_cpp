/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <vector>
#include "cuda/dcnv4/dcnv4_col2im_cuda.cuh"
#include "cuda/dcnv4/dcnv4_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

at::Tensor dcnv4_cuda_forward(const at::Tensor& value,
                              const at::Tensor& p_offset,
                              const int kernel_h,
                              const int kernel_w,
                              const int stride_h,
                              const int stride_w,
                              const int pad_h,
                              const int pad_w,
                              const int dilation_h,
                              const int dilation_w,
                              const int group,
                              const int group_channels,
                              const float offset_scale,
                              const int im2col_step,
                              const int remove_center,
                              const int d_stride,
                              const int block_thread,
                              const bool softmax) {
  AT_ASSERTM(value.is_contiguous(), "input tensor has to be contiguous");
  AT_ASSERTM(value.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(p_offset.is_contiguous(), "input tensor has to be contiguous");
  AT_ASSERTM(p_offset.type().is_cuda(), "input must be a CUDA tensor");

  const int batch = value.size(0);
  const int height_in = value.size(1);
  const int width_in = value.size(2);
  const int channels = value.size(3);
  const int padded_offset_dim = p_offset.size(3);

  // tensor core requirement
  assert(padded_offset_dim % 8 == 0);

  const int height_out =
      (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int width_out =
      (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  const int im2col_step_ = std::min(batch, im2col_step);
  AT_ASSERTM(batch % im2col_step_ == 0, "batch(", batch,
             ") must divide im2col_step(", im2col_step_, ")");
  AT_ASSERTM(
      channels == (group * group_channels),
      "Input channels and group times group channels wont match: (%d vs %d).",
      channels, group * group_channels);

  auto output = at::zeros(
      {batch, height_out, width_out, group * group_channels}, value.options());

  const int batch_n = im2col_step_;
  auto output_n = output.view({batch / batch_n, batch_n, height_out, width_out,
                               group * group_channels});
  auto per_value_size = height_in * width_in * channels;
  auto per_offset_size = height_out * width_out * padded_offset_dim;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto columns = output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "dcnv4_forward_cuda", ([&] {
          dcnv4_im2col_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              p_offset.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_offset_size,
              columns.data_ptr<scalar_t>(), kernel_h, kernel_w, stride_h,
              stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
              group_channels, batch_n, height_in, width_in, height_out,
              width_out, offset_scale, remove_center, d_stride, block_thread,
              softmax, padded_offset_dim);
        }));
  }

  return output;
}

std::vector<at::Tensor> dcnv4_cuda_backward(const at::Tensor& value,
                                            const at::Tensor& p_offset,
                                            const int kernel_h,
                                            const int kernel_w,
                                            const int stride_h,
                                            const int stride_w,
                                            const int pad_h,
                                            const int pad_w,
                                            const int dilation_h,
                                            const int dilation_w,
                                            const int group,
                                            const int group_channels,
                                            const float offset_scale,
                                            const int im2col_step,
                                            const at::Tensor& grad_output,
                                            const int remove_center,
                                            const int d_stride,
                                            const int block_thread,
                                            const bool softmax) {
  AT_ASSERTM(value.is_contiguous(), "input tensor has to be contiguous");
  AT_ASSERTM(p_offset.is_contiguous(), "offset tensor has to be contiguous");
  AT_ASSERTM(grad_output.is_contiguous(),
             "grad_output tensor has to be contiguous");

  AT_ASSERTM(value.is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(p_offset.is_cuda(), "offset must be a CUDA tensor");
  AT_ASSERTM(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

  const int batch = value.size(0);
  const int height_in = value.size(1);
  const int width_in = value.size(2);
  const int channels = value.size(3);
  const int padded_offset_dim = p_offset.size(3);
  assert(padded_offset_dim % 8 == 0);

  const int height_out =
      (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int width_out =
      (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  const int im2col_step_ = std::min(batch, im2col_step);
  AT_ASSERTM(batch % im2col_step_ == 0, "batch(", batch,
             ") must divide im2col_step(", im2col_step_, ")");
  AT_ASSERTM(
      channels == (group * group_channels),
      "Input channels and group times group channels wont match: (%d vs %d).",
      channels, group * group_channels);

  auto dtype = value.dtype();
  if (dtype == at::kHalf) {
    dtype = at::kFloat;
  }

  auto grad_input = at::zeros_like(value, dtype);
  auto grad_offset = at::zeros_like(p_offset, dtype);

  const int batch_n = im2col_step_;
  auto grad_output_n = grad_output.view(
      {batch / batch_n, batch_n, height_out, width_out, group, group_channels});
  auto per_value_size = height_in * width_in * channels;
  auto per_offset_size = height_out * width_out * padded_offset_dim;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto columns = grad_output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        value.scalar_type(), "dcnv4_backward_cuda", ([&] {
          dcnv4_col2im_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              p_offset.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_offset_size,
              columns.data_ptr<scalar_t>(), kernel_h, kernel_w, stride_h,
              stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
              group_channels, batch_n, height_in, width_in, height_out,
              width_out, offset_scale, remove_center,
              grad_input.data<opmath_t>() + n * im2col_step_ * per_value_size,
              grad_offset.data<opmath_t>() + n * im2col_step_ * per_offset_size,
              d_stride, block_thread, softmax, padded_offset_dim);
        }));
  }

  if (value.dtype() == torch::kHalf) {
    return {grad_input.to(torch::kHalf), grad_offset.to(torch::kHalf)};
  } else {
    return {grad_input, grad_offset};
  }
}
