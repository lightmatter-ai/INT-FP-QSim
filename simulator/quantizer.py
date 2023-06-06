#!/usr/bin/env python
# Copyright 2023 Lightmatter, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from pytorch_quantization.tensor_quant import fake_tensor_quant


def _tensor_percentile(x, k, per_channel_axis=None):
    if per_channel_axis is not None:
        raise NotImplementedError("Per-channel kth percentile not implemented.")
    # O(N * log N)
    abs_x = torch.abs(x.flatten())
    ordered_w, _ = torch.sort(abs_x, descending=False)
    index = (k / 100) * (abs_x.shape[-1] - 1)
    if index % 1 != 0:
        index_lower = math.floor(index)
        index_higher = math.ceil(index)
        return (ordered_w[..., index_lower] + ordered_w[..., index_higher]) / 2
    else:
        return ordered_w[..., int(index)]


def _tensor_max(x, per_channel_axis=None):
    if per_channel_axis is None:
        return torch.max(torch.abs(x))
    else:
        # per-channel axis would be 1 for ConvTranspose, 0 otherwise
        axis = per_channel_axis
        reduce_axis = list(range(x.dim()))
        reduce_axis.remove(axis)
        with torch.no_grad():
            return torch.max(x.abs(), dim=axis, keepdims=True).values


def _compute_scales(x_float, calibration, percentile=99.0, per_channel_axis=None):
    if calibration == "max":
        maxval = _tensor_max(x_float, per_channel_axis)
    elif calibration == "percentile":
        maxval = _tensor_percentile(x_float, percentile, per_channel_axis)
    else:
        raise ValueError(
            "Unsupported calibration. Choose from max / percentile or specify static scale"
        )
    return maxval


def quantize_to_int(
    x_float,
    num_bits,
    calibration="max",
    percentile=99.0,
    static_scale=None,
    per_channel_axis=None,
    unsigned=False,
):
    """
    INT quantizer utilizes TensorRT functions from here:
    https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization.
    Modified to support static max calibration for input activations.
    """
    # Perform calibration
    if static_scale is not None:
        # Static quantization scale specified
        if not isinstance(static_scale, torch.Tensor):
            static_scale = torch.Tensor(static_scale)
        amax = static_scale.to(x_float.device)
    else:
        amax = _compute_scales(x_float, calibration, percentile, per_channel_axis)

    _narrow_range = True
    return fake_tensor_quant(x_float, amax, num_bits, unsigned, _narrow_range)


class FPQuantFunction(torch.autograd.Function):
    """Defining backward pass for QAT with quantize_to_fp.
    """

    @staticmethod
    def forward(ctx, x, scales):
        ctx.save_for_backward(x, scales)
        outputs = torch.round(x / scales)
        return outputs * scales

    @staticmethod
    def backward(ctx, grad_outputs):
        x, scales = ctx.saved_tensors
        zero = grad_outputs.new_zeros(1)
        grad_inputs = torch.where(x.abs() <= scales, grad_outputs, zero)
        return grad_inputs, None, None, None, None


def quantize_to_fp(
    x_float,
    mantissa_bits,
    exponent_bits,
    calibration="max",
    per_channel_axis=0,
    percentile=99.0,
    static_scale=None,
):
    """
    FP quantizer based on AIMET implementation here: https://github.com/quic/aimet.
    Modified to support flexible exponent, mantissa specifications and flexible
    scale computation using both max and percentile. Also extended to support
    static calibration.
    """
    # Convert mantissa bits to torch.Tensor
    if not isinstance(mantissa_bits, torch.Tensor):
        mantissa_bits = torch.Tensor([mantissa_bits]).to(x_float.device)

    # Perform calibration
    if static_scale is not None:
        # Static quantization scale specified
        if not isinstance(static_scale, torch.Tensor):
            static_scale = torch.Tensor(static_scale)
        maxval = static_scale.to(x_float.device)
    else:
        maxval = _compute_scales(x_float, calibration, percentile)
    maxval = maxval.clone() + 1e-10

    # Tensorized per-channel quantization: ensure that maxval has the same number of
    # dimensions as x, where the channel that is individually quantized has size C,
    # and all other channels have size 1. E.g. for a conv tensor with C output channels,
    # maxval will have shape [C, 1, 1, 1]. This allows broadcasting maxval over the
    # input tensor in steps below.
    if (
        maxval.shape
        and maxval.shape[0] != 1
        and len(maxval.shape) != len(x_float.shape)
    ):
        new_shape = [1] * len(x_float.shape)
        new_shape[per_channel_axis] = -1
        maxval = maxval.view(new_shape)

    # Math explanation of what happens here:
    # Bias is computed from maxval: $B=2^E - \log_2(M) + \log_2(2 - 2^{-M}) - 1$
    # This follows from maxval $M=(2 - 2^{-M}) \cdot 2^{2^E-1-B}$.
    bias = (
        2**exponent_bits
        - torch.log2(maxval)
        + torch.log2(2 - 2 ** (-mantissa_bits))
        - 1
    )

    # Ensure no values are greater than the maximum value represented by an 8 bit float system
    # with M mantissa and E exponent bits. torch.min/torch.max are used to allow gradients to
    # flow to maxval
    x_clipped = torch.min(torch.max(x_float, -maxval), maxval)

    # FP quantization scale is determined per-element, and is computed as
    # \log_2 s = \left\lfloor \log_2 |x_c| + B \right\rfloor - M - B
    # the addition of bias inside the floor and subtraction outside ensures that a
    # tensor scaling $\alpha \neq 1$ is correctly incorporated
    log_scales = torch.floor(torch.log2(torch.abs(x_clipped)) + bias).detach()

    # This ensures scales are never smaller than the subnormal scale
    log_scales = torch.clamp(log_scales, 1.0)

    # Second step of computing scale $s$
    scales = 2.0 ** (log_scales - mantissa_bits - bias)

    # Using the per-element scale we can quantize the clipped input tensor to the FP grid
    return FPQuantFunction.apply(x_clipped, scales)
