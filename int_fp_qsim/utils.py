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

"""Utility functions for layer replacement."""

import warnings

import torch
from packaging import version

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_greater_or_equal_than_1_10 = parsed_torch_version_base >= version.parse("1.10")


def assign_param(module, other_module, attr_name):
    param = getattr(module, attr_name)
    if param is None:
        return
    other_param = getattr(other_module, attr_name)

    module_dtype = param.data.dtype
    other_module_dtype = other_param.data.dtype

    if module_dtype != other_module_dtype:
        warnings.warn(
            f"Casting {attr_name} from {module_dtype} to "
            f"{other_module_dtype}. Training optimization may fail "
            "depending on the order in which the optimizer class is "
            "initialized."
        )
        param.data = other_param.data.to(module_dtype)
    else:
        # These are raw torch.nn.Parameters. Training with this layer
        # will work regardless of whether the optimizer was given the
        # original module's parameters or this module's parameters.
        setattr(module, attr_name, getattr(other_module, attr_name))


def copy_linears_in_attn(qattn, mod, linear_names):
    """Copies the linear layer attributes from the original attn
    module to the quantized ones, given the names of the linear
    layers.
    """
    for name in linear_names:
        qattn_layer = getattr(qattn, name)
        other_layer = getattr(mod, name)
        assign_param(qattn_layer, other_layer, "weight")
        assign_param(qattn_layer, other_layer, "bias")
        if hasattr(other_layer, "amax_input"):
            qattn_layer.amax_input = other_layer.amax_input
        if hasattr(other_layer, "amax_weight"):
            qattn_layer.amax_weight = other_layer.amax_weight


def extra_repr(module):
    """Implement extra_repr to enable argument recording.
    A custom nn.Module must have a self._extra_repr_keys tuple
    of key strings with the key being the attribute.
    """
    arglist = [f"{key}={getattr(module, key)}" for key in module._extra_repr_keys]
    return ", ".join(arglist)


class QuantMixin:
    def init_quantizer(
        self,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        amax_input,
        amax_weight,
    ):
        self.input_quantizer = input_quantizer
        self.weight_quantizer = weight_quantizer
        self.output_quantizer = output_quantizer
        self.amax_input = amax_input
        self.amax_weight = amax_weight
        self._extra_repr_keys = (
            "input_quantizer",
            "weight_quantizer",
            "output_quantizer",
            "amax_input",
            "amax_weight",
        )

    def _quantize_input(self, input):
        if self.amax_input is not None:
            return self.input_quantizer(input, static_scale=self.amax_input)
        else:
            return self.input_quantizer(input)

    def _quantize_weight(self, weight):
        if self.amax_weight is not None:
            return self.weight_quantizer(weight, static_scale=self.amax_weight)
        else:
            return self.weight_quantizer(weight)

    def _quantize_output(self, output):
        return self.output_quantizer(output)


def meshgrid(*tensors, indexing=None):
    """
    Wrapper around torch.meshgrid to avoid warning messages about the introduced `indexing` argument.
    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    if is_torch_greater_or_equal_than_1_10:
        return torch.meshgrid(*tensors, indexing=indexing)
    else:
        if indexing != "ij":
            raise ValueError(
                'torch.meshgrid only supports `indexing="ij"` for torch<1.10.'
            )
        return torch.meshgrid(*tensors)


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
    ).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = (
        duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :]
        for t in sincos
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq)
        .to(x.device)
        .float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
