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

import torch

from .utils import QuantMixin, assign_param, extra_repr


# TODO: Use a ConvMixin class to refactor all conv code
class Conv1d(torch.nn.Conv1d, QuantMixin):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        amax_input=None,
        amax_weight=None,
    ):
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )

    def forward(self, input):
        # Keeps original weights and inputs in FP32
        quantized_input = self._quantize_input(input)
        quantized_weight = self._quantize_weight(self.weight)
        output = torch.nn.functional.conv1d(
            quantized_input,
            quantized_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return self._quantize_output(output)

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))

    @classmethod
    def from_float(cls, mod):
        """Create an new quantized module from a float module.
        Args:
            mod (Module): a float torch.nn.Conv1d module
        """
        # Instantiate a new Conv2d
        hasbias = mod.bias is not None

        qconv = cls(
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=hasbias,
            padding_mode=mod.padding_mode,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )
        assign_param(qconv, mod, "weight")
        assign_param(qconv, mod, "bias")
        if mod.training:
            qconv.train()
        else:
            qconv.eval()
        return qconv


class Conv2d(torch.nn.Conv2d, QuantMixin):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        amax_input=None,
        amax_weight=None,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )

    def forward(self, input):
        # Keeps original weights and inputs in FP32
        quantized_input = self._quantize_input(input)
        quantized_weight = self._quantize_weight(self.weight)
        output = torch.nn.functional.conv2d(
            quantized_input,
            quantized_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return self._quantize_output(output)

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))

    @classmethod
    def from_float(cls, mod):
        """Create an new quantized module from a float module.
        Args:
            mod (Module): a float torch.nn.Conv2d module
        """
        # Instantiate a new Conv2d
        hasbias = mod.bias is not None

        qconv = cls(
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=hasbias,
            padding_mode=mod.padding_mode,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )
        assign_param(qconv, mod, "weight")
        assign_param(qconv, mod, "bias")
        if mod.training:
            qconv.train()
        else:
            qconv.eval()
        return qconv


class Linear(torch.nn.Linear, QuantMixin):
    def __init__(
        self,
        in_features,
        out_features,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        bias=True,
        amax_input=None,
        amax_weight=None,
    ):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )

    def forward(self, input):
        # Keeps original weights and inputs in FP32
        quantized_input = self._quantize_input(input)
        quantized_weight = self._quantize_weight(self.weight)
        output = torch.nn.functional.linear(
            quantized_input, quantized_weight, self.bias
        )
        return self._quantize_output(output)

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))

    @classmethod
    def from_float(cls, mod):
        """Create a quantized module from a float module
        Arguments:
            mod (Module): a float torch.nn.Linear module
        """
        hasbias = mod.bias is not None

        qlinear = cls(
            mod.in_features,
            mod.out_features,
            mod.input_quantizer,
            mod.weight_quantizer,
            mod.output_quantizer,
            bias=hasbias,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )
        assign_param(qlinear, mod, "weight")
        assign_param(qlinear, mod, "bias")
        if mod.training:
            qlinear.train()
        else:
            qlinear.eval()
        return qlinear


class ConvTranspose2d(torch.nn.ConvTranspose2d, QuantMixin):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        amax_input=None,
        amax_weight=None,
    ):
        super(ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )

    def forward(self, input):
        # Keeps original weights and inputs in FP32
        quantized_input = self._quantize_input(input)
        quantized_weight = self._quantize_weight(self.weight)
        output = torch.nn.functional.conv_transpose2d(
            quantized_input,
            quantized_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return self._quantize_output(output)

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))

    @classmethod
    def from_float(cls, mod):
        """Create an new quantized module from a float module.
        Args:
            mod (Module): a float torch.nn.ConvTranspose2d module
        """
        # Instantiate a new Conv2d
        hasbias = mod.bias is not None

        qconvTr = cls(
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            stride=mod.stride,
            padding=mod.padding,
            output_padding=mod.output_padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=hasbias,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )
        assign_param(qconvTr, mod, "weight")
        assign_param(qconvTr, mod, "bias")
        if mod.training:
            qconvTr.train()
        else:
            qconvTr.eval()
        return qconvTr
