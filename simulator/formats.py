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

from functools import partial

from qtorch import FloatingPoint
from qtorch.quant import quantizer

from .quantizer import quantize_to_fp, quantize_to_int

# Based on QPytorch - https://github.com/Tiiiger/QPyTorch
E4M3 = quantizer(forward_number=FloatingPoint(exp=4, man=3), forward_rounding="nearest")
FP16 = quantizer(
    forward_number=FloatingPoint(exp=5, man=10), forward_rounding="nearest"
)

# Based on AIMET - https://github.com/quic/aimet
E1M2 = partial(quantize_to_fp, mantissa_bits=2, exponent_bits=1)
E2M1 = partial(quantize_to_fp, mantissa_bits=1, exponent_bits=2)

# Based on TensorRT - https://github.com/NVIDIA/TensorRT
INT8 = partial(quantize_to_int, num_bits=8)
INT4 = partial(quantize_to_int, num_bits=4)
