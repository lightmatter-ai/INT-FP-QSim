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

"""Convert floating-point modules to quantized modules.
"""
import torch
from transformers.models.opt import modeling_opt as opt

from .attention import *
from .layers import *

try:
    from transformers.models.bert import modeling_bert as bert
except ImportError:
    import transformers.modeling_bert as bert

import diffusers.models.attention_processor as sd
import transformers.models.codegen.modeling_codegen as cg
import transformers.models.graphormer.modeling_graphormer as gp
import transformers.models.maskformer.modeling_maskformer as mf
import transformers.models.maskformer.modeling_maskformer_swin as mfswin

DEFAULT_MAPPING = {
    torch.nn.Linear: Linear,
    torch.nn.Conv1d: Conv1d,
    torch.nn.Conv2d: Conv2d,
    torch.nn.ConvTranspose2d: ConvTranspose2d,
    torch.nn.MultiheadAttention: MultiheadAttention,
    opt.OPTAttention: OPTAttention,
    bert.BertSelfAttention: BertSelfAttention,
    mf.DetrAttention: DetrAttention,
    mfswin.MaskFormerSwinSelfAttention: MaskFormerSwinSelfAttention,
    cg.CodeGenAttention: CodeGenAttention,
    sd.Attention: Attention,
    gp.GraphormerMultiheadAttention: GraphormerMultiheadAttention,
}


def replace_layers(
    module, input_quantizer, weight_quantizer, output_quantizer, mapping=DEFAULT_MAPPING
):
    """Replace torch.nn.Module layers in-place with quantized variants.

    Arguments:
        module (torch.nn.Module): The model to be converted. This model is
            modified in-place.
        input_quantizer (callable): A function that takes the input
            activation and returns the simulated quantized input.
        weight_quantizer (callable): A function that takes the weight
            tensor and returns the simulated quantized weight.
        output_quantizer (callable): A function that takes the output
            activation and returns the simulated quantized output.
        mapping (dict): a mapping from nn.Module to quantized variant.
            The keys and values are both classes, e.g. the default
            mapping is {torch.nn.Linear: qnn.Linear}.
    """
    mapping = mapping or DEFAULT_MAPPING

    def swap(module, mapping, replace_func="from_float"):
        # To support more general replacement patterns, we allow the user
        # to override which function is called - from_float is the default.
        target = mapping[type(module)]
        try:
            replaced = getattr(target, replace_func)(module)
        # In some cases the module being replaced may have attributes
        # that the quantized / replacement module does not support, e.g.
        # QuantizedConv2d does not support non-default values for groups
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"{target}.from_float({module}) failed with:\n\t{e}")
        return replaced

    def convert(module, mapping):
        key = type(module)
        if key in mapping:
            # Register the quantizers
            module.input_quantizer = input_quantizer
            module.weight_quantizer = weight_quantizer
            module.output_quantizer = output_quantizer
            # Check if static calibration was done, else leave scales as None
            if not hasattr(module, "amax_input"):
                module.amax_input = None
            if not hasattr(module, "amax_weight"):
                module.amax_weight = None

            return swap(module, mapping)
        else:
            named_children = list(module.named_children())
            # The module doesn't have an entry in the recipe.
            if len(named_children):
                # The module may be a container with children. If so, one of
                # the children may have an entry in the recipe, so recurse and
                # possibly replace one or more child modules.
                reassign = {}
                for name, child in module.named_children():
                    reassign[name] = convert(child, mapping)

                for key, value in reassign.items():
                    module._modules[key] = value

            return module

    return convert(module, mapping)
