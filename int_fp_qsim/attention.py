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

import collections
import math

import torch
from transformers.models.opt import modeling_opt as opt

from .layers import *
from .utils import (QuantMixin, apply_rotary_pos_emb, assign_param,
                    copy_linears_in_attn, extra_repr, fixed_pos_embedding,
                    meshgrid)

try:
    from transformers.models.bert import BertConfig
    from transformers.models.bert import modeling_bert as bert
except ImportError:
    import transformers.modeling_bert as bert
    from transformers.modeling_bert import BertConfig

import transformers.models.codegen.modeling_codegen as cg
import transformers.models.maskformer.modeling_maskformer as mf
import transformers.models.maskformer.modeling_maskformer_swin as mfswin
from transformers.models.codegen.configuration_codegen import CodeGenConfig
from transformers.models.maskformer.configuration_maskformer_swin import \
    MaskFormerSwinConfig


class OPTAttention(opt.OPTAttention, QuantMixin):
    """
    This implementation mirrors the OPTAttention implementation found in HF's transformers library.
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L120
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        dropout=0.0,
        is_decoder=False,
        bias=True,
        amax_input=None,
        amax_weight=None,
    ):
        super(OPTAttention, self).__init__(
            embed_dim, num_heads, dropout, is_decoder, bias
        )
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )
        self.k_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )
        self.v_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )
        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )
        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        hidden_states: Input data of shape: Batch x Time x Channel
        key_value_states: Optional tensor of key_value states
        past_key_value: Optional tensor of previous attention layer key value state
        attention_mask: Optional Attention mask
        layer_head_mask: Optional Layer mask
        output_attentions: Flag to indicate if output_attention should be returned
        """
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)

        # Quantize the intermediate inputs
        q_query_states = self._quantize_input(query_states)
        q_key_states = self._quantize_input(key_states)
        attn_weights = torch.bmm(q_query_states, q_key_states.transpose(1, 2))
        attn_weights = self._quantize_output(attn_weights)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attn_weights.dtype == torch.float16:
            attn_weights = torch.nn.Softmax(
                dim=-1,
            )(
                attn_weights
            ).to(torch.float16)
        else:
            attn_weights = torch.nn.Softmax(
                dim=-1,
            )(attn_weights)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None
        attn_probs = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # Quantize the intermediate inputs
        q_attn_probs = self._quantize_input(attn_probs)
        q_value_states = self._quantize_input(value_states)
        attn_output = torch.bmm(q_attn_probs, q_value_states)
        attn_output = self._quantize_output(attn_output)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights_reshaped, past_key_value)

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))

    @classmethod
    def from_float(cls, mod):
        """Create a new quantized module from a float module.
        Arguments:
            mod (Module): a float OPTAttention module
        """
        qattn = cls(
            embed_dim=mod.embed_dim,
            num_heads=mod.num_heads,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            dropout=mod.dropout,
            is_decoder=mod.is_decoder,
            bias=mod.k_proj.bias is not None,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )

        linear_names = ["out_proj", "q_proj", "v_proj", "k_proj"]
        copy_linears_in_attn(qattn, mod, linear_names)

        return qattn


class BertSelfAttention(bert.BertSelfAttention, QuantMixin):
    def __init__(
        self,
        config,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        position_embedding_type=None,
        amax_input=None,
        amax_weight=None,
    ):
        super(BertSelfAttention, self).__init__(config, position_embedding_type)
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(
            config.hidden_size,
            self.all_head_size,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
        )
        self.key = Linear(
            config.hidden_size,
            self.all_head_size,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
        )
        self.value = Linear(
            config.hidden_size,
            self.all_head_size,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
        )

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = torch.nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        q_query_layer = self._quantize_input(query_layer)
        q_key_layer = self._quantize_input(key_layer)
        attention_scores = torch.matmul(q_query_layer, q_key_layer.transpose(-1, -2))
        attention_scores = self._quantize_output(attention_scores)

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        q_attention_probs = self._quantize_input(attention_probs)
        q_value_layer = self._quantize_input(value_layer)
        context_layer = torch.matmul(q_attention_probs, q_value_layer)
        context_layer = self._quantize_output(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    @classmethod
    def from_float(cls, mod):
        """Create a new quantized module from a float module.
        Arguments:
            mod (Module): a float BertSelfAttention module
        """
        num_attention_heads = mod.num_attention_heads
        hidden_size = mod.query.weight.shape[0]
        attention_head_size = hidden_size // num_attention_heads
        attention_probs_dropout_prob = mod.dropout.p

        bert_config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_size=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )

        qattn = cls(
            bert_config,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )

        linear_names = ["query", "key", "value"]
        copy_linears_in_attn(qattn, mod, linear_names)

        if mod.training:
            qattn.train()
        else:
            qattn.eval()

        return qattn

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))


class DetrAttention(mf.DetrAttention, QuantMixin):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        dropout=0.0,
        is_decoder=False,
        bias=True,
        amax_input=None,
        amax_weight=None,
    ):
        super(DetrAttention, self).__init__(
            embed_dim, num_heads, dropout, is_decoder, bias
        )
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )
        self.v_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )
        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )
        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )

    def _shape(self, tensor, seq_len, batch_size):
        return (
            tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def with_pos_embed(self, tensor, position_embeddings):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        key_value_states=None,
        key_value_position_embeddings=None,
        output_attentions=False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(
                key_value_states, key_value_position_embeddings
            )

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(
                self.v_proj(key_value_states_original), -1, batch_size
            )
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(
                self.v_proj(hidden_states_original), -1, batch_size
            )

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(
            *proj_shape
        )
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        q_query_states = self._quantize_input(query_states)
        q_key_states = self._quantize_input(key_states)
        attn_weights = torch.bmm(q_query_states, q_key_states.transpose(1, 2))
        attn_weights = self._quantize_output(attn_weights)

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(batch_size, self.num_heads, target_len, source_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(
                batch_size * self.num_heads, target_len, source_len
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                batch_size, self.num_heads, target_len, source_len
            )
            attn_weights = attn_weights_reshaped.view(
                batch_size * self.num_heads, target_len, source_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        q_attn_probs = self._quantize_input(attn_probs)
        q_value_states = self._quantize_input(value_states)
        attn_output = torch.bmm(q_attn_probs, q_value_states)
        attn_output = self._quantize_output(attn_output)

        if attn_output.size() != (
            batch_size * self.num_heads,
            target_len,
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            batch_size, self.num_heads, target_len, self.head_dim
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    @classmethod
    def from_float(cls, mod):
        """Create a new quantized module from a float module.
        Arguments:
            mod (Module): a float DetrAttention module
        """
        qattn = cls(
            embed_dim=mod.embed_dim,
            num_heads=mod.num_heads,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            dropout=mod.dropout,
            bias=mod.k_proj.bias is not None,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )

        linear_names = ["out_proj", "q_proj", "v_proj", "k_proj"]
        copy_linears_in_attn(qattn, mod, linear_names)

        return qattn

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))


class MaskFormerSwinSelfAttention(mfswin.MaskFormerSwinSelfAttention, QuantMixin):
    def __init__(
        self,
        config,
        dim,
        num_heads,
        window_size,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        amax_input=None,
        amax_weight=None,
    ):
        super(MaskFormerSwinSelfAttention, self).__init__(
            config, dim, num_heads, window_size
        )
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )

        self.relative_position_bias_table = torch.nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads
            )
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.query = Linear(
            self.all_head_size,
            self.all_head_size,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=config.qkv_bias,
        )
        self.key = Linear(
            self.all_head_size,
            self.all_head_size,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=config.qkv_bias,
        )
        self.value = Linear(
            self.all_head_size,
            self.all_head_size,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=config.qkv_bias,
        )

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        q_query_layer = self._quantize_input(query_layer)
        q_key_layer = self._quantize_input(key_layer)
        attention_scores = torch.matmul(q_query_layer, q_key_layer.transpose(-1, -2))
        attention_scores = self._quantize_output(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MaskFormerSwinModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(
                0
            )
            attention_scores = attention_scores.view(
                -1, self.num_attention_heads, dim, dim
            )

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        q_attention_probs = self._quantize_input(attention_probs)
        q_value_layer = self._quantize_input(value_layer)
        context_layer = torch.matmul(q_attention_probs, q_value_layer)
        context_layer = self._quantize_output(context_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs

    @classmethod
    def from_float(cls, mod):
        """Create a new quantized module from a float module.
        Arguments:
            mod (Module): a float SwinSelfAttention module
        """
        config = MaskFormerSwinConfig(
            qkv_bias=mod.query.bias is not None,
            attention_probs_dropout_prob=mod.dropout.p,
        )
        qattn = cls(
            config=config,
            dim=mod.attention_head_size * mod.num_attention_heads,
            num_heads=mod.num_attention_heads,
            window_size=mod.window_size,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )

        qattn.relative_position_bias_table.data = mod.relative_position_bias_table.data
        linear_names = ["query", "key", "value"]
        copy_linears_in_attn(qattn, mod, linear_names)

        if mod.training:
            qattn.train()
        else:
            qattn.eval()

        return qattn

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))


class MultiheadAttention(torch.nn.MultiheadAttention, QuantMixin):
    def __init__(
        self,
        embed_dim,
        num_heads,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        amax_input=None,
        amax_weight=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            raise NotImplementedError("qkv must have same dims.")
        else:
            self.in_proj_weight = torch.nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = torch.nn.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.in_proj_bias = None
            self.register_parameter("in_proj_bias", None)

        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            bias=bias,
        )

        if add_bias_kv:
            self.bias_k = torch.nn.Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
            self.bias_v = torch.nn.Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):
        # Based on Pytorch MultiheadAttention implementation
        if query is not key or query is not value:
            raise ValueError("MultiheadAttention only implemented for q=k=v")
        else:
            if self.batch_first:
                query = key = value = query.transpose(1, 0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // self.num_heads
        assert (
            head_dim * self.num_heads == embed_dim
        ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        #
        # compute in-projection
        #
        q_query = self.input_quantizer(query)
        q_weight = self.weight_quantizer(self.in_proj_weight)
        qkv = torch.matmul(q_query, q_weight.t()) + self.in_proj_bias
        qkv = self.output_quantizer(qkv)
        q, k, v = qkv.chunk(3, dim=-1)

        # prep attention mask
        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = (
            k.contiguous()
            .view(k.shape[0], bsz * self.num_heads, head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(v.shape[0], bsz * self.num_heads, head_dim)
            .transpose(0, 1)
        )

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
            )
            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        if not self.training:
            dropout_p = 0.0
        #
        # (deep breath) calculate attention and out projection
        #
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        q_q_scaled = self.input_quantizer(q_scaled)
        q_k = self.input_quantizer(k)

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_q_scaled, q_k.transpose(-2, -1)
            )
        else:
            attn_output_weights = torch.bmm(q_q_scaled, q_k.transpose(-2, -1))
        attn_output_weights = self.output_quantizer(attn_output_weights)
        attn_output_weights = self.softmax(attn_output_weights)
        if dropout_p > 0.0:
            attn_output_weights = self.dropout_layer(attn_output_weights, p=dropout_p)

        q_attn_output_weights = self.input_quantizer(attn_output_weights)
        q_v = self.input_quantizer(v)
        attn_output = torch.bmm(q_attn_output_weights, q_v)
        attn_output = self.output_quantizer(attn_output)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    @classmethod
    def from_float(cls, mod):
        qattn = cls(
            embed_dim=mod.embed_dim,
            num_heads=mod.num_heads,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            dropout=mod.dropout,
            bias=True if mod.in_proj_bias is not None else False,
            add_bias_kv=True
            if mod.bias_k is not None and mod.bias_v is not None
            else False,
            add_zero_attn=mod.add_zero_attn,
            kdim=mod.kdim,
            vdim=mod.vdim,
            batch_first=mod.batch_first,
            device=mod.in_proj_weight.device,
            dtype=torch.float32,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )
        assign_param(qattn.out_proj, mod.out_proj, "weight")
        assign_param(qattn.out_proj, mod.out_proj, "bias")
        assign_param(qattn, mod, "in_proj_weight")
        assign_param(qattn, mod, "in_proj_bias")
        assign_param(qattn, mod, "bias_k")
        assign_param(qattn, mod, "bias_v")

        if mod.training:
            qattn.train()
        else:
            qattn.eval()
        return qattn

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))


class ImageBindMha(MultiheadAttention):
    def forward(self, x, attn_mask):
        return super().forward(x, x, x, attn_mask=attn_mask, need_weights=False)[0]


class CodeGenAttention(cg.CodeGenAttention, QuantMixin):
    def __init__(
        self,
        config,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        amax_input=None,
        amax_weight=None,
    ):
        super(CodeGenAttention, self).__init__(config)
        self.init_quantizer(
            input_quantizer, weight_quantizer, output_quantizer, amax_input, amax_weight
        )

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
        )

        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        ).to(torch.get_default_dtype())
        self.qkv_proj = Linear(
            self.embed_dim,
            self.embed_dim * 3,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=output_quantizer,
            bias=False,
        )

        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=output_quantizer,
            bias=False,
        )
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.causal_mask[
            :, :, key_length - query_length : key_length, :key_length
        ]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        q_query = self._quantize_input(query)
        q_key = self._quantize_input(key)
        attn_weights = torch.matmul(q_query, q_key.transpose(-1, -2))
        attn_weights = self._quantize_output(attn_weights)

        attn_weights = attn_weights / self.scale_attn
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        q_attn_weights = self._quantize_input(attn_weights)
        q_value = self._quantize_input(value)
        attn_output = torch.matmul(q_attn_weights, q_value)
        attn_output = self._quantize_output(attn_output)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        qkv = self.qkv_proj(hidden_states)
        # TODO(enijkamp): factor out number of logical TPU-v4 cores or make forward pass agnostic
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(
            query, self.num_attention_heads, self.head_dim, mp_num=mp_num
        )
        key = self._split_heads(
            key, self.num_attention_heads, self.head_dim, mp_num=mp_num
        )

        value = self._split_heads(
            value, self.num_attention_heads, self.head_dim, mp_num=mp_num
        )
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_dim
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

    @classmethod
    def from_float(cls, mod):
        config = CodeGenConfig(
            n_positions=mod.causal_mask.shape[-1],
            attn_pdrop=mod.attn_dropout.p,
            resid_pdrop=mod.resid_dropout.p,
            n_embd=mod.embed_dim,
            n_head=mod.num_attention_heads,
            rotary_dim=mod.rotary_dim,
        )

        qattn = cls(
            config,
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            amax_input=mod.amax_input,
            amax_weight=mod.amax_weight,
        )

        linear_names = ["qkv_proj", "out_proj"]
        copy_linears_in_attn(qattn, mod, linear_names)

        if mod.training:
            qattn.train()
        else:
            qattn.eval()
        return qattn

    def extra_repr(self):
        return ", ".join((super().extra_repr(), extra_repr(self)))


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, input_quantizer, weight_quantizer, output_quantizer):
        self.input_quantizer = input_quantizer
        self.weight_quantizer = weight_quantizer
        self.output_quantizer = output_quantizer

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        q_attention_probs = self.input_quantizer(attention_probs)
        q_value = self.input_quantizer(value)
        hidden_states = torch.bmm(q_attention_probs, q_value)
        hidden_states = self.output_quantizer(hidden_states)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class Attention:
    """
    Quantized Stable Diffusion Attention implementation.
    """

    def __init__(
        self,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        upcast_attention,
        scale,
        upcast_softmax,
    ):
        self.input_quantizer = input_quantizer
        self.weight_quantizer = weight_quantizer
        self.output_quantizer = output_quantizer
        self.upcast_attention = upcast_attention
        self.scale = scale
        self.upcast_softmax = upcast_softmax

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        q_baddbmm_input = self.input_quantizer(baddbmm_input)
        q_query = self.input_quantizer(query)
        attention_scores = torch.baddbmm(
            q_baddbmm_input,
            q_query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        attention_scores = self.output_quantizer(attention_scores)
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    @classmethod
    def from_float(cls, mod):
        """Create a new quantized module from a float module.
        Arguments:
            mod (Module): a float Attention module
        """

        qattn = Attention(
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
            upcast_attention=mod.upcast_attention,
            scale=mod.scale,
            upcast_softmax=mod.upcast_softmax,
        )
        mod.processor = AttnProcessor(
            input_quantizer=mod.input_quantizer,
            weight_quantizer=mod.weight_quantizer,
            output_quantizer=mod.output_quantizer,
        )
        mod.get_attention_scores = qattn.get_attention_scores
        return mod
