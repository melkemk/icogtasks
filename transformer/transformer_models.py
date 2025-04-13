"""
The transformer code is modified from: https://huggingface.co/transformers/v4.2.2/_modules/transformers/models/bert/modeling_bert.html#BertModel
"""

import json
import copy
import warnings
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_extended_attention_mask, pad_sentences

import predictive_coding.predictive_coding as pc
from predictive_coding.predictive_coding.var_pc_layer import VarPCLayer


def softmax_kl_energy(inputs):
    """
    KL energy function for softmax.
    To be used at the self-attention layer
    """
    x = torch.softmax(inputs['x'], dim=-1)
    mu = torch.softmax(inputs['mu'], dim=-1)
    r = x * torch.log(x / (mu + 1e-8) + 1e-8)

    return r


def softmax_kl_loss(inputs):
    """
    KL energy function for softmax.
    To be used at the cross-entropy loss
    """
    x = inputs['x']
    mu = inputs['mu']

    mu = torch.softmax(mu, dim=-1)
    r = x * torch.log(x / (mu + 1e-8) + 1e-8)

    return r


def softmax_pc_loss(inputs):
    """
    The PC energy function for softmax.
    To be used at the cross-entropy loss
    """
    x = inputs['x']
    mu = inputs['mu']
    x = x.view(-1, x.shape[-1])
    mu = mu.view(-1, mu.shape[-1])

    mu = torch.softmax(mu, dim=-1)
    energy = (x - mu) ** 2

    return energy.view(inputs['x'].shape).sum(dim=-1)


class SelfAttention(nn.Module):
    """
    https://huggingface.co/transformers/v4.2.2/_modules/transformers/models/bert/modeling_bert.html#BertModel
    """

    def __init__(self, config, layer_id=0):
        super(SelfAttention, self).__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.pc_place = config.pc_place

        if 'il' in self.config.train_mode:
            if self.pc_place == 'after_linear':
                self.pc_layers = (nn.ModuleList([
                    EnhancedPCLayer(
                        config=config,
                        text='energy_error_query_layer' + str(layer_id)
                    ),
                    EnhancedPCLayer(
                        config=config,
                        text='energy_error_key_layer' + str(layer_id)),
                    EnhancedPCLayer(
                        config=config,
                        text='energy_error_value_layer' + str(layer_id))]
                ) if not self.config.trainable_variance else
                    nn.ModuleList([
                        EnhancedVarPCLayer(
                            config=config,
                            text='energy_error_query_layer' + str(layer_id),
                            size=(config.input_length, config.hidden_size),
                            is_trainable_log_sigma=True
                        ),
                        EnhancedVarPCLayer(
                            config=config,
                            text='energy_error_key_layer' + str(layer_id),
                            size=(config.input_length, config.hidden_size),
                            is_trainable_log_sigma=True
                        ),
                        EnhancedVarPCLayer(
                            config=config,
                            text='energy_error_value_layer' + str(layer_id),
                            size=(config.input_length, config.hidden_size),
                            is_trainable_log_sigma=True
                        )])
                    )
                self.pl = nn.Parameter()  # spacer for layer identification
            elif self.pc_place in [
                    'before', 'before_fixed', 'after_act_before_softmax']:
                text = 'energy_error_self_attn_layer' + str(layer_id)
                if self.config.kl_energy:
                    self.pc_layer = EnhancedPCLayer(
                    config=config,
                    text=text,
                    energy_fn=softmax_kl_energy
                    )
                elif self.config.trainable_variance:
                    self.pc_layer = EnhancedVarPCLayer(
                        config=config,
                        text=text,
                        size=(
                            config.num_attention_heads,
                            config.input_length,
                            config.input_length),
                        is_trainable_log_sigma=True
                    )
                else:
                    self.pc_layer = EnhancedPCLayer(
                        config=config, text=text)
                self.pl = nn.Parameter()  # spacer for layer identification
            elif self.pc_place != 'after_fixed_no_softmax':
                text = 'energy_error_self_attn_layer' + str(layer_id)
                self.pc_layer = (EnhancedPCLayer(
                    config=config,
                    text=text)
                    if not config.pc_output_clipping
                    else EnhancedPCLayer(
                    config=config, clamp_interval=[0, 1],
                    text=text
                )) if not self.config.trainable_variance else \
                    EnhancedVarPCLayer(
                        config=config,
                        text=text,
                        size=(
                            config.num_attention_heads,
                            config.input_length,
                            config.input_length),
                        is_trainable_log_sigma=True
                    )
                self.pl = nn.Parameter()  # spacer for layer identification

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = "absolute"
        if self.config.do_batch_norm:
            self.batch_norm = nn.BatchNorm2d(1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attn_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if 'il' in self.config.train_mode \
                and self.pc_place == 'after_linear':
            query_layer = self.pc_layers[0](query_layer)
            key_layer = self.pc_layers[1](key_layer)
            value_layer = self.pc_layers[2](value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        if attn_mask is not None:
            attention_scores = attention_scores + attn_mask

        if self.config.do_batch_norm:
            attention_scores = self.batch_norm(attention_scores)

        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'before', 'before_fixed', 'after_act_before_softmax']:
            attention_scores = self.pc_layer.forward(attention_scores)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if 'il' in self.config.train_mode \
                and self.pc_place in ['after', 'after_fixed']:
            attention_probs = self.pc_layer.forward(attention_probs)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)
        return outputs


class SelfOutput(nn.Module):
    def __init__(self, config, layer_id=0):
        super(SelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pc_place = config.pc_place
        if 'il' in self.config.train_mode \
                and self.pc_place == 'after_linear':
            text = 'energy_error_self_output_layer' + str(layer_id)
            self.pc_layer = EnhancedPCLayer(
                config=config,
                text=text
            ) if not self.config.trainable_variance else \
                EnhancedVarPCLayer(
                    config=config,
                    text=text,
                    size=(config.input_length, config.hidden_size),
                    is_trainable_log_sigma=True
                )
            self.pl = nn.Parameter()  # spacer for layer identification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      elementwise_affine=True)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if 'il' in self.config.train_mode \
                and self.pc_place == 'after_linear':
            hidden_states = self.pc_layer.forward(hidden_states)

        hidden_states = self.dropout(hidden_states)
        if self.config.do_layer_norm:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config, layer_id=0):
        super(Attention, self).__init__()
        self.self = SelfAttention(config, layer_id)
        self.output = SelfOutput(config, layer_id)

    def forward(self, hidden_states, attn_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attn_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
                                        1:]  # add attentions if we output them
        return outputs


class IntermediateLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super(IntermediateLayer, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.pc_place = config.pc_place
        pc_layer_text = 'energy_error_intermediate_layer' + str(layer_id)
        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'before', 'before_fixed', 'after_linear']:
            self.pc_layer = (EnhancedPCLayer(
                config=config, text=pc_layer_text)
                if not config.pc_output_clipping
                else EnhancedPCLayer(
                config=config,
                clamp_interval=[-1, 1],
                text=pc_layer_text
            )) if not self.config.trainable_variance else \
                EnhancedVarPCLayer(
                    config=config,
                    text=pc_layer_text,
                    size=(config.input_length, config.hidden_size),
                    is_trainable_log_sigma=True
                )
            self.pl = nn.Parameter()  # spacer for layer identification
        self.activation = (
            nn.Tanh() if config.activation == 'tanh' else None)
        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'after', 'after_fixed', 'after_fixed_no_softmax',
                'after_act_before_softmax']:
            self.pc_layer = (EnhancedPCLayer(
                config=config, text=pc_layer_text)
                if not config.pc_output_clipping
                else EnhancedPCLayer(
                config=config,
                clamp_interval=[-1, 1],
                text=pc_layer_text
            )) if not self.config.trainable_variance else \
                EnhancedVarPCLayer(
                    config=config,
                    text=pc_layer_text,
                    size=(config.input_length, config.hidden_size),
                    is_trainable_log_sigma=True
                )
            self.pl = nn.Parameter()  # spacer for layer identification

        if self.config.do_batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.input_length)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'before', 'before_fixed', 'after_linear']:
            hidden_states = self.pc_layer(hidden_states)
        hidden_states = self.activation(hidden_states)
        if self.config.do_batch_norm:
            hidden_states = self.batch_norm(hidden_states)
        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'after', 'after_fixed', 'after_fixed_no_softmax',
                'after_act_before_softmax']:
            hidden_states = self.pc_layer(hidden_states)
        return hidden_states


class OutputLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super(OutputLayer, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.pc_place = self.config.pc_place
        if 'il' in self.config.train_mode \
                and self.pc_place == 'after_linear':
            text = 'energy_error_output_layer' + str(layer_id)
            self.pc_layer = EnhancedPCLayer(
                config=config,
                text=text
            ) if not self.config.trainable_variance else \
                EnhancedVarPCLayer(
                    config=config,
                    text=text,
                    size=(config.input_length, config.hidden_size),
                    is_trainable_log_sigma=True
                )
            self.pl = nn.Parameter()  # spacer for layer identification
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if 'il' in self.config.train_mode \
                and self.pc_place == 'after_linear':
            hidden_states = self.pc_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.config.do_layer_norm:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super(TransformerLayer, self).__init__()
        self.layer_id = layer_id
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(config, layer_id=layer_id)
        self.is_decoder = config.is_decoder
        self.intermediate = IntermediateLayer(config, layer_id=layer_id)
        self.output = OutputLayer(config, layer_id=layer_id)

    def forward(self, hidden_states, attention_mask=None,
                output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.pc_place = self.config.pc_place
        self.layer = nn.ModuleList(
            [TransformerLayer(config, layer_id=idx)
             for idx in range(config.num_hidden_layers)])
        if 'il' in self.config.train_mode \
                and self.pc_place == 'between':
            self.pc_layers = nn.ModuleList(
                [EnhancedPCLayer(
                    config=config,
                    text='layer {}'.format(idx)
                ) if not self.config.trainable_variance else \
                    EnhancedVarPCLayer(
                        config=config,
                        text='layer {}'.format(idx),
                        size=(config.input_length, config.hidden_size),
                        is_trainable_log_sigma=True
                    ) for idx in range(config.num_hidden_layers)]
            )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions
            )

            hidden_states = layer_outputs[0]
            if 'il' in self.config.train_mode \
                    and self.pc_place == 'between':
                hidden_states = self.pc_layers[i](hidden_states)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )


class PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(PredictionHeadTransform, self).__init__()
        self.config = config
        self.pc_place = config.pc_place

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'before', 'before_fixed', 'after_linear']:
            text = 'energy_error_prediction_head'
            self.pc_layer = (EnhancedPCLayer(
                config=config, text=text)
                if not config.pc_output_clipping
                else EnhancedPCLayer(
                config=config, clamp_interval=[-1, 1],
                text=text
            )) if not self.config.trainable_variance else \
                EnhancedVarPCLayer(
                    config=config,
                    text=text,
                    size=(config.input_length, config.hidden_size),
                    is_trainable_log_sigma=True
                )
            self.pl = nn.Parameter()  # spacer for layer identification

        self.activation = (
            nn.Tanh() if config['activation'] == 'tanh' else None)

        if self.config.do_batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.input_length)

        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'after', 'after_fixed', 'after_fixed_no_softmax',
                'after_act_before_softmax']:
            text = 'energy_error_prediction_head'
            self.pc_layer = (EnhancedPCLayer(
                config=config, text=text)
                if not config.pc_output_clipping
                else EnhancedPCLayer(
                config=config, clamp_interval=[-1, 1],
                text=text
            )) if not self.config.trainable_variance else \
                EnhancedVarPCLayer(
                    config=config,
                    text=text,
                    size=(config.input_length, config.hidden_size),
                    is_trainable_log_sigma=True
                )
            self.pl = nn.Parameter()  # spacer for layer identification

        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      elementwise_affine=True)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)

        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'before', 'before_fixed', 'after_linear']:
            hidden_states = self.pc_layer(hidden_states)

        hidden_states = self.activation(hidden_states)
        if self.config.do_batch_norm:
            hidden_states = self.batch_norm(hidden_states)

        if 'il' in self.config.train_mode \
                and self.pc_place in [
                'after', 'after_fixed', 'after_fixed_no_softmax',
                'after_act_before_softmax']:
            hidden_states = self.pc_layer(hidden_states)

        if self.config.do_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LMPredictionHead(nn.Module):
    def __init__(self, config):
        super(LMPredictionHead, self).__init__()
        self.config = config
        self.pc_place = config.pc_place
        self.transform = PredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)
        self.decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class TransformerEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(TransformerEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory
        # and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(
                config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = "absolute"

    def forward(
            self, input_ids=None, position_ids=None, inputs_embeds=None,
            past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                           :,
                           past_key_values_length: seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if self.config.do_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config

        self.embeddings = TransformerEmbeddings(config)
        self.encoder = TransformerEncoder(config)

        # self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None
    ):

        output_attentions = output_attentions \
            if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and "
                             "inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or "
                             "inputs_embeds")

        device = input_ids.device if input_ids is not None \
            else inputs_embeds.device

        if attention_mask is None:
            # Get the attention mask with padding
            attention_mask = torch.where(
                (input_ids == self.config.pad_token_id).clone().detach(),
                torch.zeros(input_shape).to(device),
                torch.ones(input_shape).to(device))

        # This generates the attention mask for conditional generation (CLM)
        extended_attention_mask: torch.Tensor = \
            get_extended_attention_mask(
                attention_mask, input_shape, device,
                causal_mask=(not self.config.masking_objective))

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs[0]
        return (sequence_output,) + encoder_outputs[1:]


class PHModel(nn.Module):
    def __init__(self):
        super(PHModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class TransformerTrainerModel(nn.Module):
    """
    To be inherited by all Transformer models
    Includes evaluation, running and prediction functions
    forward() should be implemented separately
    """

    def __init__(self, config):
        super(TransformerTrainerModel, self).__init__()
        self.config = config
        self.best_ppl = 10e10
        self.best_ppl_at_epoch = 0.0
        self.best_early_stop_ppl = 10e10
        self.log_interval = 10e10
        self.dev_loader = None
        self.early_stop_loader = None
        self.ph_model = PHModel()  # placeholder model
        self.best_model_counter = 0
        self.iteration_counter = 0
        self.best_iteration = 0  # for tracking plateauing
        self.in_batch_perplexities = []
        self.all_ppl_results = []
        self.h = 0
        self.h_s = []
        self.t_s = []

        # Number of weight layers formed by PCLayers
        self.num_weight_layers = None

    def send_dict_to_device(self, input_dict):
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].to(self.device)
        return input_dict

    def evaluate(self, loader, disable_tqdm=False):
        """
        Same for Backprop and Sazil,
        but Sazil doesn't use this during batch training
        (only after batch is trained)
        """
        total_loss = 0
        num_correct = 0
        num_predictions = 0
        for step, batch_dict in tqdm(enumerate(loader), total=len(loader),
                                     disable=disable_tqdm):
            with torch.no_grad():
                batch_dict = self.send_dict_to_device(batch_dict)
                input_batch = batch_dict['input_tensor']
                y = batch_dict['target_tensor'].view(-1)
                if self.config['loss_fn'] == 'mse' and \
                        not self.target_in_emb_space:
                    # Construct 1-hot version of y
                    y_onehot = torch.zeros(
                        (y.size(0), y.size(1), self.vocab_size)).to(
                        self.device)
                    y_onehot.scatter_(2, y.unsqueeze(dim=2), 1)
                    y_onehot = y_onehot.permute(0, 2, 1)
                elif self.target_in_emb_space:
                    y_emb = self.embedder(y).permute(0, 2, 1)

                output = self(input_batch)[0].view(-1, self.config.vocab_size)
                eos_id = self.tokenizer.vocab_dict['</s>']
                if self.config['loss_fn'] == 'mse' and \
                        not self.target_in_emb_space:
                    loss = self.criterion(output, y_onehot)
                elif self.target_in_emb_space:
                    loss = self.criterion(output, y_emb)
                else:
                    # For CrossEntropyLoss ignores unwanted tokens
                    # for perplexity. We replace them with 0 to ignore.
                    criterion = nn.CrossEntropyLoss(ignore_index=0)
                    # Ignore the <\s> and <pad> tokens for
                    # loss, ppl and acc evaluations
                    ignored_ids = [
                        self.tokenizer.vocab_dict['<s>'],
                        self.tokenizer.vocab_dict['</s>'],
                        self.tokenizer.vocab_dict['<pad>']
                    ]
                    # Also ignore <mask> token for masked LM objective
                    if self.config.masking_objective:
                        ignored_ids.append(self.tokenizer.vocab_dict['<mask>'])
                    y = (torch.logical_not(
                        sum(y == idx for idx in ignored_ids).bool())) * y
                    loss = criterion(output, y)

                loss = torch.mean(loss)
                total_loss += loss.item()
                if self.target_in_emb_space:
                    # Construct all possible outputs
                    b_size = output.size(0)
                    seq_len = output.size(2)
                    all_possible_token_emb_s = self.embedder(
                        torch.tensor(list(range(self.vocab_size))
                                     ).to(self.device))
                    # print('Start sizes:', output.size(),
                    #       all_possible_token_emb_s.size())
                    all_possible_token_emb_s = torch.cat(
                        b_size * [
                            all_possible_token_emb_s.unsqueeze(dim=0)],
                        dim=0
                    )
                    all_possible_token_emb_s = torch.cat(
                        seq_len * [
                            all_possible_token_emb_s.unsqueeze(dim=3)],
                        dim=3
                    )
                    # print('II sizes:', output.size(),
                    #       all_possible_token_emb_s.size())
                    duplicated_output = torch.cat(
                        self.vocab_size * [
                            output.unsqueeze(dim=1)],
                        dim=1
                    )
                    # print('III sizes:', duplicated_output.size(),
                    #       all_possible_token_emb_s.size())
                    # Compute cosine similarity distance between them and
                    # the output
                    distances_to_emb_s = nn.functional.cosine_similarity(
                        duplicated_output, all_possible_token_emb_s, dim=2)
                    # print('Final shape:', distances_to_emb_s.size())
                    # Take argmax across the vocab_size dimension
                    # to get the prediction (all in pytorch)
                    predictions = torch.argmax(distances_to_emb_s, dim=1)
                    # print('Prediction shape:', predictions.size())
                    # print('y shape:', y.size())
                else:
                    predictions = output.argmax(dim=1)

                # Compare y and predictions:
                # Ignore the <pad> labels
                pad_mask = torch.where(
                    y == self.tokenizer.pad_id,
                    torch.zeros(y.size()).to(self.device) == 1,
                    torch.ones(y.size()).to(self.device) == 1)
                num_predictions += torch.sum((
                                                     y != eos_id * torch.ones(
                                                 y.size()).to(self.device)
                                             ) * pad_mask).item()
                num_correct += torch.sum((y == predictions) * (
                        y != eos_id * torch.ones(
                    y.size()).to(self.device)) * pad_mask).item()
                # num_predictions += predictions.numel()
                # num_correct += torch.eq(predictions, y).sum().item()

        mean_loss = total_loss / len(loader)
        result_dict = {
            'loss': mean_loss,
            'perplexity': -100,
            'accuracy': num_correct / num_predictions
        }
        if self.config['loss_fn'] == 'cross-entropy':
            result_dict['perplexity'] = np.exp(mean_loss)
        return result_dict

    def run_epoch(self, epoch, train_mode, train_loader, valid_loader,
                  tb_writer, save_dir, log_interval=200, save_interval=10e+10,
                  grad_accum_steps=1, early_stop_interval=None):
        """
        Sazil and Backprop training in one
        """
        self.log_interval = log_interval
        buffer_loss = 0
        self.train()

        def callback_before_t_fn(t, pc_trainer):
            """
            This function is executed right after backward() and
            before _optimizer_x.step().
            """
            config = pc_trainer.get_model().config
            big_t = config['T']

            param_groups = pc_trainer.get_optimizer_p().param_groups
            if hasattr(pc_trainer.get_optimizer_x(), 'param_groups'):
                x_param_groups = pc_trainer.get_optimizer_x().param_groups
            else:
                x_param_groups = None

            num_weight_layers = pc_trainer.get_model(
            ).num_weight_layers

            def set_e_lr(trainer, lr):
                # Note: this implementation is slow
                trainer._optimizer_x = torch.optim.SGD(
                    [{'params': [
                        param for name, param in
                        pc_trainer.get_model().named_parameters()
                        if name.endswith('._x')]}], lr=lr)

            if config.train_mode == 'zil_sazil':
                # Enable only the current-time-step-layer-weights
                # to be updated
                if t < num_weight_layers:
                    for t1 in range(num_weight_layers):
                        # Update layer weights going from
                        # top layer down, for the next time step
                        if t1 == num_weight_layers - 1 - t:
                            param_groups[t1]['lr'] = config.weight_lr
                        else:
                            param_groups[t1]['lr'] = 0.0
                elif t == num_weight_layers:
                    for t1 in range(num_weight_layers):
                        param_groups[t1]['lr'] = config.weight_lr

            elif config.train_mode == 'il_sazil':
                num_weight_layers = pc_trainer.get_model(
                ).num_weight_layers
                if t < num_weight_layers - 1:
                    set_e_lr(pc_trainer, 1.0)
                elif t < big_t:
                    set_e_lr(pc_trainer, config.energy_lr)

            if config.quick_error_propagate:
                # Update energy 1 step earlier, to
                # affect the next weight update
                if t < num_weight_layers - 2:
                    set_e_lr(pc_trainer, 1.0)
                else:
                    set_e_lr(pc_trainer, config.energy_lr)

            if config.debug:
                print('Step', t)
                print('p_lr-s:', [
                    param_group['lr']
                    for param_group in param_groups])
                for x_param_group in x_param_groups:
                    print('x_lr:', x_param_group['lr'])
                print('--------')
                print('Gradients:')
                for name, param in pc_trainer.get_model().named_parameters():
                    if param.grad is None:
                        print(name, 'None')
                    else:
                        print(name, '\n', param.grad.norm().cpu().item(),
                              '\n-----')
                print('--------')

        def callback_after_t_fn(t, pc_trainer, targets):
            """
            Do clamping inside (if activated)
            Do ZIL inside (if activated)
            Do evaluation inside (if activated)
            Do early stopping inside (if activated)
            :param t: current inference step
            :param pc_trainer: pc.PCTrainer instance
            :return: do operations inside, no return!
            """
            config = pc_trainer.get_model().config

            # This is used instead of the loss function for PC.
            # (?because it's bad practice to pass loss_fn to the trainer?)
            output_pc_layer = list(pc_trainer.get_model_pc_layers())[-1]
            output_pc_layer._x.data.view(-1, self.config.vocab_size).copy_(
                torch.nn.functional.one_hot(
                    targets, num_classes=config.vocab_size).float())
            
            big_t = config['T']

            for module in pc_trainer.get_model().modules():
                if str(module) == "EnhancedPCLayer()":
                    if module.clamp_interval is not None:
                        min_val, max_val = tuple(module.clamp_interval)
                        # print(min_val, max_val)
                        module._x = nn.Parameter(module._x.clamp(
                            min_val, max_val))

            if pc_trainer.get_model().log_interval < big_t and t == 0:
                pc_trainer.get_model().in_batch_perplexities = []

            # Evaluate (dev) if log interval is small enough,
            # or if early stopping for t is enabled
            if config.early_stop_t:
                pc_trainer.get_model().eval()
                dev_result = pc_trainer.get_model().evaluate(
                    pc_trainer.get_model().early_stop_loader.data_loader,
                    disable_tqdm=True)

                # Print results
                for k, v in dev_result.items():
                    dev_result[k] = round(v, 3)
                if config.print_in_batch:
                    print('In-batch dev result at t={}/{}:'.format(
                        t, big_t), dev_result)
                pc_trainer.get_model().train()

                if dev_result['perplexity'] <= pc_trainer.ppl:
                    pc_trainer.get_model().best_iteration = copy.deepcopy(
                        pc_trainer.get_model().iteration_counter)
                    pc_trainer.ppl = dev_result['perplexity']
                    pc_trainer.get_model().best_model_counter += 1
                    if (pc_trainer.get_model().best_model_counter
                        % config.model_revert_interval) == 0 \
                            or pc_trainer.get_model(
                    ).best_model_counter == 1:
                        # print(pc_trainer.ppl)
                        old_load_obj = pc_trainer.get_model(
                        ).early_stop_loader
                        stop_dataset = copy.deepcopy(old_load_obj.dataset)
                        stop_dataset.random_sample(sample_size=100)
                        new_loader = DataLoader(
                            stop_dataset,
                            batch_size=old_load_obj.batch_size,
                            collate_fn=old_load_obj.collate_fn)
                        pc_trainer.get_model().eval()
                        dev_result = pc_trainer.get_model().evaluate(
                            new_loader, disable_tqdm=True)
                        pc_trainer.get_model().train()
                        pc_trainer.ppl = dev_result['perplexity']
                        pc_trainer.get_model(
                        ).early_stop_loader.data_loader \
                            = new_loader
                        # print('new:', pc_trainer.ppl)

                    # Save the best model, x-es and optimizer
                    # checkpoints by accessing them through
                    # pc_trainer
                    pc_trainer.best_checkpoints[0] = copy.deepcopy(
                        pc_trainer.best_checkpoints[1]
                    )
                    pc_trainer.best_checkpoints[1] = copy.deepcopy([
                        pc_trainer.get_model().state_dict(),
                        # [x.detach() for x in pc_trainer.get_model()_xs()],
                        pc_trainer.get_optimizer_x().state_dict(),
                        pc_trainer.get_optimizer_p().state_dict()
                    ])

                elif t == config.min_t - 1 and t > 0:
                    # Restore the best model (at the final t)
                    #  by injecting the optimizer checkpoint, the model
                    #  checkpoint and the x-es into pc_trainer
                    if ((pc_trainer.get_model().iteration_counter
                         - pc_trainer.get_model().best_iteration)
                            == (pc_trainer.get_model().config.min_t
                                * 10)):
                        print('Trying to exit plateau... Reverting to'
                              'old checkpoint...')
                        model_state_dict, optimizer_x_state_dict, \
                        optimizer_p_state_dict = \
                            pc_trainer.best_checkpoints[0]
                    else:
                        model_state_dict, optimizer_x_state_dict, \
                        optimizer_p_state_dict = \
                            pc_trainer.best_checkpoints[1]

                    pc_trainer.get_model().load_state_dict(model_state_dict)
                    # for layer_id, pc_layer in enumerate(
                    #         pc_trainer.get_model()_pc_layers()):
                    #     pc_layer._x = trainer_xs[layer_id]
                    pc_trainer.get_optimizer_x().load_state_dict(
                        optimizer_x_state_dict)
                    pc_trainer.get_optimizer_p().load_state_dict(
                        optimizer_p_state_dict)

                    pc_trainer.get_model().eval()
                    restored_result = pc_trainer.get_model().evaluate(
                        pc_trainer.get_model(
                        ).early_stop_loader.data_loader,
                        disable_tqdm=True)
                    for k, v in restored_result.items():
                        restored_result[k] = round(v, 3)
                    if config.print_in_batch:
                        print('Restored model...:')
                        print('Restored dev result:', restored_result)
                    pc_trainer.get_model().train()

                # Hook for early stopping
                if t + 1 >= config.min_t \
                        and dev_result['perplexity'] > pc_trainer.ppl:
                    # ToDo: should I restore best model here?
                    return 'break'

            elif (pc_trainer.get_model().log_interval < big_t
                  and t % pc_trainer.get_model().log_interval == 0):
                pc_trainer.get_model().eval()
                dev_result = pc_trainer.get_model().evaluate(
                    pc_trainer.get_model().dev_loader, disable_tqdm=True)
                ppl = dev_result['perplexity']
                pc_trainer.get_model().in_batch_perplexities += [ppl]

                # Print results
                for k, v in dev_result.items():
                    dev_result[k] = round(v, 3)
                if config.print_in_batch:
                    print('In-batch dev result at t={}/{}:'.format(
                        t, big_t), dev_result)
                pc_trainer.get_model().train()

            pc_trainer.get_model().iteration_counter += 1
            return None

        for step, batch_dict in tqdm(enumerate(train_loader),
                                     total=len(train_loader)):
            batch_dict = self.send_dict_to_device(batch_dict)
            input_batch = batch_dict['input_tensor']
            y = batch_dict['target_tensor'].view(-1)
            if self.config['loss_fn'] == 'mse' and \
                    not self.target_in_emb_space:
                # Construct 1-hot version of y
                y_onehot = torch.zeros(
                    (y.size(0), y.size(1), self.vocab_size)).to(self.device)
                y_onehot.scatter_(2, y.unsqueeze(dim=2), 1)
                y = y_onehot.permute(0, 2, 1)
            elif self.target_in_emb_space:
                # We cut gradient so that we only propagate to the input,
                # not to the target
                with torch.no_grad():
                    y = self.embedder(y).permute(0, 2, 1)

            if self.debug:
                print('in:', self.tokenizer.convert_ids_to_tokens(
                    input_batch[0].tolist()))
                output = self.forward(input_batch[0].unsqueeze(dim=0))[0]
                predictions = output.argmax(dim=-1)
                print('out:', self.tokenizer.convert_ids_to_tokens(
                    predictions[0].tolist()))
                print('target:', self.tokenizer.convert_ids_to_tokens(
                    batch_dict['target_tensor'][0].tolist()
                ), '\n')

            if train_mode == 'backprop':
                logits = self.forward(batch_dict['input_tensor'])[0].view(
                    -1, self.config.vocab_size)
                loss = self.criterion(logits, y)
                buffer_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            elif 'il' in train_mode:
                # main_loss_fn = self.criterion

                # def loss_fn(out, targets):
                #     out = out[0].view(-1, self.config.vocab_size)
                #     return torch.mean(main_loss_fn(out, targets))

                # Do this one step after x-es are initialized in PCLayer
                if step == 1 and epoch == 1 and self.config.unified_optimizer:
                    # Externally modify the optimizer of PCTrainer
                    # to optimizer jointly p and x
                    self.model_trainer._optimizer_p = torch.optim.AdamW(
                        self.ph_model.parameters())
                    self.model_trainer._optimizer_x = torch.optim.AdamW(
                        [{'params': self.model_trainer.model_parameters()},
                         {'params': self.model_trainer.model_xs(),
                          'lr': self.config.energy_lr}],
                        lr=self.config.weight_lr)

                callback_after_b_kwargs = {
                    'pc_trainer': self.model_trainer
                }
                callback_after_t_kwargs = {
                    'pc_trainer': self.model_trainer, 'targets': y
                }

                self.train()
                batch_results = self.model_trainer.train_on_batch(
                    input_batch, #loss_fn=loss_fn,
                    #loss_fn_kwargs={'targets': y},
                    is_log_progress=self.config.show_sazil_training,
                    callback_after_t_kwargs=callback_after_t_kwargs,
                    callback_after_t=callback_after_t_fn,
                    callback_after_backward_kwargs=callback_after_b_kwargs,
                    callback_after_backward=callback_before_t_fn,
                    is_clear_energy_after_use=False)

                # Plot the results for each batch on top of each other
                self.model_trainer.get_model(
                ).all_ppl_results += copy.deepcopy(
                    self.model_trainer.get_model().in_batch_perplexities
                )
                self.t_s += list(range(self.config.T))
                self.h_s += [copy.deepcopy(self.h)] * self.config.T
                dataframe = {
                    'h': self.h_s,
                    't': self.t_s,
                    'value': self.model_trainer.get_model().all_ppl_results}
                adjusted_step = step + len(train_loader) * (epoch - 1)
                if adjusted_step == self.config.plot_progress_at:
                    sns.relplot(
                        data=pd.DataFrame(dataframe),
                        x="t",
                        y="value",
                        hue="h",
                        palette="rocket_r",
                        kind='line',
                        facet_kws={
                            "sharey": False,
                            "legend_out": False,
                        },
                    ).set(yscale='log')
                    plt.draw()
                    plt.pause(0.001)
                    input("small pause; press any key to continue")
                    plt.close(fig=None)
                self.h += 1

                if self.config.reset_x_to_mean:
                    self.model_trainer.reset_x(input_batch.size(0))

                for idx, (loss_value, error_value) in enumerate(zip(
                        batch_results['loss'],
                        batch_results['energy'])):
                    log_step = step + len(train_loader) * (epoch - 1)
                    tb_writer.add_scalar(
                        'train loss', loss_value,
                        self.config['T'] * (log_step + 1) + idx)
                    tb_writer.add_scalar(
                        'energy error', error_value,
                        self.config['T'] * (log_step + 1) + idx)
                    tb_writer.add_scalar(
                        'train_loss_plus_error', loss_value + error_value,
                        self.config['T'] * (log_step + 1) + idx)

                if log_interval < self.config['T']:
                    print('Batch training results:')
                    print('Train losses:', [
                        round(_, 3) for _ in batch_results['loss']])
                    print('Total energy errors:',
                          [round(_, 5) for _ in batch_results['energy']])
                    print('Total loss + energy error:',
                          [round(_, 5) for _ in batch_results['overall']])
            else:
                raise NotImplementedError

            # Adjust the log interval to account for T
            if 'il' in train_mode:
                if log_interval < self.config['T']:
                    log_interval_adjusted = 1
                else:
                    log_interval_adjusted = log_interval // self.config['T']
            else:
                log_interval_adjusted = log_interval
            # Evaluate the model while training
            if (step + 1) % log_interval_adjusted == 0:
                self.eval()
                log_step = step + len(train_loader) * (epoch - 1)
                dev_results = self.evaluate(valid_loader)
                for result_key in dev_results.keys():
                    tb_writer.add_scalar(
                        result_key, dev_results[result_key], log_step + 1)
                print('Dev results ({}):'.format(
                    (step + 1) // log_interval_adjusted), dev_results)

                if self.config.early_x_update:
                    early_stop_results = self.evaluate(
                        self.early_stop_loader.data_loader,
                        disable_tqdm=True)
                    if early_stop_results['perplexity'] < \
                            self.best_early_stop_ppl:
                        self.best_early_stop_ppl = early_stop_results[
                            'perplexity']
                    elif 'il' in train_mode:
                        # increase the number of x-updates before x-p updates
                        update_p_at = self.model_trainer._update_p_at
                        update_p_at.sort()
                        if len(update_p_at) > 1:
                            self.model_trainer._update_p_at = update_p_at[1:]
                            print('Updating p {} times'.format(
                                len(update_p_at) - 1
                            ))

                epochs_training_progress = epoch + step / len(train_loader)
                if dev_results['perplexity'] < self.best_ppl:
                    self.best_ppl = dev_results['perplexity']
                    self.best_ppl_at_epoch = epochs_training_progress
                    dev_results['epochs_trained'] = self.best_ppl_at_epoch
                    # Save best model and result
                    print('Saving best model and result: {} ppl...'.format(
                        self.best_ppl))
                    torch.save(self.state_dict(),
                               '{}/best_model.pth'.format(save_dir))
                    with open('{}/best_dev_result.json'.format(save_dir),
                              'w') as fp:
                        json.dump(dev_results, fp)
                else:
                    # Do early stopping if results don't improve
                    # over early_stop_interval epochs.
                    if early_stop_interval is not None:
                        if (
                            epochs_training_progress - self.best_ppl_at_epoch
                                ) >= early_stop_interval:
                            break

                if train_mode == 'backprop':
                    buffer_av_loss = buffer_loss / log_interval_adjusted
                    print('Train loss:', buffer_av_loss)
                    tb_writer.add_scalar('Train loss', buffer_av_loss,
                                         log_step + 1)
                    buffer_loss = 0

                self.train()

            if (step + 1) % save_interval == 0:
                print('Saving model...')
                torch.save(self.state_dict(),
                           '{}/model.pth'.format(save_dir))

        print('Saving final model...')
        torch.save(self.state_dict(),
                   '{}/final_model.pth'.format(save_dir))
        print('Evaluating final model...')
        self.eval()
        result_dict = self.evaluate(valid_loader)
        print('Final dev result:', result_dict)
        with open('{}/dev_results.json'.format(save_dir), 'w') as fp:
            json.dump(result_dict, fp)

    def predict(self, in_sentence, top_k=5, silent=False):
        # This is a placeholder_token vocabulary token we use to
        # input a <mask> token
        placeholder_token = '/'
        assert placeholder_token in self.tokenizer.vocab
        if not self.config.masking_objective:
            assert placeholder_token not in in_sentence
        in_sentence = in_sentence.replace('<mask>', placeholder_token)

        sent_tokens = self.tokenizer.tokenize(in_sentence)
        if len(sent_tokens) > self.config.input_length - 1:
            excess_len = len(sent_tokens) - self.config.input_length + 1
            print('Sentence too long! Cutting context to max length.')
            sent_tokens = sent_tokens[excess_len:]
            print('New context:', sent_tokens)

        # Substitute back the placeholder token to <mask>
        sent_tokens = ['<mask>' if placeholder_token in tok else tok
                       for tok in sent_tokens]

        # Convert tokens to ids
        sent_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        sos_id = self.tokenizer.vocab_dict['<s>']
        eos_id = self.tokenizer.vocab_dict['</s>']
        if self.config.masking_objective:
            input_ids = [sos_id] + sent_ids + [eos_id]
        else:
            input_ids = [sos_id] + sent_ids
        if not silent:
            print(input_ids)
            print(self.tokenizer.convert_ids_to_tokens(input_ids))

        self.eval()
        with torch.no_grad():
            input_tensor = pad_sentences([input_ids], self.config.input_length)
            # print(input_tensor)
            input_tensor = input_tensor.to(self.device)
            logits = self.forward(input_tensor)[0].view(
                -1, self.config.vocab_size)
            all_probs = torch.nn.functional.softmax(logits, dim=1)
            prediction_probs = torch.topk(all_probs, top_k, dim=1)[0].tolist()
            prediction_probs = np.around(
                np.array(prediction_probs), 3).tolist()
            prediction_ids = torch.topk(all_probs, top_k, dim=1)[1].tolist()
            if not silent:
                print('Top {} generations:'.format(top_k))
                for prediction in prediction_ids:
                    prediction = self.tokenizer.convert_ids_to_tokens(prediction)
                    print(prediction)
            # print(end_id)
            if self.config.masking_objective:
                mask_id = self.tokenizer.vocab_dict['<mask>']
                mask_positions = [idx for idx in range(len(input_ids))
                                  if input_ids[idx] == mask_id]
                if not silent:
                    print('Mask positions:', mask_positions)
                mask_predictions = []
                for mask_pos in mask_positions:
                    # For each mask position get the predictions&probabilities
                    prediction = self.tokenizer.convert_ids_to_tokens(
                        prediction_ids[mask_pos])
                    probabilities = prediction_probs[mask_pos]
                    mask_predictions.append((prediction, probabilities))
                return mask_predictions
            else:
                pad_id = self.tokenizer.pad_id
                if pad_id in input_ids:
                    end_id = input_ids.index(pad_id)
                else:
                    end_id = len(input_ids)
                prediction = prediction_ids[end_id - 1]
                prediction_probs_list = prediction_probs[end_id - 1]
                prediction = self.tokenizer.convert_ids_to_tokens(prediction)
                return prediction, prediction_probs_list


class TransformerLM(TransformerTrainerModel):
    def __init__(self, config):
        super(TransformerLM, self).__init__(config)
        self.config = config
        self.transformer_model = TransformerModel(config)
        self.cls = LMPredictionHead(config)
        if 'backprop' not in self.config.train_mode:
            if self.config.kl_energy:
                self.pc_layer = EnhancedPCLayer(
                        config=config,
                        text="pc_output",
                        energy_fn=softmax_kl_loss
                )
            else:
                self.pc_layer = EnhancedPCLayer(
                    config=config,
                    text="pc_output",
                    energy_fn=softmax_pc_loss
                )
        self.target_in_emb_space = False
        if self.config.do_batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.input_length)

        # self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None
    ):
        outputs = self.transformer_model.forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls.forward(sequence_output)
        if self.config.do_batch_norm:
            prediction_scores = self.batch_norm(prediction_scores)
        if 'backprop' not in self.config.train_mode:
            prediction_scores = self.pc_layer.forward(prediction_scores)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores
            # and input ids by one
            shifted_prediction_scores = prediction_scores[
                                        :, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(
                -1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((lm_loss,) + output) if lm_loss is not None else output


class CustomZeroLayer(nn.Module):
    def __init__(self, vocab_size, hidden_size, device):
        super(CustomZeroLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, input_tensor):
        return torch.zeros([
            input_tensor.size(0), input_tensor.size(1), self.hidden_size
        ]).to(self.device)


class EnhancedPCLayer(pc.PCLayer):
    """
    Adapted from https://github.com/YuhangSong/Prospective-Configuration/tree/main/predictive_coding/predictive_coding/pc_layer.py
    """
    def __init__(self, config, clamp_interval=None, text=None, **kwargs):
        super(EnhancedPCLayer, self).__init__(**kwargs)
        self.text = text
        self.transformer_config = config
        self.sazil_shuffled = (config.train_mode == 'sazil_shuffled')
        self.clamp_interval = clamp_interval
        self.step = 1

    def forward(
            self,
            mu: torch.Tensor,
            energy_fn_additional_inputs: dict = {},
    ) -> torch.Tensor:
        """Forward.
        Args:
            mu: The input.
            energy_fn_additional_inputs:
                Additional inputs to be passed to energy_fn.
        Returns:
            The output.
        """

        if self.sazil_shuffled:
            shuffled_ids = torch.randperm(n=mu.size(0))
            mu = mu[shuffled_ids]

        # sanitize args
        assert isinstance(mu, torch.Tensor)
        assert isinstance(energy_fn_additional_inputs, dict)

        if self.training:

            # detect cases where sample_x is necessary
            if not self._is_sample_x:

                # case: no initialization
                if self._x is None:

                    warnings.warn(
                        (
                            "The <self._x> has not been initialized yet, run with <pc_layer.set_is_sample_x(True)> first. We will do it for you."
                        ),
                        category=RuntimeWarning
                    )
                    self._is_sample_x = True

                # case: device changed
                elif mu.device != self._x.device:
                    warnings.warn(
                        (
                            "The device of <self._x> is not consistent with that of <mu>, run with <pc_layer.set_is_sample_x(True)> first. We will do it for you."
                        ),
                        category=RuntimeWarning
                    )
                    self._is_sample_x = True

                # case: size changed
                elif mu.size() != self._x.size():
                    warnings.warn(
                        (
                            "You have changed the shape of this layer, you should do <pc_layer.set_is_sample_x(True) when changing the shape of this layer. We will do it for you.\n"
                            "This should have been taken care of by <pc_trainer> unless you have set <is_sample_x_at_epoch_start=False> when calling <pc_trainer.train_on_batch()>,\n"
                            "in which case you should be responsible for making sure the batch size stays still."
                        ),
                        category=RuntimeWarning
                    )
                    self._is_sample_x = True

            # sample_x
            if self._is_sample_x:

                x_data = self._sample_x_fn(
                    {
                        'mu': mu,
                        'x': self._x,
                    }
                )

                self._x = nn.Parameter(x_data.to(mu.device), True)

                # _is_sample_x only takes effect for one pass
                self._is_sample_x = False

            x = self._x

            if self._S is not None:

                # this only works for linear networks
                assert mu.dim() == 2
                assert x.dim() == 2

                size_mu = mu.size(1)
                size_x = x.size(1)

                # self._S.size() = [size_mu, size_x]
                assert self._S.size(0) == size_mu
                assert self._S.size(1) == size_x

                # expand mu
                mu = mu.unsqueeze(
                    2
                ).expand(-1, -1, size_x)

                # expand x
                x = x.unsqueeze(
                    1
                ).expand(-1, size_mu, -1)

            energy_fn_inputs = {
                'mu': mu,
                'x': x,
            }
            energy_fn_inputs.update(energy_fn_additional_inputs)

            energy = self._energy_fn(energy_fn_inputs)

            if self._S is not None:
                energy = energy * self._S.unsqueeze(0)

            elif self._M is not None:

                energy = energy * self._M.unsqueeze(0)

            # if self.is_keep_energy_per_datapoint:
            # energy, keep the batch dim, other dimensions are reduced to a single dimension
            self._energy_per_datapoint = energy.sum(
                dim=list(
                    range(
                        energy.dim()
                    )
                )[1:],
                keepdim=False,
            ).unsqueeze(1)
            # [batch_size, 1]

            self._energy = energy.sum()

            if self.is_holding_error:
                self.error = (self._x.data - mu).detach().clone()

            """
            if self.text is not None:
                print('\n', self.text)
                print('Mean/std energy error:',
                      '{:0.3e}'.format(torch.mean(self._energy_error).item()),
                      '{:0.3e}'.format(torch.std(self._energy_error).item())
                      )
            """
            text = self.text if self.text is not None else ''
            if self.step % self.transformer_config.T == 0:
                self.transformer_config.tb_writer.add_scalar(
                    text + '.mean', torch.mean(energy).item(),
                    self.step + 1)
                self.transformer_config.tb_writer.add_scalar(
                    text + '.std', torch.std(energy).item(),
                    self.step + 1)
            self.step += 1

            return self._x

        else:

            return mu


class EnhancedVarPCLayer(pc.PCLayer):
    """``VarPCLayer``.
        ``VarPCLayer`` specifies a ``PCLayer`` to be preditive coding energy with variance (log_sigma), which can be specified to be trainable or not.
         Adapted from https://github.com/YuhangSong/Prospective-Configuration/tree/main/predictive_coding/predictive_coding/pc_layer.py
    """

    def __init__(
        self,
        config,
        size,
        init_log_sigma=0.0,
        is_trainable_log_sigma=True,
        text=None,
        **kwargs,
    ):
        """Creates a new instance of ``VarPCLayer``.
        Args:
            size: The size of this layer. This is required as variance is created at the start and maintained afterwards, like in creating a normalization layer you need to specify the size.
            init_log_sigma: The initial log_sigma.
            is_trainable_log_sigma: Whether train log_sigma or not.
            kwargs: The keyword arguments to be passed to underlying ``PCLayer``.
        """

        assert (
            "energy_fn" not in list(kwargs.keys())
        ), "The ``energy_fn`` is specified in VarPCLayer. Thus, cannot be specified in kwargs to underlying ``PCLayer``."

        super().__init__(
            energy_fn=self.gaussian_energy,
            ** kwargs
        )

        self.text = text
        self.transformer_config = config
        # self.step = 1

        assert isinstance(init_log_sigma, float)
        self.init_log_sigma = init_log_sigma

        assert isinstance(is_trainable_log_sigma, bool)
        self.is_trainable_log_sigma = is_trainable_log_sigma

        log_sigma = torch.full(
            size, self.init_log_sigma
        )
        if self.is_trainable_log_sigma == True:
            self.log_sigma = torch.nn.Parameter(log_sigma)
        else:
            self.log_sigma = log_sigma

    def gaussian_energy(self, inputs):

        t = inputs['mu'] - inputs['x']

        return t * t / inputs['log_sigma'].exp() + inputs['log_sigma']

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        if self.transformer_config.print_energy_errors:
            print(self.text, mu.size())
            print('training:', self.training)
            print('sigma abs sum:',
                  torch.sum(torch.abs(self.log_sigma)).cpu().item())
            print(self.log_sigma)
            if self._x is not None:
                if mu.size() != self._x.size():
                    print('x is old')
                else:
                    print(
                    '% equal x and mu:', torch.sum(
                        torch.eq(mu, self._x)).cpu().item()/torch.numel(mu))
                print('x abs sum:', torch.sum(torch.abs(self._x)).cpu().item())
                print('mu abs sum:', torch.sum(torch.abs(mu)).cpu().item())
            else:
                print('x is None')
            print('---')

        return super().forward(
            mu=mu,
            energy_fn_additional_inputs={'log_sigma': self.log_sigma},
        )

