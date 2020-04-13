"""
Rewiring Functions.

https://arxiv.org/abs/2004.04037
"""
from typing import List

import torch
from torch import nn

from dyna_bert.models.transformer import ConcatenatedSelfAttention, TransformerEncoder


def rewire_transformer_encoder(encoder: TransformerEncoder, rank: List[int]):
    """
    Rewiring both Multi Head Attention and Feed Forward Network

    https://arxiv.org/abs/2004.04037
    """
    rewire_mha(encoder.attention, rank)
    rewire_ffn(encoder, rank)


def rewire_mha(attention: ConcatenatedSelfAttention, rank: List[int]):
    """
    Rewiring Multi Head Attention

    https://arxiv.org/abs/2004.04037
    """
    if len(rank) != len(attention.heads):
        raise ValueError("Length of rank and that of attention heads should be same")

    # rewire heads
    heads = [l for _, l in sorted(zip(rank, attention.heads))]
    attention.heads = nn.ModuleList(heads)

    # rewire output linear layer
    _change_rank_of_neuron(attention.output, rank, 1, False)


def rewire_ffn(encoder: TransformerEncoder, rank: List[int]):
    """
    Rewiring Feed Forward Network

    https://arxiv.org/abs/2004.04037
    """
    _change_rank_of_neuron(encoder.intermediate, rank, 0, True)
    _change_rank_of_neuron(encoder.output, rank, 1, False)


def _change_rank_of_neuron(linear: nn.Linear, rank: List[int], weight_rank: int, change_bias: bool):
    linear_weights = [w for _, w in sorted(zip(rank, linear.weight.data.chunk(len(rank), weight_rank)))]
    linear_weight = torch.cat(linear_weights, weight_rank)

    assert linear_weight.shape == linear.weight.shape

    linear.weight.data = linear_weight.data

    if change_bias:
        linear_biases = [b for _, b in sorted(zip(rank, linear.bias.data.chunk(len(rank), 0)))]
        linear_bias = torch.cat(linear_biases, 0)
        linear.bias.data = linear_bias.data
