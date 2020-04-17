"""
Rewiring Functions.

https://arxiv.org/abs/2004.04037
"""
from typing import List

import tensorflow as tf

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
    attention.heads = heads

    # rewire output linear layer
    _change_rank_of_neuron(attention.output_dense, rank, 0, False)


def rewire_ffn(encoder: TransformerEncoder, rank: List[int]):
    """
    Rewiring Feed Forward Network

    https://arxiv.org/abs/2004.04037
    """
    _change_rank_of_neuron(encoder.intermediate, rank, 1, True)
    _change_rank_of_neuron(encoder.output_dense, rank, 0, False)


def _change_rank_of_neuron(linear: tf.keras.layers.Layer, rank: List[int], weight_rank: int, change_bias: bool):
    weights = linear.get_weights()

    linear_weights = [w for _, w in sorted(zip(rank, tf.split(weights[0], len(rank), weight_rank)))]
    linear_weight = tf.concat(linear_weights, weight_rank)

    if not change_bias:
        linear.set_weights([linear_weight, weights[1]])
    else:
        linear_biases = [b for _, b in sorted(zip(rank, tf.split(weights[1], len(rank), 0)))]
        linear_bias = tf.concat(linear_biases, 0)
        linear.set_weights([linear_weight, linear_bias])
