from typing import Generator, Tuple

import pytest
import torch
from torch import nn

from dyna_bert.models.transformer import (
    ConcatenatedSelfAttention,
    MultiHeadSelfAttention,
    SelfAttention,
    TransformerEncoder,
)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size, intermediate_size, activation",
    [pytest.param(1, 3, 8, 16, 36, "gelu"), pytest.param(3, 12, 4, 8, 16, "relu")],
)
def test_shape_of_transformer_encoder_output(
    batch_size: int, seq_len: int, num_heads: int, hidden_size: int, intermediate_size: int, activation: str
):
    """Check shape of TransformerEncoder outputs"""
    encoder = TransformerEncoder(num_heads, hidden_size, intermediate_size, 0.0, activation)

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.randint(2, (batch_size, seq_len)).bool()

    encoder_output = encoder(sequence, attention_mask)

    assert encoder_output.shape == (batch_size, seq_len, hidden_size)


def test_transformer_encoder_should_raise_with_invalid_activation_function_name():
    """Check exception raised by Transformer Encoder with invalid Activation function name"""

    with pytest.raises(ValueError):
        TransformerEncoder(10, 20, 30, 0.0, "invalid-name")


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_shape_of_multihead_self_attention_output(batch_size: int, seq_len: int, num_heads: int, hidden_size: int):
    """Check shape of MultiHeadSelfAttention outputs"""
    attention = MultiHeadSelfAttention(num_heads, hidden_size, 0.0)

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.randint(2, (batch_size, seq_len)).bool()

    attention_output = attention(sequence, attention_mask)

    assert attention_output.shape == (batch_size, seq_len, hidden_size)


def test_multihead_attention_should_raise_with_invalid_num_heads():
    """Check exception raised by MultiHead Attention with invalid num heads"""

    with pytest.raises(ValueError):
        MultiHeadSelfAttention(10, 12, 0.0)

    with pytest.raises(ValueError):
        MultiHeadSelfAttention(10, 25, 0.0)


@pytest.mark.parametrize(
    "batch_size, seq_len, input_size, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_shape_of_self_attention_output(batch_size: int, seq_len: int, input_size: int, hidden_size: int):
    """Check shape of SelfAttention outputs"""
    attention = SelfAttention(hidden_size, hidden_size, 0.0)

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.randint(2, (batch_size, seq_len)).bool()

    attention_output = attention(sequence, attention_mask)

    assert attention_output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_shape_of_concatenated_self_attention_output(batch_size: int, seq_len: int, num_heads: int, hidden_size: int):
    """Check shape of ConcatenatedSelfAttention outputs"""
    attention = ConcatenatedSelfAttention(num_heads, hidden_size, 0.0)

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.randint(2, (batch_size, seq_len)).bool()

    attention_output = attention(sequence, attention_mask)

    assert attention_output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_output_of_self_attention_and_multi_head_self_attention(
    batch_size: int, seq_len: int, num_heads: int, hidden_size: int
):
    """Check both outputs from attention and multi head attention are same"""
    assert hidden_size % num_heads == 0

    attention_multihead = MultiHeadSelfAttention(num_heads, hidden_size, 0.0)
    attention_concat = [SelfAttention(hidden_size, int(hidden_size / num_heads), 0.0) for _ in range(num_heads)]

    for i, (weight, bias) in zip(range(num_heads), _convert_attn_weight(attention_multihead.qkv_linear, num_heads)):

        assert attention_concat[i].qkv_linear.weight.shape == weight.shape
        assert attention_concat[i].qkv_linear.bias.shape == bias.shape

        attention_concat[i].qkv_linear.weight.data = weight
        attention_concat[i].qkv_linear.bias.data = bias

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.tensor([[False] * seq_len] * batch_size)

    # 1. multi head attention
    multi_output = attention_multihead(sequence, attention_mask)

    # 2. single attention + concat
    single_output = torch.cat([attention(sequence, attention_mask) for attention in attention_concat], dim=-1)
    single_output = attention_multihead.output(single_output)

    assert multi_output.shape == single_output.shape
    assert torch.allclose(single_output, multi_output)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_output_of_concatenated_attention_and_multi_head_self_attention(
    batch_size: int, seq_len: int, num_heads: int, hidden_size: int
):
    """Check both outputs from concatenated attention and multi head attention are same"""
    assert hidden_size % num_heads == 0

    attention_multihead = MultiHeadSelfAttention(num_heads, hidden_size, 0.0)
    attention_concat = ConcatenatedSelfAttention(num_heads, hidden_size, 0.0)

    for i, (weight, bias) in zip(range(num_heads), _convert_attn_weight(attention_multihead.qkv_linear, num_heads)):
        assert attention_concat.heads[i].qkv_linear.weight.shape == weight.shape
        assert attention_concat.heads[i].qkv_linear.bias.shape == bias.shape

        attention_concat.heads[i].qkv_linear.weight.data = weight
        attention_concat.heads[i].qkv_linear.bias.data = bias

    attention_concat.output.weight.data = attention_multihead.output.weight.data
    attention_concat.output.bias.data = attention_multihead.output.bias.data

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.tensor([[False] * seq_len] * batch_size)

    # 1. multi head attention
    multi_output = attention_multihead(sequence, attention_mask)

    # 2. single attention + concat
    single_output = attention_concat(sequence, attention_mask)

    assert multi_output.shape == single_output.shape
    assert torch.allclose(single_output, multi_output)


def _convert_attn_weight(
    qkv_linear: nn.Linear, num_heads: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    wq, wk, wv = qkv_linear.weight.data.chunk(3, dim=0)
    bq, bk, bv = qkv_linear.bias.data.chunk(3, dim=0)

    wq = wq.chunk(num_heads, 0)
    wk = wk.chunk(num_heads, 0)
    wv = wv.chunk(num_heads, 0)

    bq = bq.chunk(num_heads, 0)
    bk = bk.chunk(num_heads, 0)
    bv = bv.chunk(num_heads, 0)

    for i in range(num_heads):
        weight = torch.cat((wq[i], wk[i], wv[i]), dim=0)
        bias = torch.cat((bq[i], bk[i], bv[i]), dim=0)

        yield weight, bias
