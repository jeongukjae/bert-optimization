from dyna_bert.optimization.rewire import rewire_ffn, rewire_mha, rewire_transformer_encoder
from typing import List
from dyna_bert.models.transformer import ConcatenatedSelfAttention, TransformerEncoder
import pytest

import torch


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size, rank",
    [
        pytest.param(1, 2, 3, 3, [0, 2, 1]),
        pytest.param(1, 3, 8, 16, [0, 7, 4, 5, 1, 2, 3, 6]),
        pytest.param(1, 3, 8, 16, [1, 7, 4, 5, 8, 2, 3, 6]),
        pytest.param(3, 12, 4, 8, [0, 3, 2, 1]),
    ],
)
def test_rewire_mha(batch_size: int, seq_len: int, num_heads: int, hidden_size: int, rank: List[int]):
    """Check output is the same before and after rewire_mha"""
    attention = ConcatenatedSelfAttention(num_heads, hidden_size, 0.0)
    attention.eval()

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.tensor([[False] * seq_len] * batch_size)

    attention_output_before = attention(sequence, attention_mask)

    rewire_mha(attention, rank)

    attention_output_after = attention(sequence, attention_mask)

    assert torch.allclose(attention_output_before, attention_output_after, rtol=1e-3)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size, intermediate_size, rank",
    [
        pytest.param(1, 2, 3, 3, 12, [0, 2, 1]),
        pytest.param(1, 3, 8, 16, 32, [0, 7, 4, 5, 1, 2, 3, 6]),
        pytest.param(1, 3, 8, 16, 64, [1, 7, 4, 5, 8, 2, 3, 6]),
        pytest.param(3, 12, 4, 8, 48, [0, 3, 2, 1]),
    ],
)
def test_rewire_ffn(
    batch_size: int, seq_len: int, num_heads: int, hidden_size: int, intermediate_size: int, rank: List[int]
):
    """Check output is the same before and after rewire_ffn"""
    encoder = TransformerEncoder(num_heads, hidden_size, intermediate_size, 0.0, "gelu")
    encoder.eval()

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.tensor([[False] * seq_len] * batch_size)

    encoder_output_before = encoder(sequence, attention_mask)

    rewire_ffn(encoder, rank)

    encoder_output_after = encoder(sequence, attention_mask)

    assert torch.allclose(encoder_output_before, encoder_output_after, rtol=1e-3)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size, intermediate_size, rank",
    [
        pytest.param(1, 2, 3, 3, 12, [0, 2, 1]),
        pytest.param(1, 3, 8, 16, 32, [0, 7, 4, 5, 1, 2, 3, 6]),
        pytest.param(1, 3, 8, 16, 64, [1, 7, 4, 5, 8, 2, 3, 6]),
        pytest.param(3, 12, 4, 8, 48, [0, 3, 2, 1]),
    ],
)
def test_rewire_transformer_encoder(
    batch_size: int, seq_len: int, num_heads: int, hidden_size: int, intermediate_size: int, rank: List[int]
):
    """Check output is the same before and after rewire_transformer_encoder"""
    encoder = TransformerEncoder(num_heads, hidden_size, intermediate_size, 0.0, "gelu")
    encoder.eval()

    sequence = torch.rand((batch_size, seq_len, hidden_size))
    attention_mask = torch.tensor([[False] * seq_len] * batch_size)

    encoder_output_before = encoder(sequence, attention_mask)

    rewire_transformer_encoder(encoder, rank)

    encoder_output_after = encoder(sequence, attention_mask)

    assert torch.allclose(encoder_output_before, encoder_output_after, rtol=1e-3)
