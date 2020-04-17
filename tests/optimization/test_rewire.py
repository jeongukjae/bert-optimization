from typing import List

import pytest
import tensorflow as tf

from dyna_bert.models.transformer import ConcatenatedSelfAttention, TransformerEncoder
from dyna_bert.optimization.rewire import rewire_ffn, rewire_mha, rewire_transformer_encoder


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

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[False] * seq_len] * batch_size)

    attention_output_before = attention(sequence, mask=attention_mask)

    rewire_mha(attention, rank)

    attention_output_after = attention(sequence, mask=attention_mask)

    tf.debugging.assert_near(attention_output_before, attention_output_after)


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

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[False] * seq_len] * batch_size)

    encoder_output_before = encoder(sequence, attention_mask)

    rewire_ffn(encoder, rank)

    encoder_output_after = encoder(sequence, attention_mask)

    tf.debugging.assert_near(encoder_output_before, encoder_output_after)


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

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[False] * seq_len] * batch_size)

    encoder_output_before = encoder(sequence, mask=attention_mask)

    rewire_transformer_encoder(encoder, rank)

    encoder_output_after = encoder(sequence, mask=attention_mask)

    tf.debugging.assert_near(encoder_output_before, encoder_output_after)
