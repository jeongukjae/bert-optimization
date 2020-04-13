import pytest
import torch

from dyna_bert.models.transformer import MultiHeadSelfAttention, TransformerEncoder


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
