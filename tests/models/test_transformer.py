import pytest
import tensorflow as tf

from bert_optimization.models.transformer import (
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
    encoder = TransformerEncoder(num_heads, hidden_size, intermediate_size, 0.0, activation, False)

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[1.0] * seq_len] * batch_size)

    encoder_output = encoder(sequence, mask=attention_mask)

    assert encoder_output.shape == (batch_size, seq_len, hidden_size)


def test_transformer_encoder_should_raise_with_invalid_activation_function_name():
    """Check exception raised by Transformer Encoder with invalid Activation function name"""

    with pytest.raises(ValueError):
        TransformerEncoder(10, 20, 30, 0.0, "invalid-name", False)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_shape_of_multihead_self_attention_output(batch_size: int, seq_len: int, num_heads: int, hidden_size: int):
    """Check shape of MultiHeadSelfAttention outputs"""
    attention = MultiHeadSelfAttention(num_heads, hidden_size, 0.0, False)

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[1.0] * seq_len] * batch_size)

    attention_output = attention(sequence, mask=attention_mask)

    assert attention_output.shape == (batch_size, seq_len, hidden_size)


def test_multihead_attention_should_raise_with_invalid_num_heads():
    """Check exception raised by MultiHead Attention with invalid num heads"""

    with pytest.raises(ValueError):
        MultiHeadSelfAttention(10, 12, 0.0, False)

    with pytest.raises(ValueError):
        MultiHeadSelfAttention(10, 25, 0.0, False)


@pytest.mark.parametrize(
    "batch_size, seq_len, input_size, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_shape_of_self_attention_output(batch_size: int, seq_len: int, input_size: int, hidden_size: int):
    """Check shape of SelfAttention outputs"""
    attention = SelfAttention(hidden_size, 0.0, False)

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[1.0] * seq_len] * batch_size)

    attention_output = attention(sequence, mask=attention_mask)

    assert attention_output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_shape_of_concatenated_self_attention_output(batch_size: int, seq_len: int, num_heads: int, hidden_size: int):
    """Check shape of ConcatenatedSelfAttention outputs"""
    attention = ConcatenatedSelfAttention(num_heads, hidden_size, 0.0, False)

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[1.0] * seq_len] * batch_size)

    attention_output = attention(sequence, mask=attention_mask)

    assert attention_output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size", [pytest.param(1, 3, 16), pytest.param(3, 12, 8)],
)
def test_shape_of_concatenated_self_attention_output_with_head_mask(batch_size: int, seq_len: int, hidden_size: int):
    """Check shape of ConcatenatedSelfAttention outputs With Head Mask"""
    num_heads = 4

    attention = ConcatenatedSelfAttention(num_heads, hidden_size, 0.0, False)

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[1.0] * seq_len] * batch_size)
    # Masking two heads
    head_mask = tf.constant([[1.0, 1.0, 0.0, 0.0]] * batch_size)

    attention_output = attention(sequence, mask=attention_mask, head_mask=head_mask)

    assert attention_output.shape == (batch_size, seq_len, hidden_size)
    # Last two heads output must be zeros
    assert tf.reduce_all(attention_output[:, :, 2 * int(hidden_size / num_heads) :] == 0)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, hidden_size", [pytest.param(1, 3, 8, 16), pytest.param(3, 12, 4, 8)],
)
def test_output_of_concatenated_attention_and_multi_head_self_attention(
    batch_size: int, seq_len: int, num_heads: int, hidden_size: int
):
    """Check both outputs from concatenated attention and multi head attention are same"""
    assert hidden_size % num_heads == 0

    attention_multihead = MultiHeadSelfAttention(num_heads, hidden_size, 0.0, False)
    attention_concat = ConcatenatedSelfAttention(num_heads, hidden_size, 0.0, False)

    sequence = tf.random.uniform((batch_size, seq_len, hidden_size))
    attention_mask = tf.constant([[1.0] * seq_len] * batch_size)

    # Build TF Graphs
    multi_output = attention_multihead(sequence, mask=attention_mask)
    single_output = attention_concat(sequence, mask=attention_mask)

    for i, (weight, bias) in zip(range(num_heads), _convert_attn_weight(attention_multihead.qkv_projection, num_heads)):
        attention_concat.heads[i].qkv_projection.set_weights([weight, bias])
    attention_concat.output_dense.set_weights(attention_multihead.output_dense.get_weights())

    # 1. multi head attention
    multi_output = attention_multihead(sequence, mask=attention_mask)

    # 2. single attention + concat
    single_output = attention_concat(sequence, mask=attention_mask)

    assert multi_output.shape == single_output.shape
    tf.debugging.assert_near(single_output, multi_output)


def _convert_attn_weight(qkv_projection, num_heads):
    weights = qkv_projection.get_weights()
    wq, wk, wv = tf.split(weights[0], 3, axis=1)
    bq, bk, bv = tf.split(weights[1], 3, axis=0)

    wq = tf.split(wq, num_heads, 1)
    wk = tf.split(wk, num_heads, 1)
    wv = tf.split(wv, num_heads, 1)

    bq = tf.split(bq, num_heads, 0)
    bk = tf.split(bk, num_heads, 0)
    bv = tf.split(bv, num_heads, 0)

    for i in range(num_heads):
        weight = tf.concat((wq[i], wk[i], wv[i]), axis=1)
        bias = tf.concat((bq[i], bk[i], bv[i]), axis=0)

        yield weight, bias
