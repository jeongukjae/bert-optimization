import pytest
import tensorflow as tf

from bert_optimization.models.bert import BertConfig
from bert_optimization.models.heads import BertMLMHead, BertNSPHead, ClassificationHead


@pytest.mark.parametrize("batch_size, hidden_size, num_class", [pytest.param(1, 3, 12), pytest.param(3, 12, 4)])
def test_shape_of_classification_heads(batch_size: int, hidden_size: int, num_class: int):
    """Check shape of Classification head model outputs"""
    classification_head = ClassificationHead(num_class, 0.0)

    pooled_output = tf.random.uniform((batch_size, hidden_size))
    outputs = classification_head(pooled_output)

    assert outputs.shape == (batch_size, num_class)


@pytest.mark.parametrize("batch_size", [pytest.param(1), pytest.param(3)])
def test_shape_of_nsp_head_output(batch_size: int):
    """Check shape of NSP Head outputs"""
    hidden_size = 10

    nsp_head = BertNSPHead()

    pooled_output = tf.random.uniform((batch_size, hidden_size))
    outputs = nsp_head(pooled_output)

    assert outputs.shape == (batch_size, 2)


@pytest.mark.parametrize("batch_size, seq_len", [pytest.param(1, 3), pytest.param(3, 12)])
def test_shape_of_mlm_head_output(batch_size: int, seq_len: int):
    """Check shape of MLM model outputs"""
    hidden_size = 10
    vocab_size = 500

    mlm_head = BertMLMHead(hidden_size, vocab_size)

    pooled_output = tf.random.uniform((batch_size, seq_len, hidden_size))
    outputs = mlm_head(pooled_output)

    assert outputs.shape == (batch_size, seq_len, vocab_size)
