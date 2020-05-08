import pytest
import tensorflow as tf

from bert_optimization.models.bert import BertConfig, BertMLMHead, BertModel, BertNSPHead


@pytest.fixture
def bert_config():
    # use smaller dimension for faster unit tests
    return BertConfig(100, intermediate_size=128)


@pytest.mark.parametrize("batch_size, seq_len", [pytest.param(1, 3), pytest.param(3, 12)])
def test_shapes_of_bert_model_outputs(bert_config: BertConfig, batch_size: int, seq_len: int):
    """Check shapes of BERT model outputs"""
    # force to set output embedding and hidden states to True
    bert_config.output_embedding = True
    bert_config.output_hidden_states = True
    bert = BertModel(bert_config)

    input_ids = tf.random.uniform((batch_size, seq_len), maxval=bert_config.vocab_size)
    token_type_ids = tf.random.uniform((batch_size, seq_len), maxval=bert_config.type_vocab_size)
    attention_mask = tf.cast(tf.random.uniform((batch_size, seq_len), maxval=2), tf.dtypes.float32)

    outputs = bert([input_ids, token_type_ids, attention_mask])

    assert len(outputs) == 4
    assert outputs[0].shape == (batch_size, seq_len, bert_config.hidden_size)  # sequence output
    assert outputs[1].shape == (batch_size, bert_config.hidden_size)  # pooled output
    assert outputs[2].shape == (batch_size, seq_len, bert_config.hidden_size)  # embeddings
    assert all(hidn.shape == (batch_size, seq_len, bert_config.hidden_size) for hidn in outputs[3])  # hidden states


@pytest.mark.parametrize("batch_size", [pytest.param(1), pytest.param(3)])
def test_shape_of_nsp_head_output(bert_config: BertConfig, batch_size: int):
    """Check shape of NSP Head outputs"""
    nsp_head = BertNSPHead()

    pooled_output = tf.random.uniform((batch_size, bert_config.hidden_size))
    outputs = nsp_head(pooled_output)

    assert outputs.shape == (batch_size, 2)


@pytest.mark.parametrize("batch_size, seq_len", [pytest.param(1, 3), pytest.param(3, 12)])
def test_shape_of_mlm_head_output(bert_config: BertConfig, batch_size: int, seq_len: int):
    """Check shape of MLM model outputs"""
    mlm_head = BertMLMHead(bert_config)

    pooled_output = tf.random.uniform((batch_size, seq_len, bert_config.hidden_size))
    outputs = mlm_head(pooled_output)

    assert outputs.shape == (batch_size, seq_len, bert_config.vocab_size)
