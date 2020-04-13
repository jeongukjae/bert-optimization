import torch
import pytest

from dyna_bert.models.bert import BertConfig, BertModel


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

    input_ids = torch.randint(bert_config.vocab_size, (batch_size, seq_len))
    token_type_ids = torch.randint(bert_config.type_vocab_size, (batch_size, seq_len))
    attention_mask = torch.randint(2, (batch_size, seq_len)).bool()

    outputs = bert(input_ids, token_type_ids, attention_mask)

    assert len(outputs) == 4
    assert outputs[0].shape == (batch_size, seq_len, bert_config.hidden_size)  # sequence output
    assert outputs[1].shape == (batch_size, bert_config.hidden_size)  # pooled output
    assert outputs[2].shape == (batch_size, seq_len, bert_config.hidden_size)  # embeddings
    assert all(hidn.shape == (batch_size, seq_len, bert_config.hidden_size) for hidn in outputs[3])  # hidden states
