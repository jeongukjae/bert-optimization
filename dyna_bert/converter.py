from typing import cast

import numpy as np
import tensorflow as tf
import torch
from torch import nn

from .models import BertConfig, BertModel, ConcatenatedSelfAttention, SelfAttention, TransformerEncoder


def load_tf_weight_to_pytorch_bert(bert: BertModel, config: BertConfig, tf_model_path: str):
    # load embedding layer
    _load_embedding(bert.token_embeddings, tf_model_path, "bert/embeddings/word_embeddings")
    _load_embedding(bert.token_type_embeddings, tf_model_path, "bert/embeddings/token_type_embeddings")
    _load_embedding(bert.position_embeddings, tf_model_path, "bert/embeddings/position_embeddings")
    _load_layer_norm(bert.embedding_layer_norm, tf_model_path, "bert/embeddings/LayerNorm")

    # load transformer encoders
    for layer_num in range(config.num_hidden_layers):
        encoder = cast(TransformerEncoder, bert.encoders[layer_num])
        encoder_path = f"bert/encoder/layer_{layer_num}"

        _load_self_attention(encoder.attention, tf_model_path, f"{encoder_path}/attention", config.num_attention_heads)
        _load_layer_norm(encoder.attention_norm, tf_model_path, f"{encoder_path}/attention/output/LayerNorm")
        _load_layer_norm(encoder.output_norm, tf_model_path, f"{encoder_path}/output/LayerNorm")

        _load_linear(encoder.intermediate, tf_model_path, f"{encoder_path}/intermediate/dense")
        _load_linear(encoder.output, tf_model_path, f"{encoder_path}/output/dense")

    # load pooler layer
    _load_linear(bert.pooler_layer, tf_model_path, f"bert/pooler/dense")


def _load_embedding(embedding: nn.Embedding, tf_model_path: str, embedding_path: str):
    embedding_weight = _load_tf_variable(tf_model_path, embedding_path)
    _load_torch_weight(embedding.weight, embedding_weight)


def _load_layer_norm(layer_norm: torch.nn.LayerNorm, tf_model_path: str, layer_norm_base: str):
    layer_norm_gamma = _load_tf_variable(tf_model_path, f"{layer_norm_base}/gamma")
    layer_norm_beta = _load_tf_variable(tf_model_path, f"{layer_norm_base}/beta")

    _load_torch_weight(layer_norm.weight, layer_norm_gamma)
    _load_torch_weight(layer_norm.bias, layer_norm_beta)


def _load_linear(linear: torch.nn.Linear, tf_model_path: str, linear_path: str, load_bias: bool = True):
    linear_weight = _load_tf_variable(tf_model_path, f"{linear_path}/kernel")
    linear_weight = np.transpose(linear_weight)
    _load_torch_weight(linear.weight, linear_weight)

    if load_bias:
        linear_bias = _load_tf_variable(tf_model_path, f"{linear_path}/bias")
        _load_torch_weight(linear.bias, linear_bias)


def _load_self_attention(param: ConcatenatedSelfAttention, tf_model_path: str, attention_path: str, num_heads: int):
    query_weight = _load_tf_variable(tf_model_path, f"{attention_path}/self/query/kernel")
    key_weight = _load_tf_variable(tf_model_path, f"{attention_path}/self/key/kernel")
    value_weight = _load_tf_variable(tf_model_path, f"{attention_path}/self/value/kernel")

    query_weight = np.transpose(query_weight)
    key_weight = np.transpose(key_weight)
    value_weight = np.transpose(value_weight)

    query_bias = _load_tf_variable(tf_model_path, f"{attention_path}/self/query/bias")
    key_bias = _load_tf_variable(tf_model_path, f"{attention_path}/self/key/bias")
    value_bias = _load_tf_variable(tf_model_path, f"{attention_path}/self/value/bias")

    query_weight = np.split(query_weight, num_heads, 0)
    key_weight = np.split(key_weight, num_heads, 0)
    value_weight = np.split(value_weight, num_heads, 0)

    query_bias = np.split(query_bias, num_heads, 0)
    key_bias = np.split(key_bias, num_heads, 0)
    value_bias = np.split(value_bias, num_heads, 0)

    for i in range(num_heads):
        weight = np.concatenate((query_weight[i], key_weight[i], value_weight[i]), axis=0)
        bias = np.concatenate((query_bias[i], key_bias[i], value_bias[i]), axis=0)

        heads = cast(SelfAttention, param.heads[i])
        _load_torch_weight(heads.qkv_linear.weight, weight)
        _load_torch_weight(heads.qkv_linear.bias, bias)

    _load_linear(param.output, tf_model_path, f"{attention_path}/output/dense")


def _load_tf_variable(model_path: str, key: str):
    return tf.train.load_variable(model_path, key).squeeze()


def _load_torch_weight(param: torch.Tensor, data):
    assert param.shape == data.shape
    param.data = torch.from_numpy(data)
