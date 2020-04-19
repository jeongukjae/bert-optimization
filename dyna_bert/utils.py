from argparse import ArgumentParser

import tensorflow as tf

from .models.bert import BertModel
from .models.transformer import ConcatenatedSelfAttention, MultiHeadSelfAttention


def get_default_bert_argument_parser():
    """Get Default ArgumentParser for BERT downstream tasks"""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="pretrained bert model path", required=True)
    parser.add_argument("--config", type=str, help="bert config path", required=True)
    parser.add_argument("--output", type=str, help="output directory", required=True)
    parser.add_argument("--dataset", type=str, help="data path", required=True)
    parser.add_argument("--vocab", type=str, help="vocab path", required=True)

    parser.add_argument("--task", type=str, help="task name to train", required=True)
    parser.add_argument("--use-gpu", action="store_true", help="whether to use gpu")

    parser.add_argument("--epoch", type=int, default=3, help="num epoch")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="initial learing rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay of AdamW")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="ratio of warmup data")
    parser.add_argument("--do-lower-case", action="store_true", help="whether to do lower case")
    parser.add_argument("--max-sequence-length", type=int, default=128, help="max sequence length of input")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="batch size for eval")
    parser.add_argument("--train-batch-size", type=int, default=32, help="batch size for training")
    parser.add_argument("--warmup-rate", type=float, default=0.1, help="rate of trainig data to use for warm up")

    parser.add_argument("--log-interval", type=int, default=5, help="interval to log")
    parser.add_argument("--val-interval", type=int, default=50, help="interval to validate model")

    return parser


def load_bert_weights(checkpoint_path: str, model: BertModel, use_splitted: bool):
    checkpoint = tf.train.load_checkpoint(checkpoint_path)

    load_embedding_weights(checkpoint, model.token_embeddings, "bert/embeddings/word_embeddings")
    load_embedding_weights(checkpoint, model.token_type_embeddings, "bert/embeddings/token_type_embeddings")
    load_embedding_weights(checkpoint, model.position_embeddings, "bert/embeddings/position_embeddings")
    load_layer_norm_weights(checkpoint, model.embedding_layer_norm, "bert/embeddings/LayerNorm")

    for layer_index, encoder in enumerate(model.encoders):
        encoder_prefix = f"bert/encoder/layer_{layer_index}"

        if use_splitted:
            load_concatenated_attention_weights(checkpoint, encoder.attention, f"{encoder_prefix}/attention")
        else:
            load_multohead_attention_weights(checkpoint, encoder.attention, f"{encoder_prefix}/attention")

        load_layer_norm_weights(checkpoint, encoder.attention_norm, f"{encoder_prefix}/attention/output/LayerNorm")

        load_layer_norm_weights(checkpoint, encoder.output_norm, f"{encoder_prefix}/output/LayerNorm")
        load_dense_weights(checkpoint, encoder.intermediate, f"{encoder_prefix}/intermediate/dense")
        load_dense_weights(checkpoint, encoder.output_dense, f"{encoder_prefix}/output/dense")

    load_dense_weights(checkpoint, model.pooler_layer, "bert/pooler/dense")


def load_embedding_weights(checkpoint, layer: tf.keras.layers.Embedding, prefix: str):
    layer.set_weights([checkpoint.get_tensor(prefix)])


def load_layer_norm_weights(checkpoint, layer: tf.keras.layers.LayerNormalization, prefix: str):
    # https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/layers/normalization.py#L1010-L1058
    layer.set_weights([checkpoint.get_tensor(f"{prefix}/gamma"), checkpoint.get_tensor(f"{prefix}/beta")])


def load_dense_weights(checkpoint, layer: tf.keras.layers.Dense, prefix: str):
    layer.set_weights([checkpoint.get_tensor(f"{prefix}/kernel"), checkpoint.get_tensor(f"{prefix}/bias")])


def load_concatenated_attention_weights(checkpoint, layer: ConcatenatedSelfAttention, prefix: str):
    query_kernel = checkpoint.get_tensor(f"{prefix}/self/query/kernel")
    query_bias = checkpoint.get_tensor(f"{prefix}/self/query/bias")

    key_kernel = checkpoint.get_tensor(f"{prefix}/self/key/kernel")
    key_bias = checkpoint.get_tensor(f"{prefix}/self/key/bias")

    value_kernel = checkpoint.get_tensor(f"{prefix}/self/value/kernel")
    value_bias = checkpoint.get_tensor(f"{prefix}/self/value/bias")

    num_heads = len(layer.heads)
    query_kernel = tf.split(query_kernel, num_heads, 1)
    key_kernel = tf.split(key_kernel, num_heads, 1)
    value_kernel = tf.split(value_kernel, num_heads, 1)

    query_bias = tf.split(query_bias, num_heads, 0)
    key_bias = tf.split(key_bias, num_heads, 0)
    value_bias = tf.split(value_bias, num_heads, 0)

    for head_index, head in enumerate(layer.heads):
        kernel = tf.concat((query_kernel[head_index], key_kernel[head_index], value_kernel[head_index]), axis=1)
        bias = tf.concat((query_bias[head_index], key_bias[head_index], value_bias[head_index]), axis=0)

        head.set_weights([kernel, bias])

    load_dense_weights(checkpoint, layer.output_dense, f"{prefix}/output/dense")


def load_multohead_attention_weights(checkpoint, layer: MultiHeadSelfAttention, prefix: str):
    query_kernel = checkpoint.get_tensor(f"{prefix}/self/query/kernel")
    query_bias = checkpoint.get_tensor(f"{prefix}/self/query/bias")

    key_kernel = checkpoint.get_tensor(f"{prefix}/self/key/kernel")
    key_bias = checkpoint.get_tensor(f"{prefix}/self/key/bias")

    value_kernel = checkpoint.get_tensor(f"{prefix}/self/value/kernel")
    value_bias = checkpoint.get_tensor(f"{prefix}/self/value/bias")

    layer.qkv_projection.set_weights(
        [
            tf.concat([query_kernel, key_kernel, value_kernel], axis=1),
            tf.concat([query_bias, key_bias, value_bias], axis=0),
        ]
    )
    load_dense_weights(checkpoint, layer.output_dense, f"{prefix}/output/dense")
