import tensorflow as tf

from .quantized_model import QuantizedDense, QuantizedEmbedding


def get_dense(aware_quantization: bool):
    if aware_quantization:
        return QuantizedDense

    return tf.keras.layers.Dense


def get_embedding(aware_quantization: bool):
    if aware_quantization:
        return QuantizedEmbedding

    return tf.keras.layers.Embedding
