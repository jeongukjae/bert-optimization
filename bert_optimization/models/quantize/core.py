from abc import ABC, abstractmethod
from enum import Enum

import tensorflow as tf

from .functional import fake_quantize


class QuantizationMode(Enum):
    # TODO: FIXED Quantization
    NONE = 0
    FIXED = 1
    DYNAMIC = 2


class QuantizedBase(ABC, tf.keras.layers.Layer):
    def __init__(self, num_bits=8, start_step=0, mode=QuantizationMode.NONE, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mode = mode
        self.num_bits = num_bits
        self.start_step = start_step
        self._step = 0

    def call(self, input_tensors, *args, **kwargs):
        if self.mode == QuantizationMode.NONE:
            return super().call(input_tensors, *args)

        if "training" in kwargs and kwargs["training"] is None:
            return self.quantized_inference(input_tensors, *args)
        else:
            result = tf.cond(
                self._step >= self.start_step,
                lambda: self.quantized_training(input_tensors, *args),
                lambda: super().call(input_tensors, *args),
            )
            self._step += 1

        return result

    @abstractmethod
    def quantized_training(self, input_tensors, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def quantized_inference(self, input_tensors, **kwargs):
        raise NotImplementedError


class QuantizedDense(QuantizedBase, tf.keras.layers.Dense):
    def __init__(self, units, *args, num_bits=8, **kwargs):
        super().__init__(units=units, *args, **kwargs)

        self.num_bits = num_bits

    @tf.function
    def quantized_training(self, input_tensor, **kwargs):
        input_tensor = fake_quantize(input_tensor, self.num_bits)

        outputs = tf.matmul(input_tensor, fake_quantize(self.kernel, self.num_bits))
        if self.use_bias:
            return tf.nn.bias_add(outputs, self.bias)
        return outputs

    @tf.function
    def quantized_inference(self, input_tensor, **kwargs):
        return self.quantized_training(input_tensor, **kwargs)


class QuantizedEmbedding(QuantizedBase, tf.keras.layers.Embedding):
    def __init__(self, input_dim, output_dim, *args, num_bits=8, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, *args, **kwargs)

        self.num_bits = num_bits

    @tf.function
    def quantized_training(self, input_tensor, **kwargs):
        return tf.nn.embedding_lookup(fake_quantize(self.embeddings), input_tensor)

    @tf.function
    def quantized_inference(self, input_tensor, **kwargs):
        return self.quantized_training(input_tensor, **kwargs)
