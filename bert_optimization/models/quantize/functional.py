import tensorflow as tf


def get_max_quant_value(num_bits: int) -> float:
    return 2 ** (num_bits - 1) - 1


def quantize(input_tensor: tf.Tensor, scale: float, num_bits: int):
    """
    https://arxiv.org/pdf/1910.06188.pdf
    """
    threshold = tf.cast(get_max_quant_value(num_bits), tf.float32)
    return tf.clip_by_value(tf.math.round(input_tensor * scale), -threshold, threshold)


def dequantize(input_tensor: tf.Tensor, scale: float):
    return input_tensor / scale


@tf.function
def quantize_and_dequantize(x: tf.Tensor, scale: float, num_bits=8):
    quantized = quantize(x, scale=scale, num_bits=num_bits)
    return dequantize(quantized, scale=scale)


@tf.function
def get_weight_scale_factor(weight: tf.Tensor, num_bits: int) -> float:
    """
    https://arxiv.org/pdf/1910.06188.pdf
    """
    threshold = tf.math.reduce_max(tf.math.abs(weight))
    return tf.cast(get_max_quant_value(num_bits), tf.float32) / threshold


@tf.function
def fake_quantize(x: tf.Tensor, num_bits=8):
    scale = get_weight_scale_factor(x, num_bits)
    return quantize_and_dequantize(x, scale, num_bits)


@tf.custom_gradient
def fake_quantize(x, num_bits=8):
    def straight_through_estimator(dy):
        return dy

    scale = get_weight_scale_factor(x, num_bits)
    return quantize_and_dequantize(x, scale, num_bits), straight_through_estimator
