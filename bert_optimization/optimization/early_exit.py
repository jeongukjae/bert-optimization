import tensorflow as tf


@tf.function
def calculate_uncertainty(logits: tf.Tensor) -> tf.Tensor:
    """
    Input Shape: (batch size, class num)
    Output Shape: (batch size,)
    """
    return tf.math.reduce_mean(logits * tf.math.log(logits), axis=1)
