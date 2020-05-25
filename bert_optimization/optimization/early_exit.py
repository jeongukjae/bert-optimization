import tensorflow as tf


@tf.function
def calculate_uncertainty(probs: tf.Tensor, epsilon=1e-10) -> tf.Tensor:
    """
    Input Shape: (batch size, class num)
    Output Shape: (batch size,)
    """
    N = tf.shape(probs)[-1]
    summation = tf.math.reduce_sum(probs * tf.math.log(probs + epsilon), axis=1)
    return summation / tf.math.log(1.0 / tf.cast(N, tf.float32))
