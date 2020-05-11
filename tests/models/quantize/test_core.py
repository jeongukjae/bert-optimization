import tensorflow as tf

from bert_optimization.models.quantize.core import QuantizationMode, QuantizedDense


def test_quantized_dense_can_be_called():
    dense = QuantizedDense(10)
    input_tensor = tf.random.uniform((20, 20))

    result = dense(input_tensor)
    assert result.shape == (20, 10)


def test_quantized_dense_output():
    dense = QuantizedDense(2, mode=QuantizationMode.DYNAMIC)
    input_tensor = tf.constant([0.0, 1.0, 3.0, 4.0], shape=(2, 2))
    # build
    result = dense(input_tensor)

    dense.set_weights([tf.constant([0.0, 1.0, 2.0, 3.0], shape=(2, 2)), tf.constant([1.0, 2.0])])

    result = dense(input_tensor)
    assert result.shape == (2, 2)
