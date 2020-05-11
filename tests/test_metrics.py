import tensorflow as tf

from bert_optimization.metrics import F1Score


def test_f1_correctness():
    f1_score = F1Score()

    # zero division case
    f1_score.update_state(tf.constant([0, 0, 0, 0, 0, 0]), tf.constant([0, 0, 0, 0, 0, 0]))

    assert f1_score.result().shape == tuple()
    assert f1_score.result() == 0.0

    f1_score.reset_states()
    f1_score.update_state(tf.constant([1, 0, 0, 1, 0, 0]), tf.constant([0, 1, 0, 1, 1, 0]))

    assert f1_score.result().shape == tuple()
    assert f1_score.result() == 0.4

    f1_score.update_state(tf.constant([0, 0, 0, 0, 0, 0]), tf.constant([0, 0, 0, 0, 0, 0]))

    assert f1_score.result().shape == tuple()
    assert f1_score.result() != 0.0
