import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import generic_utils, metrics_utils


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)

        self.top_k = top_k
        self.class_id = class_id

        self.thresholds = metrics_utils.parse_init_thresholds(thresholds, default_threshold=0.5)
        self.true_positives = self.add_weight("true_positives", shape=(len(self.thresholds),), initializer="zeros")
        self.false_positives = self.add_weight("false_positives", shape=(len(self.thresholds),), initializer="zeros")
        self.false_negatives = self.add_weight("false_negatives", shape=(len(self.thresholds),), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

    def result(self):
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        result = tf.math.divide_no_nan(2 * (precision * recall), (precision + recall))

        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(generic_utils.to_list(self.thresholds))
        K.batch_set_value([(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {"thresholds": self.init_thresholds, "top_k": self.top_k, "class_id": self.class_id}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
