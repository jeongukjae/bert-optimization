import csv
import os
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import List, Optional, Tuple

import tensorflow as tf
import tensorflow_addons as tfa


def read_table(input_path: str, delimiter: str = "\t") -> List[List[str]]:
    """
    read table file (like tsv, csv)

    change delimiter to parse another format.
    """
    with open(input_path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        return [line for line in reader]


class GLUEClassificationProcessor(ABC):
    @abstractmethod
    def get_train(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def get_dev(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def get_test(self, path: str):
        raise NotImplementedError

    @abstractstaticmethod
    def get_label_to_index(self):
        raise NotImplementedError

    @abstractmethod
    def update_state(self, target, preds, validation=False):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self, validation=False):
        raise NotImplementedError

    @abstractmethod
    def reset_states(self, validation=False):
        raise NotImplementedError

    @abstractmethod
    def get_hash(self):
        raise NotImplementedError

    @abstractmethod
    def get_key(self):
        raise NotImplementedError


class CoLAProcessor(GLUEClassificationProcessor):
    def __init__(self):
        self.mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1)
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()

        self.val_mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1)
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    def get_train(self, path: str):
        return self.parse_cola_dataset(read_table(os.path.join(path, "train.tsv")), False)

    def get_dev(self, path: str):
        return self.parse_cola_dataset(read_table(os.path.join(path, "dev.tsv")), False)

    def get_test(self, path: str):
        return self.parse_cola_dataset(read_table(os.path.join(path, "train.tsv")), True)

    def get_label_to_index(self):
        return {"1": 1, "0": 0}

    @tf.function
    def update_state(self, targets, preds, validation=False):
        if validation:
            self.val_acc.update_state(targets, preds)
            self.val_mcc.update_state(tf.expand_dims(targets, 1), tf.expand_dims(tf.argmax(preds, -1), 1))
        else:
            self.acc.update_state(targets, preds)
            self.mcc.update_state(tf.expand_dims(targets, 1), tf.expand_dims(tf.argmax(preds, -1), 1))

    def get_metrics(self, validation=False):
        if validation:
            return {"Acc": self.val_acc.result(), "MCC": self.val_mcc.result()[0]}
        return {"Acc": self.acc.result(), "MCC": self.mcc.result()[0]}

    def reset_states(self, validation=False):
        if validation:
            self.val_acc.reset_states()
            self.val_mcc.reset_states()
        else:
            self.acc.reset_states()
            self.mcc.reset_states()

    def get_hash(self):
        return f"{self.val_mcc.result()[0]:.4f}-{self.val_acc.result():.4f}"

    def get_key(self):
        return self.val_mcc.result()[0]

    @staticmethod
    def parse_cola_dataset(lines: List[List[str]], is_test: bool) -> Tuple[Optional[List[str]], List[str]]:
        """
        Parse CoLA Dataset (GLUE)
        """
        if is_test:
            # test.tsv has header row
            return None, [line[1] for line in lines[1:]]

        return [line[1] for line in lines], [line[3] for line in lines]
