import os
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from . import tokenizer


def read_table(input_path: str, delimiter: str = "\t") -> List[List[str]]:
    """
    read table file (like tsv, csv)

    change delimiter to parse another format.
    """
    with open(input_path) as f:
        return [line.strip().split(delimiter) for line in f]


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


class MRPCProcessor(GLUEClassificationProcessor):
    def __init__(self):
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.recall = tf.keras.metrics.Recall()
        self.precision = tf.keras.metrics.Precision()

        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_recall = tf.keras.metrics.Recall()
        self.val_precision = tf.keras.metrics.Precision()

    def get_train(self, path: str):
        return self.parse_mrpc_dataset(read_table(os.path.join(path, "train.tsv")), False)

    def get_dev(self, path: str):
        return self.parse_mrpc_dataset(read_table(os.path.join(path, "dev.tsv")), False)

    def get_test(self, path: str):
        return self.parse_mrpc_dataset(read_table(os.path.join(path, "train.tsv")), True)

    def get_label_to_index(self):
        return {"1": 1, "0": 0}

    @tf.function
    def update_state(self, targets, preds, validation=False):
        maxed_preds = tf.math.argmax(preds, -1)
        if validation:
            self.val_acc.update_state(targets, preds)
            self.val_recall.update_state(targets, maxed_preds)
            self.val_precision.update_state(targets, maxed_preds)
        else:
            self.acc.update_state(targets, preds)
            self.recall.update_state(targets, maxed_preds)
            self.precision.update_state(targets, maxed_preds)

    def get_metrics(self, validation=False):
        if validation:
            return {
                "Acc": self.val_acc.result(),
                "F1": 2 / (self.val_recall.result() ** -1 + self.val_precision.result() ** -1),
            }
        return {
            "Acc": self.acc.result(),
            "F1": 2 / (self.recall.result() ** -1 + self.precision.result() ** -1),
        }

    def reset_states(self, validation=False):
        if validation:
            self.val_acc.reset_states()
            self.val_recall.reset_states()
            self.val_precision.reset_states()
        else:
            self.acc.reset_states()
            self.recall.reset_states()
            self.precision.reset_states()

    def get_hash(self):
        return f"{self.val_acc.result():.4f}"

    def get_key(self):
        return self.val_acc.result()

    @staticmethod
    def parse_mrpc_dataset(lines: List[List[str]], is_test: bool) -> Tuple[Optional[List[str]], List[str], List[str]]:
        """
        Parse MRPC Dataset (GLUE)
        """
        # dataset files have a header row
        lines = lines[1:]
        if is_test:
            return None, [line[3] for line in lines], [line[4] for line in lines]

        return [line[0] for line in lines], [line[3] for line in lines], [line[4] for line in lines]


def convert_single_sentence(
    data: Tuple[Optional[List[str]], List[str]],
    label_to_index: Dict[str, int],
    tokenizer: tokenizer.SubWordTokenizer,
    max_length: int,
):
    labels = [0] * len(data[1]) if data[0] is None else [label_to_index[label] for label in data[0]]
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for example in data[1]:
        tokens = tokenizer.tokenize(example)[: max_length - 2]
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        padding_size = max_length - len(ids)

        input_ids.append(ids + [0] * padding_size)
        token_type_ids.append([0] * max_length)
        attention_mask.append([1.0] * len(ids) + [0.0] * padding_size)

    return (labels, input_ids, token_type_ids, attention_mask)


def convert_sentence_pair(
    data: Tuple[Optional[List[str]], List[str], List[str]],
    label_to_index: Dict[str, int],
    tokenizer: tokenizer.SubWordTokenizer,
    max_length: int,
):
    labels = [0] * len(data[1]) if data[0] is None else [label_to_index[label] for label in data[0]]

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for example_index in range(len(data[1])):
        tokens_a = tokenizer.tokenize(data[1][example_index])
        tokens_b = tokenizer.tokenize(data[2][example_index])
        truncate_seq_pair(tokens_a, tokens_b, max_length)

        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"])
        padding_size = max_length - len(ids)

        input_ids.append(ids + [0] * padding_size)
        token_type_ids.append([0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) + [0] * padding_size)
        attention_mask.append([1.0] * len(ids) + [0.0] * padding_size)

    return (labels, input_ids, token_type_ids, attention_mask)


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
