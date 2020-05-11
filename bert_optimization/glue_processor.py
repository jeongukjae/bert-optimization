import os
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from . import tokenizer
from .metrics import F1Score


def read_table(input_path: str, delimiter: str = "\t") -> List[List[str]]:
    """
    read table file (like tsv, csv)

    change delimiter to parse another format.
    """
    with open(input_path) as f:
        return [line.strip().split(delimiter) for line in f]


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
        truncate_seq_pair(tokens_a, tokens_b, max_length - 3)

        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"])
        padding_size = max_length - len(ids)

        input_ids.append(ids + [0] * padding_size)
        token_type_ids.append([0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) + [0] * padding_size)
        attention_mask.append([1.0] * len(ids) + [0.0] * padding_size)

        assert len(input_ids[-1]) == max_length
        assert len(token_type_ids[-1]) == max_length
        assert len(attention_mask[-1]) == max_length

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
        self.f1 = F1Score()

        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_f1 = F1Score()

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
            self.val_f1.update_state(targets, maxed_preds)
        else:
            self.acc.update_state(targets, preds)
            self.f1.update_state(targets, maxed_preds)

    def get_metrics(self, validation=False):
        if validation:
            return {
                "Acc": self.val_acc.result(),
                "F1": self.val_f1.result(),
            }
        return {
            "Acc": self.acc.result(),
            "F1": self.f1.result(),
        }

    def reset_states(self, validation=False):
        if validation:
            self.val_acc.reset_states()
            self.val_f1.reset_states()
        else:
            self.acc.reset_states()
            self.f1.reset_states()

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


class MNLIProcessor(GLUEClassificationProcessor):
    def __init__(self):
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    def get_train(self, path: str):
        return self.parse_mnli_dataset(read_table(os.path.join(path, "train.tsv")), False)

    def get_dev(self, path: str):
        return self.parse_mnli_dataset(read_table(os.path.join(path, "dev_matched.tsv")), False)

    def get_test(self, path: str):
        return self.parse_mnli_dataset(read_table(os.path.join(path, "test_matched.tsv")), True)

    def get_label_to_index(self):
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    @tf.function
    def update_state(self, targets, preds, validation=False):
        if validation:
            self.val_acc.update_state(targets, preds)
        else:
            self.acc.update_state(targets, preds)

    def get_metrics(self, validation=False):
        if validation:
            return {"Acc": self.val_acc.result()}
        return {"Acc": self.acc.result()}

    def reset_states(self, validation=False):
        if validation:
            self.val_acc.reset_states()
        else:
            self.acc.reset_states()

    def get_hash(self):
        return f"{self.val_acc.result():.4f}"

    def get_key(self):
        return self.val_acc.result()

    @staticmethod
    def parse_mnli_dataset(lines: List[List[str]], is_test: bool) -> Tuple[Optional[List[str]], List[str], List[str]]:
        """
        Parse MNLI Dataset (GLUE)
        """
        # dataset files have a header row
        lines = lines[1:]
        if is_test:
            return None, [line[8] for line in lines], [line[9] for line in lines]

        return [line[-1] for line in lines], [line[8] for line in lines], [line[9] for line in lines]


class SST2Processor(GLUEClassificationProcessor):
    def __init__(self):
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    def get_train(self, path: str):
        return self.parse_sst2_dataset(read_table(os.path.join(path, "train.tsv")), False)

    def get_dev(self, path: str):
        return self.parse_sst2_dataset(read_table(os.path.join(path, "dev.tsv")), False)

    def get_test(self, path: str):
        return self.parse_sst2_dataset(read_table(os.path.join(path, "test.tsv")), True)

    def get_label_to_index(self):
        return {"0": 0, "1": 1}

    @tf.function
    def update_state(self, targets, preds, validation=False):
        if validation:
            self.val_acc.update_state(targets, preds)
        else:
            self.acc.update_state(targets, preds)

    def get_metrics(self, validation=False):
        if validation:
            return {"Acc": self.val_acc.result()}
        return {"Acc": self.acc.result()}

    def reset_states(self, validation=False):
        if validation:
            self.val_acc.reset_states()
        else:
            self.acc.reset_states()

    def get_hash(self):
        return f"{self.val_acc.result():.4f}"

    def get_key(self):
        return self.val_acc.result()

    @staticmethod
    def parse_sst2_dataset(lines: List[List[str]], is_test: bool) -> Tuple[Optional[List[str]], List[str]]:
        """
        Parse SST-2 Dataset (GLUE)
        """
        # dataset files have a header row
        lines = lines[1:]
        if is_test:
            return None, [line[1] for line in lines]

        return [line[1] for line in lines], [line[0] for line in lines]


class RTEProcessor(GLUEClassificationProcessor):
    def __init__(self):
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    def get_train(self, path: str):
        return self.parse_rte_data(read_table(os.path.join(path, "train.tsv")), False)

    def get_dev(self, path: str):
        return self.parse_rte_data(read_table(os.path.join(path, "dev.tsv")), False)

    def get_test(self, path: str):
        return self.parse_rte_data(read_table(os.path.join(path, "test.tsv")), True)

    def get_label_to_index(self):
        return {"not_entailment": 0, "entailment": 1}

    @tf.function
    def update_state(self, targets, preds, validation=False):
        if validation:
            self.val_acc.update_state(targets, preds)
        else:
            self.acc.update_state(targets, preds)

    def get_metrics(self, validation=False):
        if validation:
            return {"Acc": self.val_acc.result()}
        return {"Acc": self.acc.result()}

    def reset_states(self, validation=False):
        if validation:
            self.val_acc.reset_states()
        else:
            self.acc.reset_states()

    def get_hash(self):
        return f"{self.val_acc.result():.4f}"

    def get_key(self):
        return self.val_acc.result()

    @staticmethod
    def parse_rte_data(lines: List[List[str]], is_test: bool) -> Tuple[Optional[List[str]], List[str], List[str]]:
        """
        Parse RTE Dataset (GLUE)
        """
        # dataset files have a header row
        lines = lines[1:]
        if is_test:
            return None, [line[1] for line in lines], [line[2] for line in lines]

        return [line[3] for line in lines], [line[1] for line in lines], [line[2] for line in lines]


class QQPProcessor(GLUEClassificationProcessor):
    def __init__(self):
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.f1 = F1Score()

        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_f1 = F1Score()

    def get_train(self, path: str):
        return self.parse_qqp_dataset(read_table(os.path.join(path, "train.tsv")), False)

    def get_dev(self, path: str):
        return self.parse_qqp_dataset(read_table(os.path.join(path, "dev.tsv")), False)

    def get_test(self, path: str):
        return self.parse_qqp_dataset(read_table(os.path.join(path, "train.tsv")), True)

    def get_label_to_index(self):
        return {"1": 1, "0": 0}

    @tf.function
    def update_state(self, targets, preds, validation=False):
        maxed_preds = tf.math.argmax(preds, -1)
        if validation:
            self.val_acc.update_state(targets, preds)
            self.val_f1.update_state(targets, maxed_preds)
        else:
            self.acc.update_state(targets, preds)
            self.f1.update_state(targets, maxed_preds)

    def get_metrics(self, validation=False):
        if validation:
            return {
                "Acc": self.val_acc.result(),
                "F1": self.val_f1.result(),
            }
        return {
            "Acc": self.acc.result(),
            "F1": self.f1.result(),
        }

    def reset_states(self, validation=False):
        if validation:
            self.val_acc.reset_states()
            self.val_f1.reset_states()
        else:
            self.acc.reset_states()
            self.f1.reset_states()

    def get_hash(self):
        return f"{self.val_acc.result():.4f}"

    def get_key(self):
        return self.val_acc.result()

    @staticmethod
    def parse_qqp_dataset(lines: List[List[str]], is_test: bool) -> Tuple[Optional[List[str]], List[str], List[str]]:
        """
        Parse QQP Dataset (GLUE)
        """
        # dataset files have a header row
        lines = [line for line in lines[1:] if len(line) == 6]
        if is_test:
            return None, [line[3] for line in lines], [line[4] for line in lines]

        return [line[-1] for line in lines], [line[3] for line in lines], [line[4] for line in lines]
