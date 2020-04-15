import csv
import os
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import List, Optional, Tuple


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


class CoLAProcessor(GLUEClassificationProcessor):
    def get_train(self, path: str):
        return self.parse_cola_dataset(read_table(os.path.join(path, "train.tsv")), False)

    def get_dev(self, path: str):
        return self.parse_cola_dataset(read_table(os.path.join(path, "dev.tsv")), False)

    def get_test(self, path: str):
        return self.parse_cola_dataset(read_table(os.path.join(path, "train.tsv")), True)

    def get_label_to_index(self):
        return {"1": 1, "0": 0}

    @staticmethod
    def parse_cola_dataset(lines: List[List[str]], is_test: bool) -> Tuple[Optional[List[str]], List[str]]:
        """
        Parse CoLA Dataset (GLUE)
        """
        if is_test:
            # test.tsv has header row
            return None, [line[1] for line in lines[1:]]

        return [line[1] for line in lines], [line[3] for line in lines]