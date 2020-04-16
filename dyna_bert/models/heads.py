from typing import Optional

import torch
from torch import nn

from dyna_bert.models.bert import BertConfig, BertModel


class ClassificationHead(nn.Module):
    """
    Head for classification tasks

    Input Shape:
        x: (Batch Size, Hidden Size)

    Output Shape:
        logits: (Batch Size, Num Classes)
    """

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.9):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.classification_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.classification_layer(x)


class BertForClassification(nn.Module):
    """
    Bert Model with Classification Layer

    Input Shape:
        input_ids: (Batch Size, Sequence Length)
        token_type_ids:: (Batch Size, Sequence Length)
        attention_mask:: (Batch Size, Sequence Length)
        head_mask: (Batch Size, Num Heads) -> https://arxiv.org/abs/1905.10650

    Output Shape:
        logits: (Batch Size, Num Classes)
        bert_output
            sequence_output: (Batch Size, Sequence Length, Hidden Size)
            pooled_output: (Batch Size, Hidden Size)
            embeddings: (Batch Size, Sequence Length, Hidden Size)
            hidden_states: (Num Layers, Batch Size, Sequence Length, Hidden Size)

    hidden_states is a "num layers"-length list of tensor that has shape of (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, bert_config: BertConfig, num_classes: int):
        super().__init__()

        self.bert = BertModel(bert_config)
        self.classifier = ClassificationHead(bert_config.hidden_size, num_classes, bert_config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
    ):
        bert_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask)
        logits = self.classifier(bert_output[1])

        return logits, bert_output
