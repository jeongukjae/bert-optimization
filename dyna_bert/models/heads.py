import torch
from torch import nn


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
