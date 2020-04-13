from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder: https://arxiv.org/abs/1706.03762

    Input Shape:
        sequence: (Batch Size, Sequence Length, Hidden Size)
        attention_mask: (Batch Size, Sequence Length)

    Output Shape:
        encoder_output: (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(
        self, num_heads: int, hidden_size: int, intermediate_size: int, dropout: float, activation: str,
    ):
        super().__init__()

        self.attention = ConcatenatedSelfAttention(num_heads, hidden_size, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act = _get_activation_function(activation)
        self.intermediate_dropout = nn.Dropout(dropout)

        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, sequence: torch.Tensor, attention_mask: torch.Tensor):
        sequence1 = self.attention_dropout(self.attention(sequence, attention_mask))
        sequence = self.attention_norm(sequence + sequence1)

        sequence1 = self.intermediate_dropout(self.intermediate_act(self.intermediate(sequence)))
        sequence1 = self.output_dropout(self.output(sequence1))
        sequence = self.output_norm(sequence + sequence1)

        return sequence


class MultiHeadSelfAttention(nn.Module):
    """
    Multi Head Self Attention (MHA + Self Attention): https://arxiv.org/abs/1706.03762

    Input Shape:
        qkv: (Batch Size, Sequence Length, Hidden Size)
        attention_mask: (Batch Size, Sequence Length)

    Output Shape:
        attention_output: (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, num_heads: int, hidden_size: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size should be multiple of the # of attention heads")

        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)

        self.scaling_factor = (hidden_size / num_heads) ** 0.5
        self.head_dims = int(hidden_size / num_heads)
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor, attention_mask: torch.Tensor):
        sequence_length = qkv.size(1)

        query, key, value = self.qkv_linear(qkv).chunk(3, dim=-1)
        query *= self.scaling_factor

        # batch size, num heads, sequence length, head dims
        query = query.contiguous().view(-1, sequence_length, self.num_heads, self.head_dims).transpose(1, 2)
        key = key.contiguous().view(-1, sequence_length, self.num_heads, self.head_dims).transpose(1, 2)
        value = value.contiguous().view(-1, sequence_length, self.num_heads, self.head_dims).transpose(1, 2)

        # batch size, num heads, sequence length, sequence length
        scores = torch.matmul(query, key.transpose(2, 3))
        scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        distributions = self.dropout(F.softmax(scores, -1))

        # batch size, sequence length, num heads, head dims
        attention_output = torch.matmul(distributions, value).transpose(1, 2).contiguous()
        attention_output = attention_output.view(-1, sequence_length, self.hidden_size)

        return self.output(attention_output)


class SelfAttention(nn.Module):
    """
    Vanilla Self Attention

    Input Shape:
        qkv: (Batch Size, Sequence Length, Input Size)
        attention_mask: (Batch Size, Sequence Length)

    Output Shape:
        attention_output: (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()

        self.qkv_linear = nn.Linear(input_size, hidden_size * 3)
        self.dropout = nn.Dropout(dropout)
        self.scaling_factor = hidden_size ** 0.5

    def forward(self, qkv: torch.Tensor, attention_mask: torch.Tensor):
        query, key, value = self.qkv_linear(qkv).chunk(3, dim=-1)
        query *= self.scaling_factor

        scores = torch.matmul(query, key.transpose(1, 2))
        scores = scores.masked_fill(attention_mask.unsqueeze(1), float("-inf"))
        distributions = self.dropout(F.softmax(scores, -1))

        return torch.matmul(distributions, value)


class ConcatenatedSelfAttention(nn.Module):
    """
    Multi Head Self Attention using Single Head Self Attention

    Input Shape:
        qkv: (Batch Size, Sequence Length, Input Size)
        attention_mask: (Batch Size, Sequence Length)

    Output Shape:
        attention_output: (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, num_heads: int, hidden_size: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size should be multiple of the # of attention heads")

        self.heads = nn.ModuleList(
            [SelfAttention(hidden_size, int(hidden_size / num_heads), dropout) for _ in range(num_heads)]
        )
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, qkv: torch.Tensor, attention_mask: torch.Tensor):
        concatenated_head_outputs = torch.cat([head(qkv, attention_mask) for head in self.heads], dim=-1)
        attention_output = self.output(concatenated_head_outputs)
        return attention_output


def _get_activation_function(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "gelu":
        return F.gelu

    if activation == "relu":
        return F.relu

    raise ValueError("Activation Function should be a gelu or relu. Input: {}".format(activation))
