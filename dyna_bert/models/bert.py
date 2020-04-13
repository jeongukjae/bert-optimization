import json

import torch
from torch import nn
from torch.nn import functional as F


class BertConfig:
    """
    Configuration of BertModel

    vocab_size: Size of Vocab
    hidden_size: Hidden size used in embedding, MHA, pooling layers
    num_hidden_layers: # of transformer encoder layer
    num_attention_heads: # of attention heads in MHA
    intermediate_size: Intermediate size used in MHA
    hidden_act: Activation function used in transformer encoder layer
    hidden_dropout_prob: Dropout prob
    attention_probs_dropout_prob: Attention Dropout prob
    max_position_embeddings: Max Position Embeddings
    type_vocab_size: Vocab Type (2 => Sentence A/B)
    output_hidden_states: A flag for BertModel to return hidden_states
    output_embedding: A flag for BertModel to return embedding
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        output_hidden_states: bool = True,
        output_embedding: bool = True,
        **kwargs,  # unused
    ):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.output_hidden_states = output_hidden_states
        self.output_embedding = output_embedding

    @staticmethod
    def from_json(path: str) -> "BertConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return BertConfig(**file_content)


class BertModel(nn.Module):
    """
    Base Bert Model: https://arxiv.org/abs/1810.04805

    Input Shape:
        input_ids: (Batch Size, Sequence Length)
        token_type_ids:: (Batch Size, Sequence Length)
        attention_mask:: (Batch Size, Sequence Length)

    Output Shape:
        sequence_output: (Batch Size, Sequence Length, Hidden Size)
        pooled_output: (Batch Size, Hidden Size)
        embeddings: (Batch Size, Sequence Length, Hidden Size)
        hidden_states: (Num Layers, Batch Size, Sequence Length, Hidden Size)

    hidden_states is a "num layers"-length list of tensor that has shape of (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, config: BertConfig):
        super(BertModel, self).__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.encoders = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.attention_probs_dropout_prob,
                    activation=config.hidden_act,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.pooler_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooled_output_activate = nn.Tanh()

        self.output_hidden_states = config.output_hidden_states
        self.output_embedding = config.output_embedding

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor,
    ):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

        words_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        hidden_state = embeddings.permute(1, 0, 2)
        hidden_states = []
        for encoder in self.encoders:
            hidden_state = encoder(hidden_state, src_key_padding_mask=attention_mask)
            if self.output_hidden_states:
                hidden_states.append(hidden_state.permute(1, 0, 2))

        sequence_output = hidden_state.permute(1, 0, 2)
        pooled_output = self.pooled_output_activate(self.pooler_layer(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output)
        if self.output_embedding:
            outputs += (embeddings,)
        if self.output_hidden_states:
            outputs += (hidden_states,)

        return outputs


class BertMLMHead(nn.Module):
    """
    Masked LM Head used in Bert

    encoder_output Shape:
        pooled_output: (Batch Size, Sequence Length, Hidden Size)
    Output Shape:
        mlm_logit: (Batch Size, Sequence Length, Vocab Size)
    """

    def __init__(self, config: BertConfig):
        super(BertMLMHead, self).__init__()

        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_layer_norm = nn.LayerNorm(config.hidden_size)

        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        transformed = F.gelu(self.transform(encoder_output))
        transformed = self.transform_layer_norm(transformed)

        logits = self.output_layer(transformed)
        return F.softmax(logits + self.output_bias, -1)


class BertNSPHead(nn.Module):
    """
    NSP (Next Sentence Prediction) Head used in Bert

    Input Shape:
        pooled_output: (Batch Size, Hidden Size)
    Output Shape:
        nsp_logit: (Batch Size, 2)
    """

    def __init__(self, config: BertConfig):
        super(BertNSPHead, self).__init__()
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.output_layer(pooled_output), dim=-1)
