import json

import tensorflow as tf

from . import models_utils
from .transformer import TransformerEncoder
from .layer_normalization import LayerNormalization


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
        use_splitted: bool = False,
        aware_quantization: bool = False,
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
        self.use_splitted = use_splitted
        self.aware_quantization = aware_quantization

    @staticmethod
    def from_json(path: str, **kwargs) -> "BertConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return BertConfig(**file_content, **kwargs)


class BertModel(tf.keras.layers.Layer):
    """
    Base Bert Model: https://arxiv.org/abs/1810.04805

    Input Shape:
        input_ids: (Batch Size, Sequence Length)
        token_type_ids:: (Batch Size, Sequence Length)
        attention_mask:: (Batch Size, Sequence Length)
        head_mask: (Batch Size, Num Heads) -> https://arxiv.org/abs/1905.10650

    Output Shape:
        sequence_output: (Batch Size, Sequence Length, Hidden Size)
        pooled_output: (Batch Size, Hidden Size)
        embeddings: (Batch Size, Sequence Length, Hidden Size)
        hidden_states: (Num Layers, Batch Size, Sequence Length, Hidden Size)

    hidden_states is a "num layers"-length list of tensor that has shape of (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, config: BertConfig):
        super(BertModel, self).__init__()
        embedding_component = models_utils.get_embedding(config.aware_quantization)
        self.token_embeddings = embedding_component(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = embedding_component(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = embedding_component(config.max_position_embeddings, config.hidden_size)
        self.embedding_layer_norm = LayerNormalization()()
        self.embedding_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        self.encoders = [
            TransformerEncoder(
                config.num_attention_heads,
                config.hidden_size,
                config.intermediate_size,
                config.attention_probs_dropout_prob,
                config.hidden_act,
                config.aware_quantization,
                config.use_splitted,
            )
            for _ in range(config.num_hidden_layers)
        ]

        self.pooler_layer = tf.keras.layers.Dense(config.hidden_size)

        self.output_hidden_states = config.output_hidden_states
        self.output_embedding = config.output_embedding
        self.num_layers = config.num_hidden_layers

    def call(self, input_tensors, head_mask=None):
        assert len(input_tensors) == 3
        input_ids, token_type_ids, attention_mask = input_tensors

        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(tf.constant(0), seq_length, tf.constant(1), dtype=tf.dtypes.int32)

        words_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        hidden_state = embeddings

        hidden_states = tf.TensorArray(tf.float32, size=self.num_layers)
        for index in range(self.num_layers):
            hidden_state = self.encoders[index](hidden_state, mask=attention_mask, head_mask=head_mask)
            if self.output_hidden_states:
                hidden_states.write(index, hidden_state)

        sequence_output = hidden_state
        pooled_output = tf.nn.tanh(self.pooler_layer(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output)
        if self.output_embedding:
            outputs += (embeddings,)
        if self.output_hidden_states:
            outputs += (hidden_states.stack(),)

        return outputs
