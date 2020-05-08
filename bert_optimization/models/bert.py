import json

import tensorflow as tf
import tensorflow_addons as tfa

from .heads import ClassificationHead
from .transformer import TransformerEncoder


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

    @staticmethod
    def from_json(path: str) -> "BertConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return BertConfig(**file_content)


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
        self.token_embeddings = tf.keras.layers.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = tf.keras.layers.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = tf.keras.layers.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_layer_norm = tf.keras.layers.LayerNormalization()
        self.embedding_dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        self.encoders = [
            TransformerEncoder(
                config.num_attention_heads,
                config.hidden_size,
                config.intermediate_size,
                config.attention_probs_dropout_prob,
                config.hidden_act,
                config.use_splitted,
            )
            for _ in range(config.num_hidden_layers)
        ]

        self.pooler_layer = tf.keras.layers.Dense(config.hidden_size)

        self.output_hidden_states = config.output_hidden_states
        self.output_embedding = config.output_embedding

    def call(self, input_ids, token_type_ids, attention_mask, head_mask=None, training=None):
        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(tf.constant(0), seq_length, tf.constant(1), dtype=tf.dtypes.int32)

        words_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings, training=training)

        hidden_state = embeddings
        hidden_states = tuple()
        for encoder in self.encoders:
            hidden_state = encoder(hidden_state, mask=attention_mask, head_mask=head_mask, training=training)
            if self.output_hidden_states:
                hidden_states += (hidden_state,)

        sequence_output = hidden_state
        pooled_output = tf.nn.tanh(self.pooler_layer(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output)
        if self.output_embedding:
            outputs += (embeddings,)
        if self.output_hidden_states:
            outputs += (hidden_states,)

        return outputs


class BertMLMHead(tf.keras.layers.Layer):
    """
    Masked LM Head used in Bert

    encoder_output Shape:
        pooled_output: (Batch Size, Sequence Length, Hidden Size)

    Output Shape:
        mlm_logit: (Batch Size, Sequence Length, Vocab Size)
    """

    def __init__(self, config: BertConfig):
        super(BertMLMHead, self).__init__()

        self.transform = tf.keras.layers.Dense(config.hidden_size)
        self.transform_layer_norm = tf.keras.layers.LayerNormalization()

        self.output_layer = tf.keras.layers.Dense(config.vocab_size)

    def call(self, encoder_output):
        transformed = tfa.activations.gelu(self.transform(encoder_output))
        transformed = self.transform_layer_norm(transformed)

        logits = self.output_layer(transformed)
        return tf.nn.softmax(logits, -1)


class BertNSPHead(tf.keras.layers.Layer):
    """
    NSP (Next Sentence Prediction) Head used in Bert

    Input Shape:
        pooled_output: (Batch Size, Hidden Size)

    Output Shape:
        nsp_logit: (Batch Size, 2)
    """

    def __init__(self):
        super(BertNSPHead, self).__init__()
        self.output_layer = tf.keras.layers.Dense(2)

    def call(self, pooled_output):
        return tf.nn.softmax(self.output_layer(pooled_output), axis=-1)


class BertForClassification(tf.keras.Model):
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
        self.classifier = ClassificationHead(num_classes, bert_config.hidden_dropout_prob)

    def call(self, input_ids, token_type_ids, attention_mask, head_mask=None, training=None):
        bert_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            training=training,
        )
        logits = self.classifier(bert_output[1])

        return logits, bert_output
