import tensorflow as tf

from bert_optimization.optimization.early_exit import calculate_uncertainty

from . import models_utils
from .bert import BertConfig
from .heads import ClassificationHead
from .transformer import TransformerEncoder


class EarlyExitBertModelForClassification(tf.keras.Model):
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

    def __init__(self, config: BertConfig, num_classes: int):
        super(EarlyExitBertModelForClassification, self).__init__()
        embedding_component = models_utils.get_embedding(config.aware_quantization)
        self.token_embeddings = embedding_component(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = embedding_component(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = embedding_component(config.max_position_embeddings, config.hidden_size)
        self.embedding_layer_norm = tf.keras.layers.LayerNormalization()
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

        self.branches = [
            ClassificationHead(num_classes, config.aware_quantization, config.hidden_dropout_prob)
            for _ in range(config.num_hidden_layers)
        ]

        self.pooler_layer = tf.keras.layers.Dense(config.hidden_size)
        self.num_layers = config.num_hidden_layers

    def call(self, input_tensors, speed=0.7, head_mask=None, training=None):
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

        if training is None:
            result = self._calculate_layer_output(0, hidden_state, attention_mask, speed, head_mask)
            for index in range(1, self.num_layers):
                result = tf.cond(
                    result[3],
                    lambda: self._calculate_layer_output(index, result[1], attention_mask, speed, head_mask),
                    lambda: result,
                )

            return result[0], result[2]

        branch_outputs = tuple()
        for index in range(self.num_layers):
            result = self._calculate_layer_output(index, hidden_state, attention_mask, speed, head_mask)
            hidden_state = result[1]
            branch_output = result[2]

            branch_outputs += (branch_output,)

        return branch_outputs

    def _calculate_layer_output(self, index, hidden_states, attention_mask, speed, head_mask):
        """
        Input:
            * index: layer index
            * hidden states: output of previous layer
            * attention mask
            * head mask

        Output:
            * index: layer index
            * hidden states: output of current layer
            * branch output: output of current layer's branch
        """
        hidden_states = self.encoders[index](hidden_states, mask=attention_mask, head_mask=head_mask)
        branch_output = tf.nn.tanh(self.pooler_layer(hidden_states[:, 0]))
        branch_output = self.branches[index](branch_output)
        branch_output = tf.nn.softmax(branch_output, -1)

        return index + 1, hidden_states, branch_output, tf.reduce_all(calculate_uncertainty(branch_output) > speed)
