import tensorflow as tf
import tensorflow_addons as tfa


class TransformerEncoder(tf.keras.layers.Layer):
    """
    TransformerEncoder: https://arxiv.org/abs/1706.03762

    Input Shape:
        sequence: (Batch Size, Sequence Length, Hidden Size)
        attention_mask: (Batch Size, Sequence Length)
        head_mask: (Batch Size, Num Heads) -> https://arxiv.org/abs/1905.10650

    Output Shape:
        encoder_output: (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, num_heads, hidden_size, intermediate_size, dropout, activation):
        super().__init__()

        self.attention = ConcatenatedSelfAttention(num_heads, hidden_size, dropout=dropout)
        self.attention_dropout = tf.keras.layers.Dropout(dropout)
        self.attention_norm = tf.keras.layers.LayerNormalization()

        self.intermediate = tf.keras.layers.Dense(intermediate_size)
        self.intermediate_act = _get_activation_function(activation)
        self.intermediate_dropout = tf.keras.layers.Dropout(dropout)

        self.output_dense = tf.keras.layers.Dense(hidden_size)
        self.output_dropout = tf.keras.layers.Dropout(dropout)
        self.output_norm = tf.keras.layers.LayerNormalization()

    def call(self, sequence, mask, head_mask=None, training=False):
        sequence1 = self.attention_dropout(
            self.attention(sequence, mask=mask, head_mask=head_mask, training=training), training=training
        )
        sequence = self.attention_norm(sequence + sequence1)

        sequence1 = self.intermediate_dropout(self.intermediate_act(self.intermediate(sequence)), training=training)
        sequence1 = self.output_dropout(self.output_dense(sequence1), training=training)
        sequence = self.output_norm(sequence + sequence1)

        return sequence


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Multi Head Self Attention (MHA + Self Attention): https://arxiv.org/abs/1706.03762

    Input Shape:
        qkv: (Batch Size, Sequence Length, Hidden Size)
        attention_mask: (Batch Size, Sequence Length)

    Output Shape:
        attention_output: (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, num_heads, hidden_size, dropout):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size should be multiple of the # of attention heads")

        self.qkv_projection = tf.keras.layers.Dense(hidden_size * 3)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.output_dense = tf.keras.layers.Dense(hidden_size)

        self.scaling_factor = (hidden_size / num_heads) ** 0.5
        self.head_dims = int(hidden_size / num_heads)
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def call(self, qkv, mask, training=False):
        sequence_length = tf.shape(qkv)[1]

        query, key, value = tf.split(self.qkv_projection(qkv), 3, axis=-1)
        query *= self.scaling_factor

        # batch size, num heads, sequence length, head dims
        query = tf.reshape(query, [-1, sequence_length, self.num_heads, self.head_dims])
        key = tf.reshape(key, [-1, sequence_length, self.num_heads, self.head_dims])
        value = tf.reshape(value, [-1, sequence_length, self.num_heads, self.head_dims])

        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # batch size, num heads, sequence length, sequence length
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.where(tf.expand_dims(tf.expand_dims(mask, 1), 1), tf.constant(float("-inf")), scores)
        distributions = self.dropout(tf.nn.softmax(scores, -1), training=training)

        # batch size, sequence length, num heads, head dims
        attention_output = tf.transpose(tf.matmul(distributions, value), perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [-1, sequence_length, self.hidden_size])

        return self.output_dense(attention_output)


class SelfAttention(tf.keras.layers.Layer):
    """
    Vanilla Self Attention

    Input Shape:
        qkv: (Batch Size, Sequence Length, Input Size)
        attention_mask: (Batch Size, Sequence Length)

    Output Shape:
        attention_output: (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, hidden_size, dropout):
        super().__init__()

        self.qkv_projection = tf.keras.layers.Dense(hidden_size * 3)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.scaling_factor = hidden_size ** 0.5

    def call(self, qkv, mask, training=False):
        query, key, value = tf.split(self.qkv_projection(qkv), 3, axis=-1)
        query *= self.scaling_factor

        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.where(tf.expand_dims(mask, 1), tf.constant(float("-inf")), scores)
        distributions = self.dropout(tf.nn.softmax(scores, -1), training=training)

        return tf.matmul(distributions, value)


class ConcatenatedSelfAttention(tf.keras.layers.Layer):
    """
    Multi Head Self Attention using Single Head Self Attention

    Input Shape:
        qkv: (Batch Size, Sequence Length, Input Size)
        attention_mask: (Batch Size, Sequence Length)
        head_mask: (Batch Size, Num Heads) -> https://arxiv.org/abs/1905.10650

    Output Shape:
        attention_output: (Batch Size, Sequence Length, Hidden Size)
    """

    def __init__(self, num_heads, hidden_size, dropout):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size should be multiple of the # of attention heads")

        self.heads = [SelfAttention(int(hidden_size / num_heads), dropout) for _ in range(num_heads)]
        self.output_dense = tf.keras.layers.Dense(hidden_size)

        self.num_heads = num_heads

    def call(self, qkv, mask, head_mask=None, training=False):
        head_output = tf.concat([head(qkv, mask=mask, training=training) for head in self.heads], axis=-1)
        attn_output = self.output_dense(head_output)

        if head_mask is not None:
            attn_output_shape = tf.shape(attn_output)
            attn_output = tf.reshape(attn_output, (attn_output_shape[0], attn_output_shape[1], self.num_heads, -1))
            attn_output *= tf.expand_dims(tf.expand_dims(head_mask, 1), -1)
            attn_output = tf.reshape(attn_output, attn_output_shape)
        return attn_output


def _get_activation_function(activation):
    if activation == "gelu":
        return tfa.activations.gelu

    if activation == "relu":
        return tf.nn.relu

    raise ValueError("Activation Function should be a gelu or relu. Input: {}".format(activation))
