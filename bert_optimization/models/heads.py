import tensorflow as tf
import tensorflow_addons as tfa

from . import models_utils
from .bert import BertConfig, BertModel
from .layer_normalization import LayerNormalization


class BertForClassification(tf.keras.Model):
    """
    Bert Model with Classification Layer

    Input Shape:
        input_ids: (Batch Size, Sequence Length)
        token_type_ids:: (Batch Size, Sequence Length)
        attention_mask:: (Batch Size, Sequence Length)
        head_mask: (Batch Size, Num Layers, Num Heads) -> https://arxiv.org/abs/1905.10650

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
        self.classifier = ClassificationHead(
            num_classes, bert_config.aware_quantization, bert_config.hidden_dropout_prob
        )

    def call(self, input_tensors, head_mask=None):
        bert_output = self.bert(input_tensors, head_mask=head_mask)
        logits = self.classifier(bert_output[1])

        return logits, bert_output


class BertForClassificationToQuant(tf.keras.Model):
    """
    Bert Model with Classification Layer to quantize

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
        self.classifier = ClassificationHead(
            num_classes, bert_config.aware_quantization, bert_config.hidden_dropout_prob
        )

    def call(self, input_tensors, head_mask=None):
        bert_output = self.bert(input_tensors, head_mask=head_mask)
        logits = self.classifier(bert_output[1])

        return logits


class ClassificationHead(tf.keras.layers.Layer):
    """
    Head for classification tasks

    Input Shape:
        x: (Batch Size, Hidden Size)

    Output Shape:
        logits: (Batch Size, Num Classes)
    """

    def __init__(self, num_classes: int, aware_quantization: bool, dropout: float = 0.9):
        super().__init__()

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classification_layer = models_utils.get_dense(aware_quantization)(num_classes)

    def call(self, x):
        x = self.dropout(x)
        return self.classification_layer(x)


class BertMLMHead(tf.keras.layers.Layer):
    """
    Masked LM Head used in Bert

    encoder_output Shape:
        pooled_output: (Batch Size, Sequence Length, Hidden Size)

    Output Shape:
        mlm_logit: (Batch Size, Sequence Length, Vocab Size)
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super(BertMLMHead, self).__init__()

        self.transform = tf.keras.layers.Dense(hidden_size)
        self.transform_layer_norm = LayerNormalization()

        self.output_layer = tf.keras.layers.Dense(vocab_size)

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
