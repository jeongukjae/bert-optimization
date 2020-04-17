import tensorflow as tf


class ClassificationHead(tf.keras.layers.Layer):
    """
    Head for classification tasks

    Input Shape:
        x: (Batch Size, Hidden Size)

    Output Shape:
        logits: (Batch Size, Num Classes)
    """

    def __init__(self, num_classes: int, dropout: float = 0.9):
        super().__init__()

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.classification_layer = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.dropout(x)
        return self.classification_layer(x)
