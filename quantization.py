import pathlib
import tensorflow as tf

from bert_optimization.models.heads import BertForClassificationToQuant
from bert_optimization.models import BertConfig


@tf.function
def build_bert_model_graph(bert_model: BertForClassificationToQuant, bert_config: BertConfig):
    token_ids = tf.keras.Input((48,), dtype=tf.int32)
    token_type_ids = tf.keras.Input((48,), dtype=tf.int32)
    attention_mask = tf.keras.Input((48,), dtype=tf.float32)

    bert_model([token_ids, token_type_ids, attention_mask])


bert_config = BertConfig.from_json("./tmp/bert_config.json")
model = tf.keras.models.Sequential([BertForClassificationToQuant(bert_config, 2)])
build_bert_model_graph(model, bert_config)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quant = converter.convert()

tflite_models_dir = pathlib.Path("./tmp/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir / "bert_model.tflite"
print(tflite_model_file.write_bytes(tflite_model_quant))
print(tflite_model_file)
