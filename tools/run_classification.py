import sys
import logging
import os
from typing import Tuple, Optional, List, Dict

import tensorflow as tf
import tensorflow_addons as tfa

from dyna_bert import models
from dyna_bert import glue_processor
from dyna_bert import tokenizer
from dyna_bert import utils

TASKS = ["cola"]


def convert_single_sentence(
    data: Tuple[Optional[List[str]], List[str]],
    label_to_index: Dict[str, int],
    tokenizer: tokenizer.SubWordTokenizer,
    max_length: int,
):
    labels = [0] * len(data[1]) if data[0] is None else [label_to_index[label] for label in data[0]]
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for example in data[1]:
        tokens = tokenizer.tokenize(example)[: max_length - 2]
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
        padding_size = max_length - len(ids)

        input_ids.append(ids + [0] * padding_size)
        token_type_ids.append([0] * max_length)
        attention_mask.append([False] * len(ids) + [True] * padding_size)

    return (labels, input_ids, token_type_ids, attention_mask)


@tf.function
def build_bert_model_graph(bert_model: models.BertForClassification, bert_config: models.BertConfig):
    token_ids = tf.constant([[1] * bert_config.max_position_embeddings])
    token_type_ids = tf.constant([[0] * bert_config.max_position_embeddings])
    attention_mask = tf.constant([[False] * bert_config.max_position_embeddings])

    bert_model(token_ids, token_type_ids, attention_mask)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s: [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    parser = utils.get_default_bert_argument_parser()
    args = parser.parse_args()

    logger.info("Training Parameters")
    for key, val in vars(args).items():
        logger.info(f" - {key}: {val}")

    assert args.task.lower() in TASKS, f"Supported Tasks: {', '.join(TASKS)}"

    assert os.path.exists(args.output), f"Output path {args.output} does not exists"
    assert os.path.exists(args.model + ".index"), f"Model path {args.model} does not exists"
    assert os.path.exists(args.config), f"Config path {args.config} does not exists"
    assert os.path.exists(args.dataset), f"Dataset path {args.dataset} does not exists"
    assert os.path.exists(args.vocab), f"Vocab path {args.vocab} does not exists"

    vocab = tokenizer.Vocab(args.vocab)
    tokenizer = tokenizer.SubWordTokenizer(vocab, args.do_lower_case)

    logger.info("Processing Data")
    cola_processor = glue_processor.CoLAProcessor()
    label_to_index = cola_processor.get_label_to_index()
    train_dataset = cola_processor.get_train(args.dataset)
    dev_dataset = cola_processor.get_dev(args.dataset)
    # test_dataset = cola_processor.get_test(args.dataset)

    train_dataset = convert_single_sentence(train_dataset, label_to_index, tokenizer, args.max_sequence_length)
    dev_dataset = convert_single_sentence(dev_dataset, label_to_index, tokenizer, args.max_sequence_length)
    # test_dataset = convert_single_sentence(test_dataset, label_to_index, tokenizer, args.max_sequence_length)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(1000).batch(args.train_batch_size)

    logger.info("Initialize model")
    bert_config = models.BertConfig.from_json(args.config)
    logger.info("Model Config")
    for key, val in vars(bert_config).items():
        logger.info(f" - {key}: {val}")
    model = models.BertForClassification(bert_config, len(label_to_index))

    assert bert_config.vocab_size == len(vocab), "Actual vocab size and that in bert config are different."

    logger.info("Load Model Weights")
    build_bert_model_graph(model, bert_config)
    utils.load_bert_weights(args.model, model.bert)

    logger.info("Initialize Optimizer, Loss function, and Metrics")
    optimizer = tfa.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_mcc = tfa.metrics.MatthewsCorrelationCoefficient(name="train_mcc", num_classes=1)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    @tf.function
    def train_step(input_ids, token_type_ids, attention_mask, targets):
        with tf.GradientTape() as tape:
            preds, _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, training=True)
            loss = criterion(targets, preds)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(targets, preds)
        train_mcc(tf.expand_dims(targets, 1), tf.expand_dims(tf.argmax(preds, -1), 1))

    logger.info("Start Training")
    for epoch_index in range(args.epoch):
        logger.info(f"Epoch {epoch_index}")
        for step, (targets, input_ids, token_type_ids, attention_mask) in enumerate(train_dataset):
            train_step(input_ids, token_type_ids, attention_mask, targets)

            logger.info(
                f"step: {step+ 1}, loss: {train_loss.result()}, acc: {train_accuracy.result()}, MCC: {train_mcc.result()}"
            )
