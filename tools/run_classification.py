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


def get_total_batches(dataset_size, batch_size):
    return dataset_size // batch_size + bool(dataset_size % batch_size)


@tf.function
def build_bert_model_graph(bert_model: models.BertForClassification, bert_config: models.BertConfig):
    token_ids = tf.keras.Input((None,), dtype=tf.int32)
    token_type_ids = tf.keras.Input((None,), dtype=tf.int32)
    attention_mask = tf.keras.Input((None,), dtype=tf.bool)

    bert_model(token_ids, token_type_ids, attention_mask, training=True)


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

    train_dataset = convert_single_sentence(train_dataset, label_to_index, tokenizer, args.max_sequence_length)
    dev_dataset = convert_single_sentence(dev_dataset, label_to_index, tokenizer, args.max_sequence_length)

    logger.info(f"Train Dataset Size: {len(train_dataset[0])}")
    logger.info(f"Dev Dataset Size: {len(dev_dataset[0])}")
    logger.info(f"Train Batches: {get_total_batches(len(train_dataset[0]), args.train_batch_size)}")
    logger.info(f"Dev Batches: {get_total_batches(len(dev_dataset[0]), args.eval_batch_size)}")

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(1000).batch(args.train_batch_size)
    dev_dataset = tf.data.Dataset.from_tensor_slices(dev_dataset).batch(args.eval_batch_size)

    logger.info("Initialize model")
    bert_config = models.BertConfig.from_json(args.config)
    logger.info("Model Config")
    for key, val in vars(bert_config).items():
        logger.info(f" - {key}: {val}")
    model = models.BertForClassification(bert_config, len(label_to_index))

    assert bert_config.vocab_size == len(vocab), "Actual vocab size and that in bert config are different."

    logger.info("Load Model Weights")
    build_bert_model_graph(model, bert_config)
    utils.load_bert_weights(args.model, model.bert, bert_config.use_splitted)

    logger.info("Initialize Optimizer and Loss function")
    optimizer = tfa.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    eval_loss = tf.keras.metrics.Mean(name="eval_loss")

    @tf.function
    def train_step(input_ids, token_type_ids, attention_mask, targets):
        with tf.GradientTape() as tape:
            preds, _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, training=True)
            loss = criterion(targets, preds)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss.update_state(loss)
        cola_processor.update_state(targets, preds)

    @tf.function
    def eval_step(input_ids, token_type_ids, attention_mask, targets):
        preds, _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = criterion(targets, preds)

        eval_loss.update_state(loss)
        cola_processor.update_state(targets, preds, validation=True)

    def eval_dev():
        eval_loss.reset_states()
        cola_processor.reset_states(validation=True)

        for targets, input_ids, token_type_ids, attention_mask in dev_dataset:
            eval_step(input_ids, token_type_ids, attention_mask, targets)

        logger.info(
            f"Eval] Epoch {epoch_index} "
            f"step: {step + 1} "
            f"loss: {eval_loss.result()}, "
            + f", ".join([f"{key}: {val}" for key, val in cola_processor.get_metrics(validation=True).items()])
        )

    logger.info("Start Training")
    for epoch_index in range(args.epoch):
        for step, (targets, input_ids, token_type_ids, attention_mask) in enumerate(train_dataset):
            train_step(input_ids, token_type_ids, attention_mask, targets)

            if (step + 1) % args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch_index} "
                    f"step: {step + 1}, "
                    f"loss: {train_loss.result()}, "
                    + f", ".join([f"{key}: {val}" for key, val in cola_processor.get_metrics().items()])
                )
                train_loss.reset_states()
                cola_processor.reset_states()

            if (step + 1) % args.val_interval == 0:
                eval_dev()

        eval_dev()
