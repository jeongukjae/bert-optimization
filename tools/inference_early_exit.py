import logging
import os
import sys

import tensorflow as tf

from bert_optimization import glue_processor, models, tokenizer, utils
from bert_optimization.glue_processor import convert_sentence_pair, convert_single_sentence

PROCESSOR_BY_TASK = {
    "cola": glue_processor.CoLAProcessor,
    "mrpc": glue_processor.MRPCProcessor,
    "mnli": glue_processor.MNLIProcessor,
    "sst-2": glue_processor.SST2Processor,
    "rte": glue_processor.RTEProcessor,
    "qqp": glue_processor.QQPProcessor,
}


def get_total_batches(dataset_size, batch_size):
    return dataset_size // batch_size + bool(dataset_size % batch_size)


@tf.function
def build_bert_model_graph(bert_model: models.EarlyExitBertModelForClassification, bert_config: models.BertConfig):
    token_ids = tf.keras.Input((None,), dtype=tf.int32)
    token_type_ids = tf.keras.Input((None,), dtype=tf.int32)
    attention_mask = tf.keras.Input((None,), dtype=tf.float32)

    bert_model([token_ids, token_type_ids, attention_mask], speed=0.7, training=True)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s: [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    parser = utils.get_default_bert_argument_parser()
    args = parser.parse_args()
    args.eval_batch_size = 1

    logger.info("Training Parameters")
    for key, val in vars(args).items():
        logger.info(f" - {key}: {val}")

    assert args.task.lower() in PROCESSOR_BY_TASK, f"Supported Tasks: {', '.join(PROCESSOR_BY_TASK.keys())}"

    assert os.path.exists(args.output), f"Output path {args.output} does not exists"
    assert os.path.exists(args.model + ".index"), f"Model path {args.model} does not exists"
    assert os.path.exists(args.config), f"Config path {args.config} does not exists"
    assert os.path.exists(args.dataset), f"Dataset path {args.dataset} does not exists"
    assert os.path.exists(args.vocab), f"Vocab path {args.vocab} does not exists"

    vocab = tokenizer.Vocab(args.vocab)
    tokenizer = tokenizer.SubWordTokenizer(vocab, args.do_lower_case)

    logger.info("Processing Data")
    dataset_processor = PROCESSOR_BY_TASK[args.task.lower()]()
    label_to_index = dataset_processor.get_label_to_index()
    dev_dataset = dataset_processor.get_dev(args.dataset)

    if len(dev_dataset) == 2:
        # single sentence dataset
        dev_dataset = convert_single_sentence(dev_dataset, label_to_index, tokenizer, args.max_sequence_length)
    else:
        # sentence pair dataset
        dev_dataset = convert_sentence_pair(dev_dataset, label_to_index, tokenizer, args.max_sequence_length)

    logger.info(f"Dev Dataset Size: {len(dev_dataset[0])}")
    logger.info(f"Dev Batches: {get_total_batches(len(dev_dataset[0]), args.eval_batch_size)}")

    dev_dataset = tf.data.Dataset.from_tensor_slices(dev_dataset).batch(args.eval_batch_size)

    logger.info("Initialize model")
    bert_config = models.BertConfig.from_json(args.config, aware_quantization=args.aware_quantization)
    logger.info("Model Config")
    for key, val in vars(bert_config).items():
        logger.info(f" - {key}: {val}")
    model = models.EarlyExitBertModelForClassification(bert_config, len(label_to_index))

    assert bert_config.vocab_size == len(vocab), "Actual vocab size and that in bert config are different."

    logger.info("Load Model Weights")
    model.load_weights(args.model)

    logger.info("Initialize Loss function")
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    eval_loss = tf.keras.metrics.Mean(name="eval_loss")

    @tf.function
    def eval_step(input_ids, token_type_ids, attention_mask, targets, speed):
        preds = model([input_ids, token_type_ids, attention_mask], speed=speed)
        loss = criterion(targets, preds[1])

        eval_loss.update_state(loss)
        dataset_processor.update_state(targets, preds[1], validation=True)

        return tf.cast(preds[0], tf.float32)

    def eval_dev(speed: float):
        eval_loss.reset_states()
        dataset_processor.reset_states(validation=True)

        layers = []
        for targets, input_ids, token_type_ids, attention_mask in dev_dataset:
            layers.append(eval_step(input_ids, token_type_ids, attention_mask, targets, speed))

        logger.info(
            f"[Eval] "
            f"loss: {eval_loss.result()}, "
            + ", ".join([f"{key}: {val}" for key, val in dataset_processor.get_metrics(validation=True).items()])
        )
        logger.info(f"Average Used Layers: {tf.reduce_mean(layers)}")

    logger.info("Start Inference")
    for i in range(11):
        logger.info(f"Speed {i * 0.1}")
        eval_dev(i * 0.1)
