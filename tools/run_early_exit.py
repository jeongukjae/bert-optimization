import logging
import os
import sys

import tensorflow as tf
import tensorflow_addons as tfa

from bert_optimization import glue_processor, models, tokenizer, utils
from bert_optimization.glue_processor import convert_sentence_pair, convert_single_sentence
from bert_optimization.optimizer.scheduler import BertScheduler

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
    train_dataset = dataset_processor.get_train(args.dataset)
    dev_dataset = dataset_processor.get_dev(args.dataset)

    if len(train_dataset) == 2:
        # single sentence dataset
        train_dataset = convert_single_sentence(train_dataset, label_to_index, tokenizer, args.max_sequence_length)
        dev_dataset = convert_single_sentence(dev_dataset, label_to_index, tokenizer, args.max_sequence_length)
    else:
        # sentence pair dataset
        train_dataset = convert_sentence_pair(train_dataset, label_to_index, tokenizer, args.max_sequence_length)
        dev_dataset = convert_sentence_pair(dev_dataset, label_to_index, tokenizer, args.max_sequence_length)

    logger.info(f"Train Dataset Size: {len(train_dataset[0])}")
    logger.info(f"Dev Dataset Size: {len(dev_dataset[0])}")
    train_batch_size = get_total_batches(len(train_dataset[0]), args.train_batch_size)
    logger.info(f"Train Batches: {train_batch_size}")
    logger.info(f"Dev Batches: {get_total_batches(len(dev_dataset[0]), args.eval_batch_size)}")

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(10000).batch(args.train_batch_size)
    dev_dataset = tf.data.Dataset.from_tensor_slices(dev_dataset).batch(args.eval_batch_size)

    logger.info("Initialize model")
    bert_config = models.BertConfig.from_json(args.config, aware_quantization=args.aware_quantization)
    logger.info("Model Config")
    for key, val in vars(bert_config).items():
        logger.info(f" - {key}: {val}")
    model = models.EarlyExitBertModelForClassification(bert_config, len(label_to_index))

    assert bert_config.vocab_size == len(vocab), "Actual vocab size and that in bert config are different."

    logger.info("Load Model Weights")
    build_bert_model_graph(model, bert_config)
    utils.load_bert_weights(args.model, model, bert_config.use_splitted)

    logger.info("Initialize Optimizer and Loss function")
    global_step = tf.Variable(0.0, trainable=False)
    scheduler = BertScheduler(args.warmup_ratio, train_batch_size * args.epoch)
    learning_rate = lambda: args.learning_rate * scheduler(global_step)
    weight_decay = lambda: args.weight_decay * args.learning_rate * scheduler(global_step)
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, epsilon=1e-06)

    excludes = ["layer_norm", "LayerNorm", "bias"]
    decay_var_list = [v for v in model.trainable_variables if all(term not in v.name for term in excludes)]
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    criterion_self = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    eval_loss = tf.keras.metrics.Mean(name="eval_loss")

    best_model_score = 0.0

    @tf.function
    def train_step(input_ids, token_type_ids, attention_mask, targets):
        with tf.GradientTape() as tape:
            preds = model([input_ids, token_type_ids, attention_mask], training=True)
            loss = sum([((len(preds) - index) ** 0.5) * criterion(targets, pred) for index, pred in enumerate(preds)])

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables), decay_var_list=decay_var_list)

        train_loss.update_state(loss)
        dataset_processor.update_state(targets, preds[-1])

    @tf.function
    def eval_step(input_ids, token_type_ids, attention_mask, targets):
        preds = model([input_ids, token_type_ids, attention_mask], speed=0.0)
        loss = criterion(targets, preds[1])

        eval_loss.update_state(loss)
        dataset_processor.update_state(targets, preds[1], validation=True)

    def eval_dev(best_model_score):
        eval_loss.reset_states()
        dataset_processor.reset_states(validation=True)

        for targets, input_ids, token_type_ids, attention_mask in dev_dataset:
            eval_step(input_ids, token_type_ids, attention_mask, targets)

        logger.info(
            f"[Eval] Epoch {epoch_index + 1} "
            f"loss: {eval_loss.result()}, "
            + ", ".join([f"{key}: {val}" for key, val in dataset_processor.get_metrics(validation=True).items()])
        )

        if dataset_processor.get_key() > best_model_score:
            logger.info("Reached Best Score.")
            model_path = f"{args.output}/checkpoints/model-{args.task}-{dataset_processor.get_hash()}-epoch{epoch_index}-step{int(global_step.numpy()) + 1}"
            model.save_weights(model_path)
            logger.info(f"Saved model in {model_path}")
            best_model_score = dataset_processor.get_key()

        return best_model_score

    logger.info("Start Training")
    for epoch_index in range(args.epoch):
        for step, (targets, input_ids, token_type_ids, attention_mask) in enumerate(train_dataset):
            train_step(input_ids, token_type_ids, attention_mask, targets)

            if (step + 1) % args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch_index + 1} "
                    f"step: {step + 1}, "
                    f"loss: {train_loss.result()}, "
                    + ", ".join([f"{key}: {val}" for key, val in dataset_processor.get_metrics().items()])
                )
                train_loss.reset_states()
                dataset_processor.reset_states()

            if (step + 1) % args.val_interval == 0:
                best_model_score = eval_dev(best_model_score)

            global_step.assign_add(1.0)

        logger.info(
            f"Epoch {epoch_index + 1} "
            f"loss: {train_loss.result()}, "
            + ", ".join([f"{key}: {val}" for key, val in dataset_processor.get_metrics().items()])
        )
        train_loss.reset_states()
        dataset_processor.reset_states()
        best_model_score = eval_dev(best_model_score)
