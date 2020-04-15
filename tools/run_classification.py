import os
from typing import Tuple, Optional, List, Dict

from dyna_bert import models
from dyna_bert import glue_processor
from dyna_bert import tokenizer
from dyna_bert import trainer

TASKS = ["cola"]


def convert_single_sentence(
    data: Tuple[Optional[List[str]], List[str]],
    label_to_index: Dict[str, int],
    tokenizer: tokenizer.SubWordTokenizer,
    max_length: int,
):
    labels = None if data[0] is None else [label_to_index[label] for label in data[0]]
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for example in data[1]:
        tokens = tokenizer.tokenize(example)
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])

        input_ids.append(ids)
        attention_mask.append([False] * len(ids))
        token_type_ids.append([0] * len(ids))

    return labels, input_ids, attention_mask, token_type_ids


if __name__ == "__main__":
    parser = trainer.get_default_bert_argument_parser()
    args = parser.parse_args()

    assert args.task.lower() in TASKS, f"Supported Tasks: {', '.join(TASKS)}"

    assert os.path.exists(args.output), f"Output path {args.output} does not exists"
    assert os.path.exists(args.model), f"Model path {args.model} does not exists"
    assert os.path.exists(args.config), f"Config path {args.config} does not exists"
    assert os.path.exists(args.dataset), f"Dataset path {args.dataset} does not exists"
    assert os.path.exists(args.vocab), f"Vocab path {args.vocab} does not exists"

    vocab = tokenizer.Vocab(args.vocab)
    tokenizer = tokenizer.SubWordTokenizer(vocab, args.do_lower_case)

    cola_processor = glue_processor.CoLAProcessor()
    label_to_index = cola_processor.get_label_to_index()
    train_dataset = cola_processor.get_train(args.dataset)
    dev_dataset = cola_processor.get_dev(args.dataset)
    test_dataset = cola_processor.get_test(args.dataset)

    train_dataset = convert_single_sentence(train_dataset, label_to_index, tokenizer, args.max_sequence_length)
    dev_dataset = convert_single_sentence(dev_dataset, label_to_index, tokenizer, args.max_sequence_length)
    test_dataset = convert_single_sentence(test_dataset, label_to_index, tokenizer, args.max_sequence_length)

    bert_config = models.BertConfig.from_json(args.config)
    model = models.BertModel(bert_config)

    assert bert_config.vocab_size == len(vocab), "Actual vocab size and that in bert config are different."
