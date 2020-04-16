import os
from typing import Tuple, Optional, List, Dict

import torch
import tqdm

from dyna_bert import models
from dyna_bert import glue_processor
from dyna_bert import tokenizer
from dyna_bert import trainer
from dyna_bert.converter import load_tf_weight_to_pytorch_bert

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

    outputs = (labels, input_ids, token_type_ids, attention_mask)
    return tuple(torch.tensor(t) for t in outputs)


if __name__ == "__main__":
    parser = trainer.get_default_bert_argument_parser()
    args = parser.parse_args()

    assert args.task.lower() in TASKS, f"Supported Tasks: {', '.join(TASKS)}"

    assert os.path.exists(args.output), f"Output path {args.output} does not exists"
    assert os.path.exists(args.model + ".index"), f"Model path {args.model} does not exists"
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
    model = models.BertForClassification(bert_config, len(label_to_index))
    load_tf_weight_to_pytorch_bert(model.bert, bert_config, args.model)

    assert bert_config.vocab_size == len(vocab), "Actual vocab size and that in bert config are different."

    train_dataset = torch.utils.data.TensorDataset(*train_dataset)
    dev_dataset = torch.utils.data.TensorDataset(*dev_dataset)
    test_dataset = torch.utils.data.TensorDataset(*test_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch_index in range(args.epoch):
        model.train()

        for n_step, data in enumerate(tqdm.tqdm(train_dataloader)):
            optimizer.zero_grad()

            output = model(*data[1:])
            loss = criterion(output[0], data[0])

            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(args.output, f"{args.task}_model.pt"))
