from argparse import ArgumentParser


def get_default_bert_argument_parser():
    """Get Default ArgumentParser for BERT downstream tasks"""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="pretrained bert model path", required=True)
    parser.add_argument("--config", type=str, help="bert config path", required=True)
    parser.add_argument("--output", type=str, help="output directory", required=True)
    parser.add_argument("--dataset", type=str, help="data path", required=True)
    parser.add_argument("--vocab", type=str, help="vocab path", required=True)

    parser.add_argument("--task", type=str, help="task name to train", required=True)
    parser.add_argument("--use-gpu", action="store_true", help="whether to use gpu")

    parser.add_argument("--epoch", type=int, default=3, help="num epoch")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="initial learing rate")
    parser.add_argument("--do-lower-case", action="store_true", help="whether to do lower case")
    parser.add_argument("--max-sequence-length", type=int, default=128, help="max sequence length of input")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="batch size for eval")
    parser.add_argument("--train-batch-size", type=int, default=32, help="batch size for training")
    parser.add_argument("--warmup-rate", type=float, default=0.1, help="rate of trainig data to use for warm up")

    parser.add_argument("--log-interval", type=int, default=50, help="interval to log")
    parser.add_argument("--val-interval", type=int, default=1000, help="interval to validate model")
    parser.add_argument("--save-interval", type=int, default=1000, help="interval to save model")

    return parser
