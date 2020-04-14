from dyna_bert import glue_processor
from dyna_bert import tokenizer
from dyna_bert import trainer


if __name__ == "__main__":
    parser = trainer.get_default_bert_argument_parser()
    parser.parse_args()
