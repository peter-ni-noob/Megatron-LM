from datasets import _GPT2BPETokenizer

def build_tokenizer():
    from utils import get_args
    args=get_args()
    tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    return tokenizer