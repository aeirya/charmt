# pip install datasets tokenizers

import argparse
from pathlib import Path
import sys
from tokenizer import CharTokenizer, BpeTokenizer

# from datasets import load_dataset
# DATASET = "aeirya/lct-mt"


def iter_text_file(path, n=500_000):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == n: break

            line = line.strip()
            if line:
                yield line


def iter_stdin(limit=-1):
    for i, line in enumerate(sys.stdin):
        if i == limit: break

        text = line.strip()
        if text:
            yield text

def clean_kwargs(d):
    return {k: v for k, v in d.items() if v is not None}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="type", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out")
    common.add_argument("--max-len", type=int)
    common.add_argument("--min-freq", type=int)
    common.add_argument("--limit", type=int)

    p_char = sub.add_parser("char", parents=[common])
    p_char.set_defaults(type="char")

    p_bpe = sub.add_parser("bpe", parents=[common])
    p_bpe.add_argument("--vocab-size", type=int, default=8000)
    p_bpe.add_argument("--limit-alphabet", type=int, default=500)
    p_bpe.set_defaults(type="bpe")

    args = p.parse_args()

    # if args.type == "char":
    #     print("train char")
    # else:
    #     print("train bpe")


    texts = iter_stdin(limit=args.limit)

    init_kwargs = clean_kwargs({"max_len": args.max_len})

    if args.type in ["char"]:
        train_kwargs = clean_kwargs({"min_freq": args.min_freq})

        tok = CharTokenizer(**init_kwargs)
        tok.train(texts, **train_kwargs)
        if args.out:
            print("Saved char tokenizer.")

    elif args.type in ["bpe"]:
        bpe_train_kwargs = clean_kwargs({
            "min_freq": args.min_freq,
            "vocab_size": args.vocab_size,
            "limit_alphabet": args.limit_alphabet,
        })

        tok = BpeTokenizer(**init_kwargs)
        tok.train(
            texts,
            **bpe_train_kwargs,
        )
        if args.out:
            print("Saved BPE tokenizer.")
    
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tok.save(args.out)
    else:
        print(tok.save())


if __name__ == "__main__":
    main()