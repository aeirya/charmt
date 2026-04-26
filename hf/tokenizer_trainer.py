import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, Split
from tokenizers.normalizers import NFC
from tokenizers.decoders import WordPiece
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing


SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def iter_stdin():
    for line in sys.stdin:
        text = line.strip()
        if text:
            yield text

def drop_none(kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def add_post_processor(tok):
    if hasattr(tok, 'token_to_id'):
        token_to_id = tok.token_to_id
    elif hasattr(tok, 'convert_tokens_to_ids'):
        token_to_id = tok.convert_tokens_to_ids

    tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", token_to_id("<bos>")),
            ("<eos>", token_to_id("<eos>")),
        ],
    )


def train_char(texts, min_freq=None, vocab_size=None, std_chars:list=None, model_max_length=2048):
    tok = Tokenizer(WordLevel(unk_token="<unk>"))
    tok.normalizer = NFC()
    tok.pre_tokenizer = Split(pattern="", behavior="isolated")
    
    trainer = WordLevelTrainer(
        special_tokens=SPECIAL,
        **drop_none({
            'vocab_size': vocab_size,
            'min_frequency': min_freq,
        })
    )
    tok.train_from_iterator(texts, trainer=trainer)
    
    if std_chars:
        tok.add_tokens(std_chars)

    add_post_processor(tok)
    
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=model_max_length,
    )


def train_bpe(texts, min_freq=None, vocab_size=None, limit_alphabet=None, model_max_length=128):
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = NFC()
    tok.pre_tokenizer = Whitespace()
    tok.decoder = WordPiece(prefix="##", cleanup=True)

    trainer = BpeTrainer(
        **drop_none({
            'vocab_size': vocab_size,
            'min_frequency': min_freq,
            'limit_alphabet': limit_alphabet,
        }),
        special_tokens=SPECIAL,
        continuing_subword_prefix="##",
        show_progress=True,
    )

    tok.train_from_iterator(texts, trainer=trainer)
    add_post_processor(tok)
    
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=model_max_length,
    )


def main():
    p = argparse.ArgumentParser()

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--char", action="store_true")
    group.add_argument("--bpe", action="store_true")

    p.add_argument("--out", type=str)
    p.add_argument("--min-freq", type=int, default=2)

    p.add_argument('--name', type=str, default='tokenizer')
    p.add_argument("--vocab-size", type=int)

    # Char-only
    p.add_argument('--latin-vocab', action="store_true")
    # BPE-only
    p.add_argument("--limit-alphabet", type=int, default=500)

    args = p.parse_args()

    texts = iter_stdin()

    out = args.out or 'tokenizers/{type}/{name}'.format(type='char' if args.char else 'bpe' if args.bpe else '', name=args.name)

    if args.char:
        if args.latin_vocab:
            alphabet = list('abcdefghijklmnopqrstuvwxyz1234567890.,?!()\'-_')
        else:
            alphabet = None

        tok = train_char(texts, min_freq=args.min_freq, vocab_size=args.vocab_size, std_chars=alphabet)
    else:
        tok = train_bpe(
            texts,
            min_freq=args.min_freq,
            vocab_size=args.vocab_size,
            limit_alphabet=args.limit_alphabet,
        )

    Path(out).mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)

    print(f"Saved tokenizer to {out}")


if __name__ == "__main__":
    main()