# pip install tokenizers

import json
import unicodedata
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class CustomTokenizer:
    unk_id = -1

    def __init__(self, max_len):
        self.max_len = max_len
        
        self.unk_id = self.token_id(UNK)
        self.pad_id = self.token_id(PAD)
        self.bos_id = self.token_id(BOS)
        self.eos_id = self.token_id(EOS)

    def token_id(self, token):
        return 3
    
    def get_token(self, id):
        return UNK

    def train(self, text, min_freq=0):
        return self

    def _encode(self, tokens):
        # default behavior
        return [self.token_id(tok) for tok in tokens]

    def encode(self, text, return_len=False):
        ids = [self.bos_id] + self._encode(text)[:self.max_len-2] + [self.eos_id]
        n = len(ids)
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        if return_len:
            return ids, n
        return ids
    
    def _decode(self, ids):
        return [self.get_token(i) for i in ids]

    def decode(self, ids):
        skip = [self.pad_id, self.bos_id, self.eos_id]
        ids = [i for i in ids if i not in skip]
        return self._decode(ids)


class CharTokenizer(CustomTokenizer):
    def __init__(self, max_len=256, itoc=None):
        self.max_len = max_len
        
        self.itoc = itoc or SPECIAL_TOKENS
        self.ctoi = {ch: i for i, ch in enumerate(self.itoc)}
        super().__init__(max_len)

    def token_id(self, token):
        return self.ctoi.get(token, self.unk_id)

    def get_token(self, id):
        return self.itoc.get(id, UNK)

    def train(self, texts, min_freq=3):
        counter = Counter()
        for text in texts:
            text = self.normalize(text)
            counter.update(text)

        chars = sorted(ch for ch, n in counter.items() if n >= min_freq)
        self.itoc += chars
        self.ctoi = {ch: i for i, ch in enumerate(self.itoc)}
        return self

    def normalize(self, text):
        return unicodedata.normalize("NFC", text)

    def _encode(self, text):
        return super()._encode(list(text))

    def encode(self, text, return_len=False):
        text = self.normalize(text)
        return super().encode(text, return_len)

    def _decode(self, ids):
        return ''.join(self.itoc[i] for i in ids)

    def save(self, path):
        data = {
            "max_len": self.max_len,
            "itoc": self.itoc,
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(itoc=data["itoc"], max_len=data["max_len"])


class BpeTokenizer(CustomTokenizer):
    def __init__(self, max_len, tok=None):
        tok = tok or Tokenizer(BPE(unk_token=UNK))
        tok.pre_tokenizer = Whitespace()
        self.tok = tok

        super().__init__(max_len)

    def token_id(self, token):
        return self.tok.token_to_id(token)

    def _encode(self, text):
        return self.tok.encode(text).ids

    def _decode(self, ids):
        return self.tok.decode(ids)

    def save(self, path):
        tok_path = 'bpe_tokenizer'
        data = {
            "max_len": self.max_len,
            "tok_path": tok_path,
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.tok.save(str(tok_path))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = Tokenizer.from_file(data['tok_path'])
        return cls(tok=tok, max_len=data["max_len"])

    def train(self, texts, min_freq=1, vocab_size=8_000, limit_alphabet=500):
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=SPECIAL_TOKENS,
            limit_alphabet=limit_alphabet,
        )
        self.tok.train_from_iterator(texts, trainer)
        return self


# def train_bpe_tokenizer(
#     texts,
#     vocab_size=8_000,
#     min_frequency=2,
#     limit_alphabet=500,
# ):
#     tok = Tokenizer(BPE(unk_token=UNK))
#     tok.pre_tokenizer = Whitespace()

#     trainer = BpeTrainer(
#         vocab_size=vocab_size,
#         min_frequency=min_frequency,
#         special_tokens=SPECIAL_TOKENS,
#         limit_alphabet=limit_alphabet,
#     )

#     tok.train_from_iterator(texts, trainer)
#     return tok


# def encode_bpe(tok, text, max_len=128):
#     bos_id = tok.token_to_id(BOS)
#     eos_id = tok.token_to_id(EOS)
#     pad_id = tok.token_to_id(PAD)

#     ids = [bos_id]
#     ids += tok.encode(text).ids[: max_len - 2]
#     ids += [eos_id]

#     return ids + [pad_id] * (max_len - len(ids))


# def decode_bpe(tok, ids):
#     skip = {
#         tok.token_to_id(PAD),
#         tok.token_to_id(BOS),
#         tok.token_to_id(EOS),
#     }
#     ids = [i for i in ids if i not in skip]
#     return tok.decode(ids)


# def save_bpe(tok, path):
#     tok.save(str(path))


# def load_bpe(path):
#     return Tokenizer.from_file(str(path))

tok = BpeTokenizer(20)
tok.train('rhisrhrh sample esgtrasxrrtt', min_freq=2)

out = tok.encode('sample test')
# print(out)


tok.save('path')
# tok = `BpeTokenizer.load('path')
tok = BpeTokenizer.load('path')
dec = tok.decode(out)
print(dec)
print(tok.max_len)
