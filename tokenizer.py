# pip install tokenizers

import json
import unicodedata
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class CustomTokenizer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.__init_special_tokens__()
        
    def __init_special_tokens__(self):
        self.unk_id = self.token_id(UNK)
        self.pad_id = self.token_id(PAD)
        self.bos_id = self.token_id(BOS)
        self.eos_id = self.token_id(EOS)

    def encode(self, text, return_len=False):
        '''
        appends BOS, EOS, PAD tokens, and calls __encode__
        
        :param text: input string
        :param return_len: if True, returns a tuple (ids, n), where n is the length of the unpadded token list.
        '''
        ids = [self.bos_id] + self.__encode__(text)[:self.max_len-2] + [self.eos_id]
        n = len(ids)
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        if return_len:
            return ids, n
        return ids
    
    def decode(self, ids):
        '''
        calls __decode__, skipping special tokens
        '''
        skip = [self.pad_id, self.bos_id, self.eos_id]
        ids = [i for i in ids if i not in skip]
        return self.__decode__(ids)
    
    def token_id(self, token):
        pass
    
    def get_token(self, id):
        pass

    def train(self, text, min_freq=0):
        pass

    def __encode__(self, text):
        pass
    
    def __decode__(self, ids):
        pass


class CharTokenizer(CustomTokenizer):
    def __init__(self, max_len=256, itoc=None):
        self.max_len = max_len
        
        self.itoc = itoc or SPECIAL_TOKENS
        self.ctoi = {ch: i for i, ch in enumerate(self.itoc)}
        self.unk = '*'
        super().__init__(max_len)

    def token_id(self, token):
        return self.ctoi.get(token, self.ctoi.get(self.unk, 3))

    def get_token(self, id):
        if id == self.unk_id:
            return self.unk
        return self.itoc[id]

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

    def __encode__(self, text):
        return [self.token_id(c) for c in text]

    def encode(self, text, return_len=False):
        text = self.normalize(text)
        return super().encode(text, return_len)

    def __decode__(self, ids):
        return ''.join(self.get_token(i) for i in ids)

    def save(self, path=None):
        data = {
            "max_len": self.max_len,
            "itoc": self.itoc,
        }
        text = json.dumps(data, ensure_ascii=False, indent=2)
        
        if path:
            Path(path).write_text(
                text,
                encoding="utf-8",
            )
        else:
            return text

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(itoc=data["itoc"], max_len=data["max_len"])


class BpeTokenizer(CustomTokenizer):
    def __init__(self, max_len, tok=None):
        self.tok = tok or self.__init_tokenizer__()
        super().__init__(max_len)

    def token_id(self, token):
        return self.tok.token_to_id(token)

    def __encode__(self, text):
        return self.tok.encode(text).ids

    def __decode__(self, ids):
        return self.tok.decode(ids)

    def save(self, path=None):
        tok_path = 'bpe_tokenizer'
        data = {
            "max_len": self.max_len
        }
        if path is None:
            data['bpe'] = self.tok.to_str(True)
        else:
            data['tok_path'] = tok_path
        
        text = json.dumps(data, ensure_ascii=False, indent=2)

        if path is None:
            return text
        else:
            self.tok.save(str(tok_path))
            Path(path).write_text(
                text,
                encoding="utf-8",
            )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = Tokenizer.from_file(data['tok_path'])
        return cls(tok=tok, max_len=data["max_len"])

    def __init_tokenizer__(self):
        tok = Tokenizer(BPE(unk_token=UNK))
        tok.pre_tokenizer = Whitespace()
        tok.decoder = WordPiece(prefix="##", cleanup=True)
        tok.add_special_tokens(SPECIAL_TOKENS)
        return tok

    def train(self, texts, min_freq=1, vocab_size=8_000, limit_alphabet=500):
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=SPECIAL_TOKENS,
            limit_alphabet=limit_alphabet,
            continuing_subword_prefix='##',
            show_progress=True,
        )
        self.tok.train_from_iterator(texts, trainer)
        return self

