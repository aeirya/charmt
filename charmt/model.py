import torch
import torch.nn as nn

import math
from charmt.config import ModelConfig, TrainConfig
from dataclasses import replace


class TinySeq2SeqTransformer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c

        self.src_emb = nn.Embedding(c.src_vocab, c.dim, padding_idx=c.src_pad_id)
        self.tgt_emb = nn.Embedding(c.tgt_vocab, c.dim, padding_idx=c.tgt_pad_id)
        
        self.src_pos_emb = nn.Embedding(c.src_max_len, c.dim)
        self.tgt_pos_emb = nn.Embedding(c.tgt_max_len, c.dim)

        self.dropout = nn.Dropout(c.dropout)

        self.lm_head = nn.Linear(c.dim, c.tgt_vocab)

        self.transformer = nn.Transformer(
            d_model=c.dim,
            nhead=c.nhead,
            num_encoder_layers=c.n_encoder_layers,
            num_decoder_layers=c.n_decoder_layers,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            batch_first=True,
        )

    @staticmethod
    def add_pos(x, pos_emb):
        B, T, _ = x.shape
        assert T <= pos_emb.num_embeddings, (T, pos_emb.num_embeddings)
        pos = torch.arange(T, device=x.device).expand(B, T)
        return x + pos_emb(pos)
    
    def forward(self, src_ids, src_pad_mask, tgt_ids, tgt_pad_mask):
        # src_pad_mask = src_ids.eq(self.c.src_pad_id)
        # tgt_pad_mask = tgt_ids.eq(self.c.tgt_pad_id)

        src = self.encode_tokens(src_ids, self.src_emb, self.src_pos_emb)
        tgt = self.encode_tokens(tgt_ids, self.tgt_emb, self.tgt_pos_emb)

        hidden = self.transformer(
            src=src,
            tgt=tgt,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=self.triu_mask_like(tgt),
            tgt_is_causal=True,
        )

        return self.lm_head(hidden)

    def encode_tokens(self, ids, embedding, pos_emb):
        x = embedding(ids) * math.sqrt(self.c.dim)
        x = self.add_pos(x, pos_emb)
        return self.dropout(x)

    @staticmethod
    def causal_mask_like(tgt):
        size = tgt.size(1)
        device = tgt.device
        return nn.Transformer.generate_square_subsequent_mask(size, device=device).not_equal(0.0)

    @staticmethod
    def triu_mask_like(tgt):
        tgt_len = tgt.size(1)
        device = tgt.device
        return torch.triu(
            torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
            diagonal=1,
        )


def build_model(cfg: TrainConfig, tok_src, tok_tgt):
    model_cfg = replace(
        cfg.model,
        src_vocab=len(tok_src),
        tgt_vocab=len(tok_tgt),
        src_pad_id=tok_src.pad_token_id,
        tgt_pad_id=tok_tgt.pad_token_id,
    )
    return TinySeq2SeqTransformer(model_cfg)

