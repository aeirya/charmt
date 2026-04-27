from dataclasses import dataclass
import yaml

@dataclass
class ModelConfig:
    src_vocab: int = None
    tgt_vocab: int = None
    src_pad_id: int = None
    tgt_pad_id: int = None
    
    dim: int = 64
    nhead: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    src_max_len: int = 512
    tgt_max_len: int = 128


@dataclass
class TrainConfig:
    model: ModelConfig
    lr: float = 1e-3
    n_epoch: int = 100
    limit: int = 1000
    lang: str = 'tur'
    batch_size: int = 256
    workers: int = 2

    def get_limit(self):
        if self.limit and self.limit > 0:
            return self.limit
        return None
    

def load_config(path='config.yaml'):
    with open(path) as f:
        raw_cfg = yaml.safe_load(f)

    cfg = TrainConfig(
        lr=raw_cfg["lr"],
        n_epoch=raw_cfg['n_epoch'],
        lang=raw_cfg['lang'],
        limit=raw_cfg['limit'],
        batch_size=raw_cfg['batch_size'],
        model=ModelConfig(**raw_cfg["model"]),
        workers=raw_cfg['workers'],
    )
    return cfg