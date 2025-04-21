from dataclasses import dataclass


@dataclass
class ModelConfig:
    # 小 LLaMA
    vocab_size: int = 50257
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    rope_theta: float = 10000.0
    dropout: float = 0.1

    # Edit 任务
    index_loss_weight: float = 1.0
    lm_loss_weight: float = 0.5


@dataclass
class TrainConfig:
    seq_len: int = 512
    batch_size: int = 16
    total_steps: int = 100_000
    lr: float = 2e-4
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    fp16: bool = True
    ckpt_every: int = 1_000
