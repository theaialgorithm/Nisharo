"""Central configuration for the Nisharo from-scratch language model.

Everything here is plain Python - no third-party services. Tune these values
to trade off training time vs. model quality on this CPU-only machine.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Transformer architecture (GPT-style decoder)
    vocab_size: int = 8000        # set automatically after tokenizer training
    block_size: int = 256         # context length (tokens the model can see)
    n_layer: int = 6              # number of transformer blocks
    n_head: int = 6               # number of attention heads
    n_embd: int = 384             # embedding / hidden dimension
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    # Optimisation
    batch_size: int = 32
    max_iters: int = 3000
    eval_interval: int = 500
    eval_iters: int = 40
    log_interval: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    lr_decay_iters: int = 3000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Bookkeeping
    always_save_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"


# Vocabulary size target for the BPE tokenizer
VOCAB_SIZE = 8000
