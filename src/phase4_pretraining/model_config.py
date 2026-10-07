"""
Phase 4: Model Pretraining & Experimentation -- model and training settings.

These settings are kept IDENTICAL across all four dataset-variant runs, so the
comparison in Phase 5 isolates the effect of the data, not the model.

The model is a small GPT-style decoder-only transformer, sized so that one run
trains in about 5 minutes on a laptop CPU.
"""

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 4096        # must match the shared tokenizer (tokenize_dataset.py)
    context_length: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 32          # sequences per step -> 32 x 256 = 8,192 tokens per step
    learning_rate: float = 1e-3
    min_lr_ratio: float = 0.1     # cosine decay down to 10% of the peak LR
    warmup_fraction: float = 0.05
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    num_evals: int = 6            # validation checks spread over the run
    eval_tokens: int = 65_536     # tokens per held-out set used for these quick checks

    def to_dict(self):
        return asdict(self)