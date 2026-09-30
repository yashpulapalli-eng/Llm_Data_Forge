"""
Phase 4: Model Pretraining & Experimentation — model architecture/hyperparameters.

Keep this config fixed across all four dataset-variant training runs
so the comparison in Phase 5 isolates the effect of the data, not the model.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Small/medium-scale decoder-only transformer config.
    Pick sizes appropriate to your available compute — document the
    final choice and rationale in the Phase 4 report.
    """
    vocab_size: int = 32_000
    context_length: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """
    Training budget — kept IDENTICAL across dataset variants so the
    comparison in Phase 5 is fair (same compute, different data).
    """
    batch_size: int = 32
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    seed: int = 42
