"""
Phase 4: Model Pretraining & Experimentation — training loop.

Run once per dataset variant with an identical ModelConfig/TrainConfig,
varying only --data-dir and --run-name, so results are comparable.

TODO before this is real training code:
    - Define the Dataset/DataLoader over tokenized sequences (see tokenize_dataset.py).
    - Define or import the model architecture matching ModelConfig.
    - Wire up an optimizer (AdamW), LR schedule (cosine + warmup), and
      the training/eval loop.
    - Log loss, tokens/sec, wall-clock time, and step count per run
      (e.g. to Weights & Biases) — Phase 5 needs these for comparison.
"""

import argparse
import time
from pathlib import Path

from model_config import ModelConfig, TrainConfig


def build_model(config: ModelConfig):
    """Construct the model from ModelConfig. TODO: implement/import architecture."""
    raise NotImplementedError


def build_dataloader(data_dir: Path, batch_size: int):
    """Build the training DataLoader from tokenized data. TODO: implement."""
    raise NotImplementedError


def train(data_dir: Path, run_name: str, model_cfg: ModelConfig, train_cfg: TrainConfig):
    model = build_model(model_cfg)
    dataloader = build_dataloader(data_dir, train_cfg.batch_size)

    start = time.time()
    # TODO: real training loop — forward/backward/step, periodic eval,
    # checkpointing, and metric logging (loss, tokens/sec, elapsed time).
    raise NotImplementedError

    elapsed = time.time() - start  # noqa: F841 (unreachable until loop is implemented)
    print(f"[{run_name}] training complete in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Pretrain the model on one dataset variant.")
    parser.add_argument("--data-dir", required=True, help="Tokenized data dir (see tokenize_dataset.py).")
    parser.add_argument("--run-name", required=True, help="e.g. raw / filtered / dedup / filtered_dedup.")
    args = parser.parse_args()

    train(Path(args.data_dir), args.run_name, ModelConfig(), TrainConfig())


if __name__ == "__main__":
    main()
