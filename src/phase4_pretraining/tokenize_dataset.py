"""
Phase 4: Model Pretraining & Experimentation — tokenization.

Tokenizes a given dataset variant (raw / filtered / deduplicated /
filtered_deduplicated) into the format the training loop consumes.
"""

import argparse
from pathlib import Path


def load_or_train_tokenizer(vocab_size: int = 32_000, tokenizer_path: str = None):
    """
    Load an existing tokenizer, or train a new one on the corpus.

    TODO:
        - Decide: train a fresh BPE tokenizer per variant, or reuse
          one tokenizer across all variants (recommended, so token
          count is directly comparable across dataset variants).
        - Pick a library: `tokenizers` (HuggingFace) or `sentencepiece`.
    """
    raise NotImplementedError


def tokenize_variant(input_dir: Path, output_dir: Path, tokenizer):
    """
    Tokenize every document in `input_dir` (a Parquet dataset variant)
    and write tokenized/packed sequences to `output_dir`.

    TODO:
        - Decide sequence packing strategy (fixed context length,
          document boundary handling, padding vs. concatenation).
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Tokenize a dataset variant for training.")
    parser.add_argument("--input", required=True, help="Path to a data/<variant>/ Parquet dir.")
    parser.add_argument("--output", required=True, help="Output dir for tokenized sequences.")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    args = parser.parse_args()

    tokenizer = load_or_train_tokenizer(vocab_size=args.vocab_size)
    tokenize_variant(Path(args.input), Path(args.output), tokenizer)


if __name__ == "__main__":
    main()
