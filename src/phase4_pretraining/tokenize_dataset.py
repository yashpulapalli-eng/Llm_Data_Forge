"""
Phase 4: Model Pretraining & Experimentation -- data preparation.

What this does (once, for all four dataset variants):
  1. Picks two fixed HELD-OUT validation sets and removes them from every
     variant's training data, so all models are tested on identical text:
       - heldout_clean: 500 documents that exist in ALL four variants
                        (i.e. documents that pass the quality filters).
       - heldout_raw:   500 random documents from the raw data (includes
                        low-quality text, like real web data).
  2. Trains ONE shared BPE tokenizer on the raw training documents, so token
     counts and losses are directly comparable across variants.
  3. Tokenizes each variant (and the held-out sets) into uint16 .npy files,
     with an end-of-text token between documents.

Run from the project root, for example:
    python src/phase4_pretraining/tokenize_dataset.py
Outputs go to data/tokenized/ (not committed to git).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VARIANTS = ["raw", "filtered", "deduplicated", "filtered_deduplicated"]
EOS = "<|endoftext|>"


def read_variant(path: Path):
    """Return (doc_ids, texts) for a Parquet dataset directory."""
    table = pq.read_table(str(path), columns=["doc_id", "text"])
    return table.column("doc_id").to_pylist(), table.column("text").to_pylist()


def train_tokenizer(texts, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_docs(tok: Tokenizer, texts, eos_id: int) -> np.ndarray:
    """Tokenize documents and join them with an end-of-text token."""
    chunks = []
    batch = 500
    for i in range(0, len(texts), batch):
        for enc in tok.encode_batch(texts[i:i + batch]):
            chunks.append(np.asarray(enc.ids + [eos_id], dtype=np.uint16))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Tokenize all dataset variants for training.")
    parser.add_argument("--data-dir", default="data", help="Folder that contains raw/, filtered/, ...")
    parser.add_argument("--output", default="data/tokenized")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--heldout-docs", type=int, default=500, help="Documents per held-out set.")
    parser.add_argument("--tokenizer-docs", type=int, default=8000, help="Max raw docs used to train the tokenizer.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- load all variants -------------------------------------------------
    print("Loading the four dataset variants ...")
    ids, texts = {}, {}
    for v in VARIANTS:
        ids[v], texts[v] = read_variant(data_dir / v)
        print(f"  {v:<22} {len(ids[v]):>7,} documents")

    # ---- held-out sets (same documents removed from every variant) ---------
    rng = np.random.default_rng(args.seed)
    common = sorted(set.intersection(*[set(ids[v]) for v in VARIANTS]))
    clean_ids = set(rng.choice(common, size=args.heldout_docs, replace=False).tolist())
    rest = sorted(set(ids["raw"]) - clean_ids)
    raw_ids = set(rng.choice(rest, size=args.heldout_docs, replace=False).tolist())
    heldout_all = clean_ids | raw_ids
    print(f"Held-out: {len(clean_ids)} clean docs + {len(raw_ids)} raw docs "
          f"(documents present in all four variants: {len(common):,})")

    text_by_id = dict(zip(ids["raw"], texts["raw"]))
    held_clean_texts = [text_by_id[i] for i in sorted(clean_ids)]
    held_raw_texts = [text_by_id[i] for i in sorted(raw_ids)]

    # ---- one shared tokenizer, trained on raw TRAINING documents -----------
    raw_train = [t for i, t in zip(ids["raw"], texts["raw"]) if i not in heldout_all]
    sample = rng.choice(len(raw_train), size=min(args.tokenizer_docs, len(raw_train)), replace=False)
    print(f"Training shared BPE tokenizer (vocab {args.vocab_size}) on {len(sample):,} raw documents ...")
    tok = train_tokenizer([raw_train[i] for i in sample], args.vocab_size)
    tok.save(str(out_dir / "tokenizer.json"))
    eos_id = tok.token_to_id(EOS)

    # ---- tokenize ----------------------------------------------------------
    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": eos_id, "seed": args.seed, "variants": {}}
    for name, doc_texts in (("heldout_clean", held_clean_texts), ("heldout_raw", held_raw_texts)):
        arr = encode_docs(tok, doc_texts, eos_id)
        np.save(out_dir / f"{name}.npy", arr)
        meta[name] = {"documents": len(doc_texts), "tokens": int(arr.size)}
        print(f"  {name:<22} {len(doc_texts):>7,} docs  {arr.size:>12,} tokens")

    for v in VARIANTS:
        keep = [t for i, t in zip(ids[v], texts[v]) if i not in heldout_all]
        arr = encode_docs(tok, keep, eos_id)
        np.save(out_dir / f"{v}.npy", arr)
        meta["variants"][v] = {"documents": len(keep), "tokens": int(arr.size),
                               "heldout_removed": len(ids[v]) - len(keep)}
        print(f"  {v:<22} {len(keep):>7,} docs  {arr.size:>12,} tokens")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {time.time() - t0:.0f}s. Saved to {out_dir}/")


if __name__ == "__main__":
    main()