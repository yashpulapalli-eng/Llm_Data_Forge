# Phase 4: Model Pretraining & Experimentation

**Objective:** Tokenize each curated dataset variant and pretrain a small language model on each, under matched, comparable compute budgets.

## Planned Tasks

- [x] Choose and configure the small/medium-scale model architecture
- [x] Tokenize each dataset variant (raw, filtered-only, deduplicated-only, filtered+deduplicated)
- [x] Define a fixed, comparable training budget (steps/tokens/compute) across variants
- [x] Run pretraining for each dataset variant
- [x] Log training curves (loss, tokens/sec, wall-clock time, compute cost) for each run

## Files

- `model_config.py` — model architecture and training settings (identical for every run)
- `tokenize_dataset.py` — builds two fixed held-out test sets, trains ONE shared BPE tokenizer, tokenizes all four variants
- `train.py` — small GPT model + training loop (one variant, fixed token budget)
- `run_experiments.py` — measures speed, fixes one token budget, trains every variant for every seed
- Outputs: `data/tokenized/` (tokens, not committed), `checkpoints/` (not committed), `results/phase4/*.json` (training logs, committed)

## How to Run (from the project root)

```
pip install torch tokenizers matplotlib
python src/phase4_pretraining/tokenize_dataset.py
python src/phase4_pretraining/run_experiments.py --minutes 5 --seeds 1 2
```

Runs that already have a result file are skipped, so you can stop and resume.

## Fair-comparison design

- Same model, same settings, same token budget (saved in `results/phase4/budget.json`) for every variant.
- Same starting weights for each seed across variants (paired comparison).
- Same held-out text for every model: 500 documents present in all four variants ("clean") and 500 random raw documents ("raw"). These are removed from every variant's training data.
- One shared tokenizer, so losses are directly comparable.

## Status

See [`reports/Phase4_Report.docx`](../../reports/Phase4_Report.docx) for the full write-up.