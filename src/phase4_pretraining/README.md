# Phase 4: Model Pretraining & Experimentation

**Objective:** Tokenize each curated dataset variant and pretrain a small/medium-scale language model on each, under matched, comparable compute budgets.

## Planned Tasks

- [ ] Choose and configure the small/medium-scale model architecture
- [ ] Tokenize each dataset variant (raw, filtered-only, deduplicated-only, filtered+deduplicated)
- [ ] Define a fixed, comparable training budget (steps/tokens/compute) across variants
- [ ] Run pretraining for each dataset variant
- [ ] Log training curves (loss, tokens/sec, wall-clock time, compute cost) for each run

## Files (add as you build)

- `tokenize_dataset.py` — tokenizes a given dataset variant
- `model_config.py` — model architecture/hyperparameters
- `train.py` — pretraining loop, run once per dataset variant
- `logs/` — training curves / run metadata per variant

## Status

See [`reports/Phase4_Report.docx`](../../reports/Phase4_Report.docx) for the full write-up once this phase is complete.
