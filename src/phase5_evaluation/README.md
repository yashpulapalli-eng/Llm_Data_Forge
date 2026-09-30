# Phase 5: Evaluation & Analysis

**Objective:** Evaluate and compare the models trained on each dataset variant, and determine whether curation reduced data/compute requirements without hurting performance — directly answering the project's objective.

## Planned Tasks

- [ ] Define downstream evaluation tasks/benchmarks and metrics
- [ ] Run evaluation for each model (one per dataset variant)
- [ ] Compare training efficiency (time/compute/cost) across variants
- [ ] Compare downstream performance across variants
- [ ] Synthesize findings into a clear answer to the project's objective and end goal

## Files (add as you build)

- `evaluate.py` — runs benchmark(s) against a trained model checkpoint
- `compare_results.py` — aggregates metrics across all four dataset variants
- `plots/` — comparison charts (size vs. performance, compute vs. performance, etc.)

## Status

See [`reports/Phase5_Report.docx`](../../reports/Phase5_Report.docx) and [`reports/Final_Project_Report.docx`](../../reports/Final_Project_Report.docx) for the full write-up once this phase is complete.
