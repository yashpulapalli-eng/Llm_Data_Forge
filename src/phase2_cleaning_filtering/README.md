# Phase 2: Cleaning & Quality Filtering Pipeline

**Objective:** Build the Spark-based ETL stage that cleans raw text and applies quality filters, producing a "filtered" dataset variant.

## Planned Tasks

- [ ] Implement text cleaning (boilerplate/HTML/markup stripping, encoding fixes)
- [ ] Define and implement quality-filtering heuristics (language ID, length thresholds, repetition ratio, symbol/word ratio, perplexity or heuristic quality score)
- [ ] Run the filtering pipeline over the full raw dataset at scale in Spark
- [ ] Produce and store the "filtered-only" dataset variant in `data/filtered/`
- [ ] Quantify how much data was removed and why

## Files (add as you build)

- `clean_text.py` — boilerplate/markup stripping, encoding normalization
- `quality_filters.py` — filtering heuristics and thresholds
- `run_filtering.py` — Spark job that applies the above over the raw dataset

## Status

See [`reports/Phase2_Report.docx`](../../reports/Phase2_Report.docx) for the full write-up once this phase is complete.
