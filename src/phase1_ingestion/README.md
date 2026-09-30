# Phase 1: Data Acquisition & Infrastructure Setup

**Objective:** Source raw text corpora for pretraining experiments, and stand up the distributed processing environment (Apache Spark cluster/environment and Parquet-based, partitioned storage) that all later phases will run on.

**Compute target:** local PySpark (`local[*]`) — no cluster needed, per the project's scope.

**Dataset:** [`allenai/c4`](https://huggingface.co/datasets/allenai/c4) (English config), streamed via HuggingFace `datasets` — see the rationale in `ingest.py`'s module docstring. C4 is used instead of an already-clean corpus (e.g. WikiText) specifically because it still has realistic near-duplicates and boilerplate, which is what makes Phases 2–3 (filtering/dedup) have a visible, measurable effect. Streaming means only the first `--num-docs` documents are ever pulled — not the full ~750GB corpus.

## Planned Tasks

- [x] Identify raw text corpus to use as the base dataset — `allenai/c4` (en), streamed
- [x] Set up Apache Spark environment — local mode via `spark_session.py`
- [x] Define the Parquet storage schema — `doc_id` (string, uuid), `text` (string), `source` (string)
- [x] Implement ingestion (`ingest.py`) — streams N docs, loads into Spark, writes partitioned Parquet
- [ ] Run `ingest.py` for real and confirm `data/raw/` is populated at your target scale
- [ ] Record the actual document/token count and raw size in the Phase 1 report

## Files

- `spark_session.py` — shared Spark session/config used by later phases (implemented)
- `ingest.py` — streams the corpus from HuggingFace and writes it to `data/raw/` as partitioned Parquet (implemented — run it to actually populate `data/raw/`)

## How to Run

```bash
cd src/phase1_ingestion
python ingest.py --num-docs 20000        # adjust --num-docs to change scale
```

This was smoke-tested with synthetic data to confirm the Spark write/read path works; the actual HuggingFace download needs to run in an environment with network access to huggingface.co (should be fine on your laptop).

## Status

See [`reports/Phase1_Report.docx`](../../reports/Phase1_Report.docx) for the full write-up once this phase is complete — including the real document count and raw dataset size once you've run `ingest.py`.
