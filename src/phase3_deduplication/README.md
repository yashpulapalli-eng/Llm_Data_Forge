# Phase 3: Deduplication Pipeline

**Objective:** Implement exact and near-duplicate detection at scale to produce deduplicated dataset variants (deduplicated-only, and filtered+deduplicated).

## Planned Tasks

- [ ] Implement exact-match deduplication via hashing
- [ ] Implement MinHash signature generation for near-duplicate detection
- [ ] Implement Locality-Sensitive Hashing (LSH) to find candidate near-duplicate pairs/clusters at scale
- [ ] Tune similarity threshold and number of hash permutations/bands
- [ ] Produce and store the "deduplicated-only" and "filtered+deduplicated" dataset variants

## Files (add as you build)

- `exact_dedup.py` — exact-match hashing dedup
- `minhash_lsh.py` — MinHash signatures + LSH near-duplicate detection
- `run_dedup.py` — Spark job producing `data/deduplicated/` and `data/filtered_deduplicated/`

## Status

See [`reports/Phase3_Report.docx`](../../reports/Phase3_Report.docx) for the full write-up once this phase is complete.
