# LLM Data Forge

**Distributed Data Curation and Quality Optimization for Efficient LLM Pretraining**
CSCI 6996 Independent Study

## Overview

Massive training datasets are required for large language models, but these datasets frequently include inconsistent data sources, redundant text, duplicate content, low-quality data, and boilerplate. Processing and training on redundant or poor-quality data increases storage, preprocessing, and compute expenses.

**LLM Data Forge** builds a distributed Big Data pipeline for collecting, processing, filtering, deduplicating, and constructing LLM training datasets, using Apache Spark, Parquet, hashing, MinHash, and Locality-Sensitive Hashing (LSH) to produce multiple curated versions of a training corpus. A small/medium-scale language model is then pretrained on each version to study the effect of data composition, deduplication, and quality on training efficiency and model performance.

## Problem Statement

Large Language Models require large volumes of training data, but these datasets often contain duplicate, repetitive, noisy, or low-quality content. Processing unnecessary data increases storage, processing time, and computational cost. This project addresses the challenge of identifying and improving the quality of LLM training data using scalable distributed data-processing techniques.

## Objective

Develop a distributed data-curation pipeline that filters, deduplicates, and organizes large-scale LLM training data, and evaluate the impact of these techniques on training efficiency and model performance.

## End Goal

Determine whether better-curated training data can reduce data and computational requirements while maintaining or improving LLM performance, providing practical insights into more efficient LLM training.

## Project Phases

| Phase | Name | Code | Report |
|---|---|---|---|
| 1 | Data Acquisition & Infrastructure Setup | [`src/phase1_ingestion`](src/phase1_ingestion) | [Phase 1 Report](reports/Phase1_Report.docx) |
| 2 | Cleaning & Quality Filtering Pipeline | [`src/phase2_cleaning_filtering`](src/phase2_cleaning_filtering) | [Phase 2 Report](reports/Phase2_Report.docx) |
| 3 | Deduplication Pipeline (Hashing / MinHash / LSH) | [`src/phase3_deduplication`](src/phase3_deduplication) | [Phase 3 Report](reports/Phase3_Report.docx) |
| 4 | Model Pretraining & Experimentation | [`src/phase4_pretraining`](src/phase4_pretraining) | [Phase 4 Report](reports/Phase4_Report.docx) |
| 5 | Evaluation & Analysis | [`src/phase5_evaluation`](src/phase5_evaluation) | [Phase 5 Report](reports/Phase5_Report.docx) |
| — | **Final Project Report** | — | [Final Project Report](reports/Final_Project_Report.docx) |

Each `src/phaseN_*` folder has its own README with that phase's objective and planned tasks — code goes there as it's written. Each phase report in `reports/` is filled in progressively as that phase is completed (see [`docs/Project_Proposal.pdf`](docs/Project_Proposal.pdf) for the original proposal).

## Repository Structure

```
llm-data-forge/
├── docs/                 # Original proposal and reference material
├── reports/              # Phase 1-5 reports + final project report (.docx)
├── src/
│   ├── phase1_ingestion/
│   ├── phase2_cleaning_filtering/
│   ├── phase3_deduplication/
│   ├── phase4_pretraining/
│   └── phase5_evaluation/
├── data/                 # Local data staging (gitignored — see data/README.md)
├── notebooks/            # Exploratory notebooks
├── requirements.txt
└── .gitignore
```

## Getting Started (PyCharm)

1. Open the `llm-data-forge/` folder as a PyCharm project.
2. Right-click `src/` → **Mark Directory as** → **Sources Root** (fixes imports like `from spark_session import get_spark_session` across phase folders).
3. Create a virtualenv/interpreter for the project and install dependencies: `pip install -r requirements.txt`.
4. Each phase's `.py` files are scaffolded with function signatures, docstrings, and `TODO`s but not implemented yet — fill them in as you work through that phase. Every script has a runnable `if __name__ == "__main__":` entry point with `argparse` args, so you can add a PyCharm Run Configuration per script once it's implemented.
5. `.idea/` is already gitignored, so your local PyCharm project settings won't get committed.

## Tech Stack

- **Distributed processing:** Apache Spark (PySpark), Parquet
- **Deduplication:** exact hashing, MinHash, Locality-Sensitive Hashing (LSH)
- **Modeling:** PyTorch, tokenization (BPE/SentencePiece)
- **Experiment tracking:** Weights & Biases / TensorBoard (pick one)

## Workflow / How This Repo Is Used

This project is documented **as it progresses**, not all at once:

1. Work on a phase → commit code to that phase's `src/` folder as you go.
2. When the phase is done, fill in the corresponding report in `reports/`.
3. Commit with a message referencing the phase, e.g. `git commit -m "Phase 2: implement quality filtering heuristics"`.
4. After Phase 5, fill in the Final Project Report, answering whether the project's objective was achieved.

## References

1. L. Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," arXiv:2101.00027, 2020.
2. K. Lee et al., "Deduplicating Training Data Makes Language Models Better," arXiv:2107.06499, 2022.
3. G. Penedo et al., "The Refined Web Dataset for Falcon LLMs," 2023.
4. G. Penedo et al., "The Fine Web Datasets: Decanting the Web for the Finest Text Data at Scale," NeurIPS, 2024.
5. J. Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361, 2020.
6. J. Hoffmann et al., "Training Compute-Optimal Large Language Models," arXiv:2203.15556, 2022.
7. S. Y. Gadre et al., "DataComp: In Search of the Next Generation of Multimodal Datasets," arXiv:2304.14108, 2023.
