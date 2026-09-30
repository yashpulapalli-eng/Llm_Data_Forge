"""
Phase 1: Data Acquisition & Infrastructure Setup

Streams a bounded number of documents from a public HuggingFace corpus
and writes them to data/raw/ as partitioned Parquet, establishing the
schema every later phase reads.

Dataset choice (default: allenai/c4, English config):
C4 (Colossal Clean Crawled Corpus) is used instead of an already
heavily-cleaned corpus like WikiText, because C4 still contains
realistic near-duplicate pages and boilerplate -- which is exactly
what Phase 2 (filtering) and Phase 3 (dedup) need in order to have a
measurable effect. It's still much easier to work with than raw
Common Crawl WARC files, and HuggingFace `datasets` lets us stream it
without downloading the full (~750GB) corpus -- we only ever pull the
first --num-docs examples.

Swap --dataset/--config/--split to point at a different HuggingFace
corpus if you choose one instead; see the Phase 1 report for whichever
corpus you actually settle on and why.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from spark_session import get_spark_session

RAW_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DEFAULT_DATASET = "allenai/c4"
DEFAULT_CONFIG = "en"
DEFAULT_SPLIT = "train"


def stream_source_corpus(
    num_docs: int,
    dataset_name: str = DEFAULT_DATASET,
    config_name: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
) -> list[dict]:
    """
    Stream the first `num_docs` non-empty documents from a HuggingFace
    dataset without downloading the entire corpus.

    Returns:
        A list of {"doc_id": str, "text": str, "source": str} dicts,
        ready to hand to Spark via spark.createDataFrame(...).
    """
    from datasets import load_dataset  # imported lazily -- later phases don't need this installed

    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)

    docs = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        docs.append({
            "doc_id": str(uuid.uuid4()),
            "text": text,
            "source": f"{dataset_name}/{config_name}",
        })
        if len(docs) >= num_docs:
            break
    return docs


def load_source_corpus(spark, num_docs: int, dataset_name: str, config_name: str, split: str):
    """
    Stream the source corpus and load it into a Spark DataFrame with
    `doc_id`, `text`, and `source` columns.
    """
    docs = stream_source_corpus(num_docs, dataset_name, config_name, split)
    if not docs:
        raise RuntimeError(
            f"No documents pulled from {dataset_name}/{config_name} ({split}). "
            "Check the dataset name/config and network access."
        )
    return spark.createDataFrame(docs)


def write_partitioned_parquet(df, output_dir: Path, num_partitions: int = 8):
    """
    Write a DataFrame to Parquet, partitioned for efficient downstream reads.

    TODO once corpus size is known: revisit num_partitions -- a good
    rule of thumb is aiming for ~100-250MB per output file.
    """
    (df.repartition(num_partitions)
       .write
       .mode("overwrite")
       .parquet(str(output_dir)))


def main():
    parser = argparse.ArgumentParser(description="Ingest raw corpus into partitioned Parquet.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset name.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="HuggingFace dataset config/subset.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--num-docs", type=int, default=20_000,
                         help="How many documents to pull (streamed, not the full corpus).")
    parser.add_argument("--output", default=str(RAW_OUTPUT_DIR), help="Output directory for Parquet.")
    parser.add_argument("--partitions", type=int, default=8)
    args = parser.parse_args()

    spark = get_spark_session(app_name="phase1-ingestion")
    try:
        df = load_source_corpus(spark, args.num_docs, args.dataset, args.config, args.split)
        count = df.count()
        write_partitioned_parquet(df, Path(args.output), args.partitions)
        print(f"Wrote {count:,} documents to {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
