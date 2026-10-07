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

How it scales (changed after the 20,000-document pilot):
Documents are streamed and written to Parquet in fixed-size batches
(--rows-per-file) with pyarrow, so memory use stays constant no matter
how many documents are pulled. (The pilot version collected everything
in a Python list and handed it to Spark, which does not scale to
hundreds of thousands of documents.) Spark is used afterwards to read the
Parquet back and verify it, and by every later phase.

Swap --dataset/--config/--split to point at a different HuggingFace
corpus if you choose one instead; see the Phase 1 report for whichever
corpus you actually settle on and why.
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

RAW_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DEFAULT_DATASET = "allenai/c4"
DEFAULT_CONFIG = "en"
DEFAULT_SPLIT = "train"

SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("text", pa.string()),
    ("source", pa.string()),
])


def iter_source_corpus(
    num_docs: int,
    dataset_name: str = DEFAULT_DATASET,
    config_name: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
) -> Iterator[dict]:
    """
    Stream the first `num_docs` non-empty documents from a HuggingFace
    dataset without downloading the entire corpus, one at a time.

    Yields:
        {"doc_id": str, "text": str, "source": str} dicts.
    """
    from datasets import load_dataset  # imported lazily -- later phases don't need this installed

    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    source = f"{dataset_name}/{config_name}"

    pulled = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        yield {"doc_id": str(uuid.uuid4()), "text": text, "source": source}
        pulled += 1
        if pulled >= num_docs:
            break


def stream_source_corpus(num_docs: int, dataset_name: str = DEFAULT_DATASET,
                         config_name: str = DEFAULT_CONFIG, split: str = DEFAULT_SPLIT) -> list[dict]:
    """List version of iter_source_corpus (fine for small samples only)."""
    return list(iter_source_corpus(num_docs, dataset_name, config_name, split))


def clear_previous_output(output_dir: Path) -> None:
    """Make the output folder ready: remove files from an earlier run (Parquet parts and Spark leftovers)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.parquet", "*.crc", "_SUCCESS"):
        for old in output_dir.glob(pattern):
            old.unlink()


def write_parquet_batches(docs: Iterable[dict], output_dir: Path, rows_per_file: int = 20_000) -> int:
    """
    Write documents to `output_dir` as Snappy-compressed Parquet files of
    `rows_per_file` rows each (part-00000.parquet, part-00001.parquet, ...).
    Only one batch is ever held in memory.

    Returns:
        Total number of documents written.
    """
    clear_previous_output(output_dir)

    batch: list[dict] = []
    file_index = 0
    total = 0
    started = time.time()

    def flush() -> None:
        nonlocal batch, file_index, total
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=SCHEMA)
        pq.write_table(table, output_dir / f"part-{file_index:05d}.parquet", compression="snappy")
        total += len(batch)
        file_index += 1
        rate = total / max(time.time() - started, 1e-9)
        print(f"  wrote file {file_index}: {total:,} documents so far ({rate:,.0f} docs/sec)", flush=True)
        batch = []

    for doc in docs:
        batch.append(doc)
        if len(batch) >= rows_per_file:
            flush()
    flush()
    return total


def verify_with_spark(output_dir: Path) -> int:
    """Read the Parquet back with Spark (the engine every later phase uses) and return the row count."""
    from spark_session import get_spark_session

    spark = get_spark_session(app_name="phase1-verify")
    try:
        return spark.read.parquet(str(output_dir)).count()
    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Ingest raw corpus into partitioned Parquet.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset name.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="HuggingFace dataset config/subset.")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--num-docs", type=int, default=100_000,
                        help="How many documents to pull (streamed, not the full corpus).")
    parser.add_argument("--output", default=str(RAW_OUTPUT_DIR), help="Output directory for Parquet.")
    parser.add_argument("--rows-per-file", type=int, default=20_000,
                        help="Documents per Parquet file (one file per batch).")
    parser.add_argument("--no-verify", action="store_true", help="Skip the Spark read-back check.")
    # Old option from the pilot version (Spark repartitioning). Accepted and ignored so
    # existing PyCharm run configurations that still pass it keep working.
    parser.add_argument("--partitions", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = Path(args.output)
    print(f"Streaming {args.num_docs:,} documents from {args.dataset}/{args.config} ({args.split}) ...", flush=True)
    docs = iter_source_corpus(args.num_docs, args.dataset, args.config, args.split)
    written = write_parquet_batches(docs, output_dir, args.rows_per_file)
    if written == 0:
        raise RuntimeError(
            f"No documents pulled from {args.dataset}/{args.config} ({args.split}). "
            "Check the dataset name/config and network access."
        )
    print(f"Wrote {written:,} documents to {output_dir}", flush=True)

    if not args.no_verify:
        count = verify_with_spark(output_dir)
        status = "OK" if count == written else "MISMATCH"
        print(f"Spark read back {count:,} documents ({status}).")


if __name__ == "__main__":
    main()