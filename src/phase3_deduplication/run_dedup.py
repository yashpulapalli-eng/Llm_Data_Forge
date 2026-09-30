"""
Phase 3: Deduplication Pipeline — Spark job entry point.

Reads data/raw/ and data/filtered/ (Phase 2 output), and produces:
  - data/deduplicated/            (raw + dedup, no quality filtering)
  - data/filtered_deduplicated/   (filtered + dedup)
"""

import argparse
import sys
from pathlib import Path

from exact_dedup import deduplicate_exact
from minhash_lsh import LSHConfig, find_near_duplicate_clusters

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_ingestion"))
from spark_session import get_spark_session  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def dedup_variant(spark, input_dir: Path, output_dir: Path, config: LSHConfig):
    df = spark.read.parquet(str(input_dir))
    before = df.count()

    df = deduplicate_exact(df)
    after_exact = df.count()

    clusters = find_near_duplicate_clusters(df, config)
    # TODO: join `clusters` back onto `df` and keep one representative per cluster.
    df_final = df  # placeholder until near-dup resolution is implemented

    after_all = df_final.count()
    df_final.write.mode("overwrite").parquet(str(output_dir))

    print(f"{input_dir.name}: {before:,} -> {after_exact:,} (exact dedup) "
          f"-> {after_all:,} (+ near-dup removal)")
    # TODO: log this summary into the Phase 3 report's Results table.


def main():
    parser = argparse.ArgumentParser(description="Run exact + near-duplicate dedup.")
    parser.add_argument("--raw-input", default=str(DATA_DIR / "raw"))
    parser.add_argument("--filtered-input", default=str(DATA_DIR / "filtered"))
    parser.add_argument("--dedup-output", default=str(DATA_DIR / "deduplicated"))
    parser.add_argument("--filtered-dedup-output", default=str(DATA_DIR / "filtered_deduplicated"))
    args = parser.parse_args()

    config = LSHConfig()
    spark = get_spark_session(app_name="phase3-dedup")
    try:
        dedup_variant(spark, Path(args.raw_input), Path(args.dedup_output), config)
        dedup_variant(spark, Path(args.filtered_input), Path(args.filtered_dedup_output), config)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
