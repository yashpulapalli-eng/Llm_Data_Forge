"""
Phase 2: Cleaning & Quality Filtering Pipeline — Spark job entry point.

Reads data/raw/, applies clean_text + quality_filters, writes the
"filtered-only" dataset variant to data/filtered/.
"""

import argparse
import sys
from pathlib import Path

from clean_text import clean_text_udf
from quality_filters import FilterThresholds, passes_all_filters
from pyspark.sql.functions import col, udf
from pyspark.sql.types import BooleanType

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_ingestion"))
from spark_session import get_spark_session  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def run(input_dir: Path, output_dir: Path, thresholds: FilterThresholds):
    spark = get_spark_session(app_name="phase2-filtering")
    try:
        df = spark.read.parquet(str(input_dir))
        raw_count = df.count()

        df = df.withColumn("text", clean_text_udf()(col("text")))

        keep_udf = udf(lambda t: passes_all_filters(t, thresholds), BooleanType())
        filtered_df = df.filter(keep_udf(col("text")))
        kept_count = filtered_df.count()

        filtered_df.write.mode("overwrite").parquet(str(output_dir))

        removed = raw_count - kept_count
        pct_removed = 100 * removed / raw_count if raw_count else 0
        print(f"Raw: {raw_count:,} | Kept: {kept_count:,} | Removed: {removed:,} ({pct_removed:.1f}%)")
        # TODO: log this summary into the Phase 2 report's Results table.
    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Run quality filtering over the raw corpus.")
    parser.add_argument("--input", default=str(DATA_DIR / "raw"))
    parser.add_argument("--output", default=str(DATA_DIR / "filtered"))
    args = parser.parse_args()

    run(Path(args.input), Path(args.output), FilterThresholds())


if __name__ == "__main__":
    main()
