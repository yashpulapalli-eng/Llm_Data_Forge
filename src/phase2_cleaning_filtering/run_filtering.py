"""
Phase 2: Cleaning & Quality Filtering Pipeline -- Spark job entry point.

Reads data/raw/, cleans the text, applies every quality filter, and writes
the "filtered-only" dataset variant to data/filtered/.

It also answers the Phase 2 report's key question -- "how much was removed,
and why?" -- by recording, for every filter:
  * how many documents fail it (a document can fail several filters), and
  * how many documents fail ONLY that filter.
It saves a few removed examples per filter for spot-checking, and compares
the dataset before vs. after filtering.

Run:
    python run_filtering.py
    python run_filtering.py --min-length 300 --min-alpha 0.75    # try other thresholds

Outputs:
    data/filtered/                               filtered Parquet (gitignored)
    results/phase2/filtering_summary.json        numbers for the report (commit this)
    results/phase2/filtered_stats.json           same stats as raw_stats.json, after filtering
    results/phase2/removed_samples.jsonl         examples of removed docs (local only, gitignored)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from functools import reduce
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StructField, StructType

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[0] / "phase1_ingestion"))
from spark_session import get_spark_session  # noqa: E402

from baseline_stats import compute_stats  # noqa: E402
from clean_text import clean_text_udf  # noqa: E402
from quality_filters import FILTER_NAMES, FilterThresholds, filter_flags  # noqa: E402

REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "phase2"

SAMPLES_PER_FILTER = 5
SAMPLE_CHARS = 600

FLAGS_SCHEMA = StructType([StructField(name, BooleanType(), False) for name in FILTER_NAMES])


def run(input_dir: Path, output_dir: Path, results_dir: Path, thresholds: FilterThresholds, partitions: int):
    results_dir.mkdir(parents=True, exist_ok=True)
    spark = get_spark_session(app_name="phase2-filtering")
    try:
        # Make our helper modules importable inside Spark's Python workers.
        for module in ("clean_text.py", "quality_filters.py"):
            spark.sparkContext.addPyFile(str(HERE / module))

        raw = spark.read.parquet(str(input_dir))
        raw_count = raw.count()

        # 1. Clean, 2. compute every filter's verdict for every document.
        cleaned = raw.withColumn("text", clean_text_udf()(F.col("text")))
        # useArrow=False: see the note in clean_text.clean_text_udf (large-document hang).
        flags_udf = F.udf(lambda t: filter_flags(t, thresholds), FLAGS_SCHEMA, useArrow=False)
        flagged = cleaned.withColumn("flags", flags_udf(F.col("text"))).cache()

        passes = reduce(lambda a, b: a & b, [F.col(f"flags.{n}") for n in FILTER_NAMES])

        # 3. Per-filter removal counts (all in one pass over the data).
        exprs = [F.sum((~passes).cast("int")).alias("removed_total")]
        for name in FILTER_NAMES:
            fails = ~F.col(f"flags.{name}")
            others_pass = reduce(lambda a, b: a & b,
                                 [F.col(f"flags.{m}") for m in FILTER_NAMES if m != name])
            exprs.append(F.sum(fails.cast("int")).alias(f"fail_{name}"))
            exprs.append(F.sum((fails & others_pass).cast("int")).alias(f"only_{name}"))
        counts = flagged.agg(*exprs).first().asDict()

        # 4. Save a few removed examples per filter for spot-checking.
        samples_path = results_dir / "removed_samples.jsonl"
        with open(samples_path, "w", encoding="utf-8") as fh:
            for name in FILTER_NAMES:
                rows = (flagged.filter(~F.col(f"flags.{name}"))
                        .select("doc_id", "text").limit(SAMPLES_PER_FILTER).collect())
                for r in rows:
                    fh.write(json.dumps({
                        "failed_filter": name,
                        "doc_id": r["doc_id"],
                        "chars": len(r["text"] or ""),
                        "text_preview": (r["text"] or "")[:SAMPLE_CHARS],
                    }, ensure_ascii=False) + "\n")

        # 5. Write the filtered dataset.
        kept = flagged.filter(passes).select("doc_id", "text", "source")
        kept.repartition(partitions).write.mode("overwrite").parquet(str(output_dir))
        flagged.unpersist()

        # 6. Before / after statistics (re-read from disk = what later phases will see).
        raw_stats_path = results_dir / "raw_stats.json"
        raw_stats = (json.loads(raw_stats_path.read_text())
                     if raw_stats_path.exists() else compute_stats(raw))
        filtered_stats = compute_stats(spark.read.parquet(str(output_dir)))
        (results_dir / "filtered_stats.json").write_text(json.dumps(filtered_stats, indent=2))
    finally:
        spark.stop()

    kept_count = filtered_stats["documents"]
    removed = counts["removed_total"] or 0
    assert kept_count + removed == raw_count, "kept + removed should equal raw document count"

    summary = {
        "thresholds": asdict(thresholds),
        "raw_documents": raw_count,
        "kept_documents": kept_count,
        "removed_documents": removed,
        "pct_removed": round(100 * removed / raw_count, 2) if raw_count else 0,
        "per_filter": {
            name: {"fail": counts[f"fail_{name}"] or 0, "fail_only_this": counts[f"only_{name}"] or 0}
            for name in FILTER_NAMES
        },
        "raw_estimated_tokens": raw_stats["estimated_tokens"],
        "filtered_estimated_tokens": filtered_stats["estimated_tokens"],
        "raw_text_size_mb": raw_stats["raw_text_size_mb"],
        "filtered_text_size_mb": filtered_stats["raw_text_size_mb"],
    }
    (results_dir / "filtering_summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"\nFiltered dataset written to {output_dir}")
    print(f"Summary saved to {results_dir / 'filtering_summary.json'}")
    print(f"Removed-document examples (for spot-checking): {samples_path}")


def print_summary(s: dict) -> None:
    raw = s["raw_documents"] or 1
    print("\n=== Phase 2 filtering summary ===")
    print(f"Raw documents:      {s['raw_documents']:,}")
    print(f"Kept documents:     {s['kept_documents']:,}")
    print(f"Removed documents:  {s['removed_documents']:,} ({s['pct_removed']}%)")
    print("\nWhy documents were removed (a document can fail more than one filter):")
    print(f"  {'filter':<14}{'fails this':>14}{'fails ONLY this':>20}")
    for name, v in s["per_filter"].items():
        print(f"  {name:<14}{v['fail']:>8,} ({100 * v['fail'] / raw:4.1f}%){v['fail_only_this']:>14,}")
    print(f"\nText size:          {s['raw_text_size_mb']} MB -> {s['filtered_text_size_mb']} MB")
    print(f"Estimated tokens:   ~{s['raw_estimated_tokens']:,} -> ~{s['filtered_estimated_tokens']:,}")


def main():
    d = FilterThresholds()
    parser = argparse.ArgumentParser(description="Run cleaning + quality filtering over the raw corpus.")
    parser.add_argument("--input", default=str(DATA_DIR / "raw"))
    parser.add_argument("--output", default=str(DATA_DIR / "filtered"))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--min-length", type=int, default=d.min_length_chars)
    parser.add_argument("--max-length", type=int, default=d.max_length_chars)
    parser.add_argument("--min-alpha", type=float, default=d.min_alpha_ratio)
    args = parser.parse_args()

    thresholds = FilterThresholds(
        min_length_chars=args.min_length,
        max_length_chars=args.max_length,
        min_alpha_ratio=args.min_alpha,
    )
    run(Path(args.input), Path(args.output), Path(args.results_dir), thresholds, args.partitions)


if __name__ == "__main__":
    main()