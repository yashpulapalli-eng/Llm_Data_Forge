"""
Phase 2: Cleaning & Quality Filtering Pipeline -- raw baseline statistics.

Measures the raw (unfiltered) dataset in data/raw/ BEFORE any cleaning or
filtering, so every later variant (filtered, deduplicated, ...) can be
compared against a real baseline: how many documents, how much text, how
long the documents are, and how many look low quality.

Uses only built-in Spark SQL functions (no Python UDFs), so it is fast
and has no Python-worker overhead.

Token counts here are a ROUGH ESTIMATE (characters / 4, a common rule of
thumb for English text). Exact token counts come from the real tokenizer
in Phase 4.

Run:
    python baseline_stats.py                  # reads data/raw
    python baseline_stats.py --input some/dir # any Parquet dir with a `text` column

Writes the numbers to results/phase2/<name>_stats.json (small, safe to
commit) and prints a summary table you can paste into the Phase 2 report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1_ingestion"))
from spark_session import get_spark_session  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "phase2"

CHARS_PER_TOKEN = 4          # rough English rule of thumb
SHORT_DOC_CHARS = 200        # matches FilterThresholds.min_length_chars
LOW_ALPHA_RATIO = 0.7        # matches FilterThresholds.min_alpha_ratio


def compute_stats(df) -> dict:
    """Compute dataset-level statistics for a DataFrame with a `text` column."""
    df = df.select(
        F.length("text").alias("chars"),
        F.size(F.split(F.trim(F.col("text")), r"\s+")).alias("words"),
        (F.length(F.regexp_replace("text", r"[^A-Za-z]", "")) /
         F.greatest(F.length("text"), F.lit(1))).alias("alpha_ratio"),
    ).cache()

    agg = df.agg(
        F.count("*").alias("docs"),
        F.sum("chars").alias("total_chars"),
        F.sum("words").alias("total_words"),
        F.avg("chars").alias("mean_chars"),
        F.sum((F.col("chars") < SHORT_DOC_CHARS).cast("int")).alias("short_docs"),
        F.sum((F.col("alpha_ratio") < LOW_ALPHA_RATIO).cast("int")).alias("low_alpha_docs"),
        F.avg("alpha_ratio").alias("mean_alpha_ratio"),
    ).first()

    docs = agg["docs"]
    if not docs:                                   # empty dataset: nothing to measure
        df.unpersist()
        return {
            "documents": 0, "total_characters": 0, "total_words": 0, "estimated_tokens": 0,
            "token_estimate_method": f"characters / {CHARS_PER_TOKEN} (rough estimate)",
            "raw_text_size_mb": 0.0,
            "chars_per_doc": {"mean": 0.0, "p5": 0.0, "median": 0.0, "p95": 0.0, "max": 0},
            f"docs_shorter_than_{SHORT_DOC_CHARS}_chars": 0,
            f"docs_alpha_ratio_below_{LOW_ALPHA_RATIO}": 0,
            "mean_alpha_ratio": 0.0,
        }

    p5, p50, p95 = df.approxQuantile("chars", [0.05, 0.5, 0.95], 0.001)
    longest = df.agg(F.max("chars")).first()[0]
    df.unpersist()

    total_chars = agg["total_chars"] or 0
    return {
        "documents": docs,
        "total_characters": total_chars,
        "total_words": agg["total_words"],
        "estimated_tokens": int(total_chars / CHARS_PER_TOKEN),
        "token_estimate_method": f"characters / {CHARS_PER_TOKEN} (rough estimate)",
        "raw_text_size_mb": round(total_chars / 1_000_000, 2),
        "chars_per_doc": {
            "mean": round(agg["mean_chars"], 1),
            "p5": p5, "median": p50, "p95": p95, "max": longest,
        },
        f"docs_shorter_than_{SHORT_DOC_CHARS}_chars": agg["short_docs"],
        f"docs_alpha_ratio_below_{LOW_ALPHA_RATIO}": agg["low_alpha_docs"],
        "mean_alpha_ratio": round(agg["mean_alpha_ratio"], 3),
    }


def print_summary(name: str, s: dict) -> None:
    docs = s["documents"] or 1
    short = s[f"docs_shorter_than_{SHORT_DOC_CHARS}_chars"]
    low = s[f"docs_alpha_ratio_below_{LOW_ALPHA_RATIO}"]
    c = s["chars_per_doc"]
    print(f"\n=== Baseline statistics: {name} ===")
    print(f"Documents:               {s['documents']:,}")
    print(f"Raw text size:           {s['raw_text_size_mb']:,} MB ({s['total_characters']:,} characters)")
    print(f"Words:                   {s['total_words']:,}")
    print(f"Estimated tokens:        ~{s['estimated_tokens']:,}  ({s['token_estimate_method']})")
    print(f"Chars per doc:           mean {c['mean']:,} | p5 {c['p5']:,.0f} | median {c['median']:,.0f} | "
          f"p95 {c['p95']:,.0f} | max {c['max']:,}")
    print(f"Short docs (<{SHORT_DOC_CHARS} chars): {short:,} ({100 * short / docs:.1f}%)")
    print(f"Low alpha ratio (<{LOW_ALPHA_RATIO}):  {low:,} ({100 * low / docs:.1f}%)")
    print(f"Mean alpha ratio:        {s['mean_alpha_ratio']}")


def main():
    parser = argparse.ArgumentParser(description="Compute baseline statistics for a Parquet text dataset.")
    parser.add_argument("--input", default=str(DATA_DIR / "raw"), help="Parquet directory with a `text` column.")
    parser.add_argument("--name", default=None, help="Label for the output file (default: input folder name).")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    input_dir = Path(args.input)
    name = args.name or input_dir.name

    spark = get_spark_session(app_name="phase2-baseline-stats")
    try:
        df = spark.read.parquet(str(input_dir))
        stats = compute_stats(df)
    finally:
        spark.stop()

    print_summary(name, stats)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_stats.json"
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()