"""
Phase 3: Deduplication Pipeline -- Spark job entry point.

Reads data/raw/ and data/filtered/ (Phase 2 output) and produces:
  - data/deduplicated/            raw      -> exact dedup -> near-duplicate removal
  - data/filtered_deduplicated/   filtered -> exact dedup -> near-duplicate removal

Run:
    python run_dedup.py
    python run_dedup.py --threshold 0.7 --tables 8     # try other LSH settings

Outputs:
    data/deduplicated/, data/filtered_deduplicated/    Parquet (gitignored)
    results/phase3/dedup_summary.json                  numbers for the report (commit this)
    results/phase3/near_dup_samples_<variant>.jsonl    example near-duplicate pairs (local only, gitignored)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from pyspark.sql import functions as F

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[0] / "phase1_ingestion"))
from spark_session import get_spark_session  # noqa: E402

from exact_dedup import deduplicate_exact  # noqa: E402
from minhash_lsh import LSHConfig, cluster_pairs, find_near_duplicate_pairs, remove_near_duplicates  # noqa: E402

REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "phase3"

SAMPLE_PAIRS = 10
SAMPLE_CHARS = 400


def size_stats(df) -> dict:
    row = df.agg(F.count("*").alias("docs"), F.sum(F.length("text")).alias("chars")).first()
    chars = row["chars"] or 0
    return {"documents": row["docs"], "text_mb": round(chars / 1e6, 2), "estimated_tokens": int(chars / 4)}


def dedup_variant(spark, name: str, input_dir: Path, output_dir: Path, config: LSHConfig, results_dir: Path) -> dict:
    df = spark.read.parquet(str(input_dir))
    start = size_stats(df)

    # Stage 1: exact duplicates
    df_exact = deduplicate_exact(df).cache()
    after_exact = size_stats(df_exact)

    # Stage 2: near-duplicates (MinHash + LSH)
    pairs = find_near_duplicate_pairs(df_exact, config).cache()
    n_pairs = pairs.count()
    clusters = cluster_pairs(pairs, config.max_cluster_iterations).cache() if n_pairs else None

    if clusters is not None:
        df_final = remove_near_duplicates(df_exact, clusters)
        cluster_sizes = clusters.groupBy("cluster_id").count()
        agg = cluster_sizes.agg(F.count("*").alias("n"), F.max("count").alias("largest")).first()
        n_clusters, largest = agg["n"], agg["largest"]
        docs_in_clusters = clusters.count()
        write_samples(pairs, df_exact, results_dir / f"near_dup_samples_{name}.jsonl")
    else:
        df_final = df_exact
        n_clusters = largest = docs_in_clusters = 0

    df_final.write.mode("overwrite").parquet(str(output_dir))
    final = size_stats(spark.read.parquet(str(output_dir)))   # what later phases will actually read

    summary = {
        "input": str(input_dir.name),
        "start": start,
        "after_exact_dedup": after_exact,
        "near_duplicate_pairs": n_pairs,
        "near_duplicate_clusters": n_clusters,
        "documents_in_clusters": docs_in_clusters,
        "largest_cluster": largest,
        "final": final,
        "removed_exact": start["documents"] - after_exact["documents"],
        "removed_near": after_exact["documents"] - final["documents"],
        "removed_total": start["documents"] - final["documents"],
        "pct_removed": round(100 * (start["documents"] - final["documents"]) / max(start["documents"], 1), 2),
    }
    df_exact.unpersist()
    pairs.unpersist()
    return summary


def write_samples(pairs, df, path: Path) -> None:
    """Save a few near-duplicate pairs (with text previews) for spot-checking."""
    top = pairs.orderBy(F.col("jaccard").asc()).limit(SAMPLE_PAIRS)    # lowest-similarity matches = the interesting borderline cases
    a = df.select(F.col("doc_id").alias("doc_id_a"), F.col("text").alias("text_a"))
    b = df.select(F.col("doc_id").alias("doc_id_b"), F.col("text").alias("text_b"))
    rows = top.join(a, "doc_id_a").join(b, "doc_id_b").collect()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "jaccard": round(r["jaccard"], 3),
                "doc_a": {"doc_id": r["doc_id_a"], "chars": len(r["text_a"]), "preview": r["text_a"][:SAMPLE_CHARS]},
                "doc_b": {"doc_id": r["doc_id_b"], "chars": len(r["text_b"]), "preview": r["text_b"][:SAMPLE_CHARS]},
            }, ensure_ascii=False) + "\n")


def print_summary(name: str, s: dict) -> None:
    st, ex, fi = s["start"], s["after_exact_dedup"], s["final"]
    print(f"\n=== Dedup summary: {name} (input: {s['input']}) ===")
    print(f"  {'stage':<28}{'documents':>12}{'text MB':>10}{'est. tokens':>14}")
    print(f"  {'start':<28}{st['documents']:>12,}{st['text_mb']:>10}{st['estimated_tokens']:>14,}")
    print(f"  {'after exact dedup':<28}{ex['documents']:>12,}{ex['text_mb']:>10}{ex['estimated_tokens']:>14,}")
    print(f"  {'after near-dup removal':<28}{fi['documents']:>12,}{fi['text_mb']:>10}{fi['estimated_tokens']:>14,}")
    print(f"  Removed: {s['removed_exact']:,} exact + {s['removed_near']:,} near-duplicate "
          f"= {s['removed_total']:,} ({s['pct_removed']}%)")
    print(f"  Near-duplicate pairs: {s['near_duplicate_pairs']:,} | clusters: {s['near_duplicate_clusters']:,} "
          f"| docs in clusters: {s['documents_in_clusters']:,} | largest cluster: {s['largest_cluster']}")


def main():
    d = LSHConfig()
    parser = argparse.ArgumentParser(description="Run exact + near-duplicate dedup (MinHash/LSH).")
    parser.add_argument("--raw-input", default=str(DATA_DIR / "raw"))
    parser.add_argument("--filtered-input", default=str(DATA_DIR / "filtered"))
    parser.add_argument("--dedup-output", default=str(DATA_DIR / "deduplicated"))
    parser.add_argument("--filtered-dedup-output", default=str(DATA_DIR / "filtered_deduplicated"))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--threshold", type=float, default=d.jaccard_threshold, help="Jaccard similarity for near-duplicates.")
    parser.add_argument("--tables", type=int, default=d.num_hash_tables, help="Number of MinHash tables.")
    parser.add_argument("--shingle", type=int, default=d.shingle_size, help="Words per shingle.")
    args = parser.parse_args()

    config = LSHConfig(jaccard_threshold=args.threshold, num_hash_tables=args.tables, shingle_size=args.shingle)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    spark = get_spark_session(app_name="phase3-dedup")
    try:
        summaries = {
            "deduplicated": dedup_variant(spark, "deduplicated", Path(args.raw_input),
                                          Path(args.dedup_output), config, results_dir),
            "filtered_deduplicated": dedup_variant(spark, "filtered_deduplicated", Path(args.filtered_input),
                                                   Path(args.filtered_dedup_output), config, results_dir),
        }
    finally:
        spark.stop()

    (results_dir / "dedup_summary.json").write_text(json.dumps({"config": asdict(config), **summaries}, indent=2))
    for name, s in summaries.items():
        print_summary(name, s)
    print(f"\nSummary saved to {results_dir / 'dedup_summary.json'}")
    print(f"Example near-duplicate pairs (for spot-checking): {results_dir}")


if __name__ == "__main__":
    main()