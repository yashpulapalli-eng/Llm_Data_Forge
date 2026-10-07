"""
Phase 3: Deduplication Pipeline -- exact-match deduplication.

Removes documents whose text is identical after light normalization
(lower-casing and collapsing whitespace), using a SHA-256 content hash.
This is the cheap first pass; near-duplicates that differ by even a few
words are left for MinHash/LSH (minhash_lsh.py).

Implemented with built-in Spark SQL functions only (no Python UDFs), which
is faster and avoids the large-document UDF hang noted in Phase 2.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def add_content_hash(df: DataFrame, text_col: str = "text", hash_col: str = "_content_hash") -> DataFrame:
    """Add a SHA-256 hash of the normalized text (lower-cased, whitespace collapsed)."""
    normalized = F.lower(F.regexp_replace(F.trim(F.col(text_col)), r"\s+", " "))
    return df.withColumn(hash_col, F.sha2(normalized, 256))


def deduplicate_exact(df: DataFrame, text_col: str = "text") -> DataFrame:
    """
    Drop exact (normalized) duplicates, keeping ONE document per content hash:
    the one with the smallest doc_id. (doc_ids are random UUIDs, so which copy
    survives is arbitrary but deterministic -- re-running gives the same result.)
    """
    hashed = add_content_hash(df, text_col)
    w = Window.partitionBy("_content_hash").orderBy(F.col("doc_id"))
    return (hashed.withColumn("_rn", F.row_number().over(w))
                  .filter(F.col("_rn") == 1)
                  .drop("_rn", "_content_hash"))