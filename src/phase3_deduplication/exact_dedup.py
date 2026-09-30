"""
Phase 3: Deduplication Pipeline — exact-match deduplication.

Removes byte-identical (or normalized-identical) documents using a
content hash, before near-duplicate detection via MinHash/LSH.
"""

import hashlib

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType


def content_hash(text: str) -> str:
    """SHA-256 hash of normalized document text, used as an exact-dup key."""
    if text is None:
        return ""
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_hash_udf():
    return udf(content_hash, StringType())


def deduplicate_exact(df: DataFrame, text_col: str = "text") -> DataFrame:
    """
    Drop exact (normalized) duplicate documents, keeping the first
    occurrence of each content hash.
    """
    hashed = df.withColumn("_content_hash", content_hash_udf()(col(text_col)))
    deduped = hashed.dropDuplicates(["_content_hash"]).drop("_content_hash")
    return deduped
