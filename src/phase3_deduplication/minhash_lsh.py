"""
Phase 3: Deduplication Pipeline — MinHash + LSH near-duplicate detection.

Uses Spark ML's MinHashLSH to find and remove near-duplicate documents
that exact-match hashing (exact_dedup.py) won't catch.
"""

from dataclasses import dataclass

from pyspark.ml.feature import MinHashLSH
from pyspark.sql import DataFrame


@dataclass
class LSHConfig:
    """
    Tunable MinHash/LSH parameters. Document the chosen values (and
    why) in the Phase 3 report — they directly control the
    precision/recall tradeoff of near-duplicate detection.
    """
    num_hash_tables: int = 5          # more tables = higher recall, more compute
    num_features: int = 2 ** 18       # hashing-TF feature space size
    jaccard_threshold: float = 0.8    # similarity above which docs are "near-duplicate"
    shingle_size: int = 5             # word-shingle size used to build the token set


def build_shingles(df: DataFrame, text_col: str = "text", shingle_size: int = 5) -> DataFrame:
    """
    Convert each document's text into a set of word shingles (n-grams),
    the input representation MinHash operates on.

    TODO: implement proper word-shingling — a sliding window of
    `shingle_size` consecutive words per document, hashed into a
    sparse feature vector (see HashingTF) for MinHashLSH input.
    """
    raise NotImplementedError("Build word-shingle sets for MinHash input.")


def fit_minhash_lsh(df: DataFrame, config: LSHConfig, features_col: str = "features"):
    """
    Fit a MinHashLSH model over the (hashed) shingle features.

    Returns:
        The fitted MinHashLSH model, for use with approxSimilarityJoin
        or approxNearestNeighbors to find near-duplicate pairs.
    """
    mh = MinHashLSH(inputCol=features_col, outputCol="hashes",
                     numHashTables=config.num_hash_tables)
    return mh.fit(df)


def find_near_duplicate_clusters(df: DataFrame, config: LSHConfig) -> DataFrame:
    """
    End-to-end: shingle -> hash -> LSH self-join -> cluster near-duplicates.

    Returns:
        A DataFrame of (doc_id, cluster_id) assignments, where documents
        sharing a cluster_id are near-duplicates of each other.

    TODO:
        - Implement the approxSimilarityJoin self-join at config.jaccard_threshold.
        - Cluster the resulting pairwise matches (e.g. connected
          components) into cluster_ids.
        - Decide the keep-one-per-cluster policy (first seen? longest doc?).
    """
    raise NotImplementedError
