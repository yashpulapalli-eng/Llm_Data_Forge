"""
Phase 3: Deduplication Pipeline -- MinHash + LSH near-duplicate detection.

Finds documents that are *almost* identical (same article with a different
header, boilerplate, a few edited words) which exact hashing cannot catch.

Pipeline (all inside Spark, no Python UDFs):
  1. Shingling      each document -> set of overlapping word 5-grams
  2. Hashing        shingles -> sparse binary vector (HashingTF)
  3. MinHash + LSH  Spark ML MinHashLSH buckets similar documents together
  4. Self-join      approxSimilarityJoin keeps pairs whose true Jaccard
                    similarity >= config.jaccard_threshold
  5. Clustering     connected components over the matching pairs
  6. Resolution     keep ONE document per cluster (the longest; ties -> smallest doc_id)

Jaccard similarity of two documents = |shared shingles| / |all distinct shingles|.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml.feature import HashingTF, MinHashLSH, NGram
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


@dataclass
class LSHConfig:
    """
    Tunable MinHash/LSH parameters. Document the chosen values (and
    why) in the Phase 3 report -- they directly control the
    precision/recall tradeoff of near-duplicate detection.
    """
    num_hash_tables: int = 5          # more tables = higher recall, more compute
    num_features: int = 2 ** 18       # hashing-TF feature space size
    jaccard_threshold: float = 0.8    # similarity at/above which docs are "near-duplicate"
    shingle_size: int = 5             # word-shingle size used to build the token set
    max_cluster_iterations: int = 20  # safety cap for connected-components


def build_shingles(df: DataFrame, text_col: str = "text", shingle_size: int = 5,
                   num_features: int = 2 ** 18) -> DataFrame:
    """
    Add a `features` column (sparse binary vector of hashed word shingles)
    and `n_shingles` (how many shingles the document has).

    Text is lower-cased and punctuation removed before shingling, so
    "Hello, World!" and "hello world" produce the same shingles.
    Documents shorter than `shingle_size` words get n_shingles = 0 and are
    skipped by the near-duplicate search (exact dedup still covers them).
    """
    cleaned = F.trim(F.regexp_replace(F.lower(F.col(text_col)), r"[^\p{L}\p{N}\s]", " "))
    words_df = df.withColumn("_words", F.split(cleaned, r"\s+"))
    ngrams = NGram(n=shingle_size, inputCol="_words", outputCol="_shingles").transform(words_df)
    with_count = ngrams.withColumn("n_shingles", F.size("_shingles"))
    hashed = HashingTF(inputCol="_shingles", outputCol="features",
                       numFeatures=num_features, binary=True).transform(with_count)
    return hashed.drop("_words", "_shingles")


def fit_minhash_lsh(df: DataFrame, config: LSHConfig, features_col: str = "features"):
    """Fit a MinHashLSH model over the hashed shingle vectors."""
    mh = MinHashLSH(inputCol=features_col, outputCol="hashes",
                    numHashTables=config.num_hash_tables, seed=42)
    return mh.fit(df)


def find_near_duplicate_pairs(df: DataFrame, config: LSHConfig) -> DataFrame:
    """
    Return a DataFrame of near-duplicate pairs: (doc_id_a, doc_id_b, jaccard),
    one row per unordered pair with jaccard >= config.jaccard_threshold.
    """
    feats = (build_shingles(df.select("doc_id", "text"), "text", config.shingle_size, config.num_features)
             .filter(F.col("n_shingles") > 0)
             .select("doc_id", "features"))
    model = fit_minhash_lsh(feats, config)
    # approxSimilarityJoin works on Jaccard DISTANCE (= 1 - similarity), strict "<".
    max_distance = (1.0 - config.jaccard_threshold) + 1e-9
    joined = model.approxSimilarityJoin(feats, feats, max_distance, distCol="jaccard_distance")
    return (joined.select(F.col("datasetA.doc_id").alias("doc_id_a"),
                          F.col("datasetB.doc_id").alias("doc_id_b"),
                          (1.0 - F.col("jaccard_distance")).alias("jaccard"))
                  .filter(F.col("doc_id_a") < F.col("doc_id_b")))   # drop self-pairs and mirrored pairs


def cluster_pairs(pairs: DataFrame, max_iterations: int = 20) -> DataFrame:
    """
    Group documents into near-duplicate clusters = connected components of
    the pair graph (if A~B and B~C, then A, B, C share one cluster).

    Min-label propagation: every document starts as its own cluster label
    (its doc_id) and repeatedly adopts the smallest label among its
    neighbours until nothing changes.

    Returns (doc_id, cluster_id) for every document that appears in a pair.
    """
    a, b = "doc_id_a", "doc_id_b"
    sym = (pairs.select(F.col(a).alias("src"), F.col(b).alias("dst"))
                .union(pairs.select(F.col(b).alias("src"), F.col(a).alias("dst")))
                .distinct())
    nodes = sym.select(F.col("src").alias("doc_id")).distinct() \
               .withColumn("cluster_id", F.col("doc_id")).localCheckpoint()

    for i in range(max_iterations):
        neighbour_label = (sym.join(nodes.select(F.col("doc_id").alias("src"),
                                                 F.col("cluster_id").alias("src_label")), "src")
                              .groupBy("dst").agg(F.min("src_label").alias("nmin"))
                              .withColumnRenamed("dst", "doc_id"))
        updated = (nodes.join(neighbour_label, "doc_id", "left")
                        .select("doc_id",
                                F.least("cluster_id", F.coalesce("nmin", "cluster_id")).alias("cluster_id"))
                        .localCheckpoint())
        changed = (updated.alias("n").join(nodes.alias("o"), "doc_id")
                          .filter(F.col("n.cluster_id") != F.col("o.cluster_id")).count())
        nodes = updated
        if changed == 0:
            return nodes
    print(f"WARNING: clustering did not fully converge after {max_iterations} iterations.")
    return nodes


def find_near_duplicate_clusters(df: DataFrame, config: LSHConfig) -> DataFrame:
    """End-to-end convenience: shingle -> LSH join -> clusters. Returns (doc_id, cluster_id)."""
    pairs = find_near_duplicate_pairs(df, config)
    return cluster_pairs(pairs, config.max_cluster_iterations)


def remove_near_duplicates(df: DataFrame, clusters: DataFrame) -> DataFrame:
    """
    Keep every document that is in no cluster, plus ONE representative per
    cluster: the longest document (ties broken by smallest doc_id).
    """
    in_cluster = df.join(clusters, "doc_id", "inner")
    singles = df.join(clusters, "doc_id", "left_anti")
    w = Window.partitionBy("cluster_id").orderBy(F.length("text").desc(), F.col("doc_id"))
    reps = (in_cluster.withColumn("_rn", F.row_number().over(w))
                      .filter(F.col("_rn") == 1)
                      .drop("_rn", "cluster_id"))
    return singles.unionByName(reps)