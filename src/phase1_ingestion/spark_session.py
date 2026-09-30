"""
Shared Spark session/config used across all pipeline phases.

Import get_spark_session() from any phaseN script instead of building
a SparkSession ad hoc, so every phase runs against the same config.
"""

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "llm-data-forge", **config_overrides) -> SparkSession:
    """
    Build (or fetch the existing) SparkSession for this pipeline.

    Args:
        app_name: Spark application name, shown in the Spark UI.
        **config_overrides: Any spark.conf key/value pairs to set,
            e.g. get_spark_session(**{"spark.sql.shuffle.partitions": "200"}).

    Returns:
        A configured SparkSession.

    TODO:
        - Decide local vs. cluster deploy mode (local[*] for dev,
          a real cluster/master URL for the full-scale run).
        - Tune spark.sql.shuffle.partitions and executor memory/cores
          once the actual corpus size is known.
    """
    builder = SparkSession.builder.appName(app_name)

    # Sensible dev defaults — override for cluster runs.
    default_config = {
        "spark.sql.shuffle.partitions": "8",
        "spark.sql.parquet.compression.codec": "snappy",
    }
    default_config.update(config_overrides)

    for key, value in default_config.items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


if __name__ == "__main__":
    spark = get_spark_session()
    print(f"Spark session started: {spark.sparkContext.appName} (version {spark.version})")
    spark.stop()
