"""
Shared Spark session/config used across all pipeline phases.

Import get_spark_session() from any phaseN script instead of building
a SparkSession ad hoc, so every phase runs against the same config.
"""

import glob
import os
import re
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def _java_exe_exists(java_home: str) -> bool:
    bin_dir = Path(java_home) / "bin"
    return (bin_dir / "java.exe").exists() or (bin_dir / "java").exists()


def ensure_java_home() -> None:
    """
    PySpark needs a Java runtime. If JAVA_HOME isn't visible to this process
    (common on Windows right after installing a JDK), look for an installed
    JDK in the usual locations and point JAVA_HOME / PATH at it.
    """
    current = os.environ.get("JAVA_HOME")
    if current and _java_exe_exists(current):
        return  # already fine

    if os.name != "nt":
        return  # on Mac/Linux, rely on the system Java

    patterns = [
        r"C:\Program Files\Eclipse Adoptium\jdk-*",
        r"C:\Program Files\Java\jdk-*",
        r"C:\Program Files\Microsoft\jdk-*",
        r"C:\Program Files\Amazon Corretto\jdk*",
        r"C:\Program Files\Zulu\zulu*",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    candidates = [c for c in candidates if _java_exe_exists(c)]
    if not candidates:
        return  # nothing found; PySpark will raise its usual error

    def version_key(path: str):
        return [int(n) for n in re.findall(r"\d+", Path(path).name)]

    java_home = sorted(candidates, key=version_key)[-1]
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ.get("PATH", "")


def ensure_hadoop_home() -> None:
    """
    On Windows, Hadoop (used by Spark to write files) needs winutils.exe.
    Look for it in the usual places and point HADOOP_HOME at it.
    Expected layout: C:\\hadoop\\bin\\winutils.exe
    """
    if os.name != "nt":
        return  # only needed on Windows

    def has_winutils(home: str) -> bool:
        return (Path(home) / "bin" / "winutils.exe").exists()

    hadoop_home = os.environ.get("HADOOP_HOME")
    if not (hadoop_home and has_winutils(hadoop_home)):
        hadoop_home = None
        for candidate in (r"C:\hadoop", r"C:\winutils", str(Path.home() / "hadoop")):
            if has_winutils(candidate):
                hadoop_home = candidate
                break
    if not hadoop_home:
        return  # not found; Spark will raise its usual winutils error

    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ["PATH"] = str(Path(hadoop_home) / "bin") + os.pathsep + os.environ.get("PATH", "")


def get_spark_session(app_name: str = "llm-data-forge", **config_overrides) -> SparkSession:
    """
    Build (or fetch the existing) SparkSession for this pipeline.

    Args:
        app_name: Spark application name, shown in the Spark UI.
        **config_overrides: Any spark.conf key/value pairs to set,
            e.g. get_spark_session(**{"spark.sql.shuffle.partitions": "200"}).
    """
    ensure_java_home()
    ensure_hadoop_home()

    # Make Spark's Python workers use the same interpreter that's running this script.
    # Without this, Windows resolves "python" to the Microsoft Store shortcut and
    # the workers never start ("Python worker failed to connect back").
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

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