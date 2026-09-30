"""
Phase 2: Cleaning & Quality Filtering Pipeline — text cleaning.

Boilerplate/HTML stripping, encoding normalization, and other
non-judgmental cleanup applied before quality filtering.
"""

import re

from pyspark.sql import Column
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML/markup tags from a raw text string."""
    if text is None:
        return text
    return _HTML_TAG_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing space."""
    if text is None:
        return text
    return _WHITESPACE_RE.sub(" ", text).strip()


def fix_encoding(text: str) -> str:
    """
    Fix common encoding artifacts (mojibake, stray control characters).

    TODO:
        - Decide whether to use a library (e.g. ftfy) or handle a
          fixed set of known artifacts from the chosen corpus.
    """
    raise NotImplementedError("Implement encoding-fix logic for the chosen corpus.")


def clean_text_udf() -> Column:
    """
    Spark UDF wrapping the cleaning steps above, for use in a
    DataFrame .withColumn("text", clean_text_udf()(col("text"))) call.
    """
    def _clean(text):
        text = strip_html(text)
        text = normalize_whitespace(text)
        return text
    return udf(_clean, StringType())
