"""
Phase 2: Cleaning & Quality Filtering Pipeline -- text cleaning.

Non-judgmental cleanup applied BEFORE quality filtering: strip leftover
HTML/markup, fix encoding artifacts, and normalize whitespace.

Cleaning never deletes a whole document -- that is the filters' job
(quality_filters.py). It only makes the text itself tidier.

Design note: line breaks are PRESERVED (only runs of 3+ blank lines are
collapsed). The repetition filter works on lines/paragraphs, and models
learn document structure from them, so flattening everything into one
long line would throw useful signal away.
"""

from __future__ import annotations

import html
import re
import unicodedata

from pyspark.sql import Column
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Only match things that look like real tags (<p>, </div>, <br/>, <a href=...>),
# not text such as "if a < b and c > d".
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
# Control characters except newline (\n) and tab (\t).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Invisible / junk characters: zero-width chars, BOM, Unicode replacement char.
_JUNK_RE = re.compile("[​‌‍⁠﻿�]")
_HSPACE_RE = re.compile(r"[^\S\n]+")      # any whitespace except newline
_BLANK_LINES_RE = re.compile(r"\n{3,}")   # 3+ newlines in a row


def strip_html(text: str) -> str:
    """Remove HTML/markup tags from a raw text string."""
    if text is None:
        return text
    return _HTML_TAG_RE.sub(" ", text)


def fix_encoding(text: str) -> str:
    """
    Fix common encoding artifacts:
      - HTML entities (&amp; &nbsp; &#39; ...) -> real characters
      - Unicode normalization (NFC) so the same character has one form
      - stray control characters, zero-width characters, BOMs, U+FFFD
      - classic mojibake ("â€™" for an apostrophe) via ftfy, if installed
        (optional: `pip install ftfy`; skipped silently if missing)
    """
    if text is None:
        return text
    text = html.unescape(text)
    try:
        import ftfy  # optional dependency
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub("", text)
    text = _JUNK_RE.sub("", text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse repeated spaces/tabs inside each line, strip each line,
    and squeeze 3+ consecutive newlines down to a single blank line.
    Line breaks themselves are kept.
    """
    if text is None:
        return text
    lines = [_HSPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline for one document (pure Python, easy to test)."""
    if text is None:
        return ""
    text = strip_html(text)
    text = fix_encoding(text)
    text = normalize_whitespace(text)
    return text


def clean_text_udf() -> Column:
    """
    Spark UDF wrapping clean_text(), for use as
    df.withColumn("text", clean_text_udf()(col("text"))).

    useArrow=False on purpose: with PySpark 4.x + pandas 3.x the default
    Arrow-optimized UDF path can hang on very large documents (hundreds of
    thousands of characters). The plain Python UDF path does not.
    """
    return udf(clean_text, StringType(), useArrow=False)