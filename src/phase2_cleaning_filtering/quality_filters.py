"""
Phase 2: Cleaning & Quality Filtering Pipeline -- quality heuristics.

Each filter takes (cleaned) document text and returns True if the
document should be KEPT. filter_flags() runs every filter and returns
all results, so the Spark job can report exactly WHY documents were
removed (the Phase 2 report asks for this).

All heuristics are simple, deterministic and need no extra installs.
They follow the style of the Gopher / C4 / FineWeb rule-based filters
(see the references in the project README).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class FilterThresholds:
    """
    Tunable thresholds for the quality filters below.
    Document the chosen values (and why) in the Phase 2 report.
    """
    # Length (characters): drop tiny fragments and absurdly long dumps.
    min_length_chars: int = 200
    max_length_chars: int = 100_000
    # Fraction of characters that are letters (drops number/symbol soup).
    min_alpha_ratio: float = 0.7
    # Repetition (Gopher-style): drop docs dominated by repeated lines/phrases.
    max_duplicate_line_ratio: float = 0.30        # fraction of lines that repeat an earlier line
    max_duplicate_line_char_ratio: float = 0.30   # fraction of characters inside repeated lines
    max_top_trigram_char_ratio: float = 0.18      # fraction of characters in the most frequent 3-word phrase
    min_words_for_ngram_check: int = 50           # skip the n-gram check on very short docs
    # "Is this English?" -- must contain at least this many distinct common English words,
    # but only docs with at least `min_words_for_language_check` words are judged
    # (the stopword rule is unreliable on tiny snippets; the length filter handles those).
    min_english_stopwords: int = 2
    min_words_for_language_check: int = 50


# Gopher's English stopword test: a real English document contains
# at least two of these very common words.
ENGLISH_STOPWORDS = frozenset(["the", "be", "to", "of", "and", "that", "have", "with"])
_WORD_RE = re.compile(r"[a-z']+")


# ---------------------------------------------------------------- filters

def length_filter(text: str, thresholds: FilterThresholds) -> bool:
    """Keep documents within a reasonable character-length range."""
    n = len(text) if text else 0
    return thresholds.min_length_chars <= n <= thresholds.max_length_chars


def alpha_ratio_filter(text: str, thresholds: FilterThresholds) -> bool:
    """Keep documents where a large-enough fraction of characters are alphabetic."""
    if not text:
        return False
    alpha_chars = sum(c.isalpha() for c in text)
    return (alpha_chars / max(len(text), 1)) >= thresholds.min_alpha_ratio


def repetition_filter(text: str, thresholds: FilterThresholds) -> bool:
    """
    Keep documents that are not dominated by repeated content.
    A document FAILS if any of these is too high:
      1. the fraction of non-empty lines that duplicate an earlier line
      2. the fraction of characters inside those duplicated lines
      3. the fraction of characters covered by the single most frequent
         3-word phrase (only checked on docs with enough words)
    """
    if not text:
        return False

    # 1 & 2: duplicate lines
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2:
        seen = set()
        dup_lines = 0
        dup_chars = 0
        for ln in lines:
            if ln in seen:
                dup_lines += 1
                dup_chars += len(ln)
            else:
                seen.add(ln)
        total_chars = sum(len(ln) for ln in lines)
        if dup_lines / len(lines) > thresholds.max_duplicate_line_ratio:
            return False
        if total_chars and dup_chars / total_chars > thresholds.max_duplicate_line_char_ratio:
            return False

    # 3: most frequent 3-word phrase
    words = text.split()
    if len(words) >= thresholds.min_words_for_ngram_check:
        trigrams = Counter(zip(words, words[1:], words[2:]))
        trigram, count = trigrams.most_common(1)[0]
        if count > 1:
            total_word_chars = sum(len(w) for w in words)
            covered = count * sum(len(w) for w in trigram)
            if covered / max(total_word_chars, 1) > thresholds.max_top_trigram_char_ratio:
                return False

    return True


def language_filter(text: str, thresholds: FilterThresholds) -> bool:
    """
    Keep documents that look like English prose.

    Uses the Gopher stopword rule: the text must contain at least
    `min_english_stopwords` of a small set of very common English words.
    It is a cheap, dependency-free stand-in for a language-ID model
    (fastText/langdetect are awkward to install on Windows). C4 is already
    language-filtered upstream, so expect this to remove very little.

    Spot-checking the first run showed the rule wrongly fails short English
    snippets (e.g. "Thank you VERY much!"), so documents with fewer than
    `min_words_for_language_check` words are not judged here at all -- the
    length filter already decides about them.
    """
    if not text:
        return False
    if len(text.split()) < thresholds.min_words_for_language_check:
        return True
    found = set(_WORD_RE.findall(text.lower())) & ENGLISH_STOPWORDS
    return len(found) >= thresholds.min_english_stopwords


# ---------------------------------------------------------------- combined

FILTER_NAMES = ("length", "alpha_ratio", "repetition", "language")


def filter_flags(text: str, thresholds: FilterThresholds) -> dict:
    """Run EVERY filter (no short-circuit) and return {filter_name: kept?}."""
    text = text or ""
    return {
        "length": length_filter(text, thresholds),
        "alpha_ratio": alpha_ratio_filter(text, thresholds),
        "repetition": repetition_filter(text, thresholds),
        "language": language_filter(text, thresholds),
    }


def passes_all_filters(text: str, thresholds: FilterThresholds) -> bool:
    """Convenience: True only if the document passes every filter above."""
    return all(filter_flags(text, thresholds).values())