"""
Phase 2: Cleaning & Quality Filtering Pipeline — quality heuristics.

Each filter takes document text and returns True if the document
should be KEPT. Combine filters in run_filtering.py.
"""

from dataclasses import dataclass


@dataclass
class FilterThresholds:
    """
    Tunable thresholds for the quality filters below.
    Document the chosen values (and why) in the Phase 2 report.
    """
    min_length_chars: int = 200
    max_length_chars: int = 100_000
    max_repetition_ratio: float = 0.3
    min_alpha_ratio: float = 0.7
    allowed_languages: tuple = ("en",)


def length_filter(text: str, thresholds: FilterThresholds) -> bool:
    """Keep documents within a reasonable character-length range."""
    n = len(text) if text else 0
    return thresholds.min_length_chars <= n <= thresholds.max_length_chars


def repetition_filter(text: str, thresholds: FilterThresholds) -> bool:
    """
    Keep documents that aren't dominated by a single repeated line/phrase.

    TODO: implement a repetition-ratio heuristic (e.g. fraction of
    duplicate lines, or top-n-gram frequency) — see Lee et al. 2022
    and the Gopher/FineWeb filtering heuristics in the references.
    """
    raise NotImplementedError


def alpha_ratio_filter(text: str, thresholds: FilterThresholds) -> bool:
    """Keep documents where a large-enough fraction of characters are alphabetic."""
    if not text:
        return False
    alpha_chars = sum(c.isalpha() for c in text)
    return (alpha_chars / max(len(text), 1)) >= thresholds.min_alpha_ratio


def language_filter(text: str, thresholds: FilterThresholds) -> bool:
    """
    Keep documents in an allowed language.

    TODO: wire in a language-ID library (e.g. fastText lid.176, langdetect).
    """
    raise NotImplementedError


def passes_all_filters(text: str, thresholds: FilterThresholds) -> bool:
    """Convenience: True only if the document passes every filter above."""
    return (
        length_filter(text, thresholds)
        and alpha_ratio_filter(text, thresholds)
        and repetition_filter(text, thresholds)
        and language_filter(text, thresholds)
    )
