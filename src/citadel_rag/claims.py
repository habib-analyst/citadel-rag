"""Atomic claim extraction.

Splits source text into atomic, self-contained claims. Deterministic and
dependency-free: sentence segmentation plus clause splitting on strong
discourse boundaries. The unit of analysis for the contradiction graph.
"""

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_CLAUSE_SPLIT = re.compile(r"\s*;\s*|\s+,\s+which\s+|\s+,\s+where\s+")


def _clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_claims(text: str, min_words: int = 4) -> list[str]:
    """Extract atomic claims from a source text.

    Returns a list of claim strings in document order. Claims shorter than
    ``min_words`` are dropped (headings, fragments).
    """
    if not text or not text.strip():
        return []
    claims: list[str] = []
    for sent in _SENT_SPLIT.split(text.strip()):
        for clause in _CLAUSE_SPLIT.split(sent):
            c = _clean(clause)
            # strip trailing sentence punctuation for canonical form
            c = c.rstrip(".!?").strip()
            if len(c.split()) >= min_words:
                claims.append(c)
    # de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in claims:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
