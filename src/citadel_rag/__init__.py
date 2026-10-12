"""citadel-rag: citation-contradiction detection for RAG.

When retrieved sources disagree, most RAG systems silently pick one side.
citadel-rag builds a contradiction graph over atomic claims and surfaces
disagreement honestly — both sides, both citations, calibrated confidence.
"""

from .claims import extract_claims
from .nli import score_pair, NLIRelation
from .graph import ContradictionGraph, build_graph
from .answer import answer_honestly, HonestAnswer

__all__ = [
    "extract_claims",
    "score_pair",
    "NLIRelation",
    "ContradictionGraph",
    "build_graph",
    "answer_honestly",
    "HonestAnswer",
]

__version__ = "0.1.0"
