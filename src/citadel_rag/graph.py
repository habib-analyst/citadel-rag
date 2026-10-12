"""Contradiction graph over claims from multiple sources.

Nodes: atomic claims (with source id). Edges: NLI relation + score for every
cross-source claim pair. The graph is what lets citadel-rag *see*
disagreement instead of silently resolving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .nli import NLIRelation, Thresholds, classify, score_pair


@dataclass
class ClaimNode:
    id: int
    text: str
    source: str


@dataclass
class ClaimEdge:
    a: int
    b: int
    relation: NLIRelation
    score: float  # score of the winning relation


@dataclass
class ContradictionGraph:
    nodes: list[ClaimNode] = field(default_factory=list)
    edges: list[ClaimEdge] = field(default_factory=list)

    def contradictions(self) -> list[ClaimEdge]:
        return [e for e in self.edges if e.relation == NLIRelation.CONTRADICTION]

    def entailments(self) -> list[ClaimEdge]:
        return [e for e in self.edges if e.relation == NLIRelation.ENTAILMENT]

    def contradicted_nodes(self) -> set[int]:
        out: set[int] = set()
        for e in self.contradictions():
            out.add(e.a)
            out.add(e.b)
        return out

    def has_disagreement(self) -> bool:
        return len(self.contradictions()) > 0


def build_graph(sources: dict[str, list[str]],
                th: Thresholds | None = None) -> ContradictionGraph:
    """Build the contradiction graph for claims grouped by source.

    sources: {source_id: [claim, ...]}. Only cross-source pairs are scored
    (a source contradicting itself is a data problem, not a disagreement).
    """
    from .nli import DEFAULT_THRESHOLDS
    th = th or DEFAULT_THRESHOLDS
    g = ContradictionGraph()
    idx: list[tuple[int, str]] = []  # (node_id, source)
    for source, claims in sources.items():
        for c in claims:
            nid = len(g.nodes)
            g.nodes.append(ClaimNode(id=nid, text=c, source=source))
            idx.append((nid, source))
    n = len(idx)
    for i in range(n):
        for j in range(i + 1, n):
            ni, si = idx[i]
            nj, sj = idx[j]
            if si == sj:
                continue
            s = score_pair(g.nodes[ni].text, g.nodes[nj].text)
            rel = classify(g.nodes[ni].text, g.nodes[nj].text, th)
            g.edges.append(ClaimEdge(a=ni, b=nj, relation=rel,
                                     score=s[rel.value]))
    return g
