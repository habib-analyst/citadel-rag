"""Honest answering over a contradiction graph.

The core behavior: when the answer's key claims sit on contradicted edges,
the output SURFACES the disagreement — both sides, both citations, honest
confidence — instead of silently picking a winner. When sources agree, it
answers normally with citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .claims import extract_claims
from .nli import content_tokens
from .graph import ContradictionGraph, build_graph
from .nli import NLIRelation, Thresholds


@dataclass
class CitedClaim:
    text: str
    sources: list[str] = field(default_factory=list)


@dataclass
class Disagreement:
    claim_a: CitedClaim
    claim_b: CitedClaim
    score: float


@dataclass
class HonestAnswer:
    question: str
    answer_claims: list[CitedClaim] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)
    confidence: str = "unknown"  # high | medium | low (disputed)

    def render(self) -> str:
        lines = [f"Q: {self.question}", ""]
        if self.answer_claims:
            lines.append("Answer (sources agree):")
            for c in self.answer_claims:
                cites = ", ".join(f"[{s}]" for s in c.sources)
                lines.append(f"  - {c.text} {cites}")
            lines.append("")
        for d in self.disagreements:
            ca = ", ".join(f"[{s}]" for s in d.claim_a.sources)
            cb = ", ".join(f"[{s}]" for s in d.claim_b.sources)
            lines.append("SOURCES DISAGREE (shown, not resolved):")
            lines.append(f"  - {d.claim_a.text} {ca}")
            lines.append(f"  - {d.claim_b.text} {cb}")
            lines.append(f"  (contradiction score {d.score:.2f})")
            lines.append("")
        lines.append(f"Confidence: {self.confidence}")
        return "\n".join(lines)


def _relevance(question: str, claim: str) -> float:
    q = set(content_tokens(question))
    c = set(content_tokens(claim))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q)


def answer_honestly(question: str,
                    sources: dict[str, str],
                    th: Thresholds | None = None,
                    top_k: int = 5) -> HonestAnswer:
    """Answer a question from raw source texts, honestly.

    sources: {source_id: full_text}. Returns an HonestAnswer whose
    disagreements list is non-empty iff the key claims are contradicted.
    """
    claims_by_source = {s: extract_claims(t) for s, t in sources.items()}
    graph = build_graph(claims_by_source, th)

    # rank claims by question relevance
    ranked: list[tuple[float, int]] = []
    for node in graph.nodes:
        ranked.append((_relevance(question, node.text), node.id))
    ranked.sort(reverse=True)
    key_ids = [nid for _, nid in ranked[:top_k] if _ > 0] or \
        [nid for _, nid in ranked[:top_k]]

    contradicted = graph.contradicted_nodes()
    ans = HonestAnswer(question=question)

    # group key claims by source for citation
    for nid in key_ids:
        node = graph.nodes[nid]
        if nid in contradicted:
            continue
        # merge same-text claims across sources
        for c in ans.answer_claims:
            if c.text.lower() == node.text.lower():
                if node.source not in c.sources:
                    c.sources.append(node.source)
                break
        else:
            ans.answer_claims.append(
                CitedClaim(text=node.text, sources=[node.source]))

    for e in graph.contradictions():
        if e.a in key_ids or e.b in key_ids:
            na, nb = graph.nodes[e.a], graph.nodes[e.b]
            ans.disagreements.append(Disagreement(
                claim_a=CitedClaim(text=na.text, sources=[na.source]),
                claim_b=CitedClaim(text=nb.text, sources=[nb.source]),
                score=e.score,
            ))

    if ans.disagreements:
        ans.confidence = "low (sources disagree — both sides shown)"
    elif ans.answer_claims:
        ent = sum(1 for e in graph.entailments()
                  if e.a in key_ids or e.b in key_ids)
        ans.confidence = "high" if ent else "medium"
    else:
        ans.confidence = "unknown (no relevant claims)"
    return ans
