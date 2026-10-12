# citadel-rag: Citation-Contradiction Detection for RAG

**Habib Ur Rehman** · October 2026

## Abstract

Retrieval-augmented generation systems fail silently when retrieved sources
disagree: the generator picks one side — usually the first or most fluent —
and presents it as fact, with citations attached. citadel-rag is built on a
different contract: **when sources disagree, the system must say so.**
It extracts atomic claims per source, scores every cross-source pair with a
deterministic NLI heuristic, builds a contradiction graph, and answers
honestly — both sides, both citations, calibrated confidence. On a labeled
set of 102 claim-pairs across 34 synthetic multi-source scenarios, it reaches
**0.97 F1** on contradiction detection, **42 points above** a tuned TF-IDF
similarity baseline.

## 1. The gap

Claim-level citation checking exists (RAGAS Faithfulness, trustrag): every
claim in the answer is verified against *some* retrieved span. What none of
them do is detect when two retrieved spans **contradict each other** and
change the answering behavior accordingly. The failure mode is real and
common: contradictory sources are the norm on contested topics (dosages,
dates, statistics), and current pipelines resolve them by accident —
position bias, fluency bias, recency bias — never by design.

## 2. Method

**Claim extraction** (`claims.py`). Deterministic sentence segmentation
plus clause splitting on strong discourse boundaries (`;`, `, which`, `,
where`), de-duplicated, minimum length 4 words.

**NLI heuristic** (`nli.py`). No model downloads, no APIs — four calibrated
signals, each designed to be inspectable:

1. **Numeric mismatch.** Both claims state a value for the same unit and the
   values differ. Years (1000–2100) are special: any difference contradicts,
   but only when the rest of the claim is essentially the same statement
   (non-year token Jaccard ≥ 0.6 or near-subset) *and* the differing
   predicates are not antonyms — so "ended in 1945" vs "began in 1939"
   correctly does *not* fire. Other quantities need >5% relative difference
   *and* a same-quantity gate: the non-numeric content must overlap
   (Jaccard ≥ 0.3) and any "of X" complements must agree, so "71% of Earth's
   surface" vs "97% of Earth's water" does not fire.
2. **Negation flip.** Exactly one side is negated and content overlap ≥ 0.55.
3. **Antonym clash.** An antonym pair (40-pair lexicon, stemmed) with
   residual overlap ≥ 0.5 — but suppressed when the claims state different
   numbers, since then the antonymy describes different events.
4. **Minimal-pair entity swap.** Overlap coefficient ≥ 0.6, exactly two
   differing content tokens, and *both* must be entity-like (numeric, or a
   capitalized word in the raw text that is not a sentence-initial common
   noun — "Paris"/"Lyon" yes, "Humans" no).

Supporting machinery: light consistent stemming (plural/inflectional -s,
-ed, -ing, a tiny irregular-verb map), proper nouns never stemmed, unit
normalization (kilometres→km, %→percent). Decision thresholds are
grid-searched on a dev split to maximize contradiction F1.

**Contradiction graph** (`graph.py`). Nodes are claims (with source ids);
edges are scored NLI relations for every cross-source pair. Same-source
pairs are never scored.

**Honest answering** (`answer.py`). Claims are ranked by question relevance;
key claims sitting on contradicted edges trigger the disagreement path:
both claims, both citations, contradiction score, confidence "low (sources
disagree — both sides shown)". Otherwise a normal cited answer with
confidence high/medium from entailment support.

## 3. Evaluation

**Test set** (`eval.py`). 34 synthetic multi-source scenarios, 102 labeled
claim-pairs (dev 30 / test 72; test: 17 contradiction, 17 entailment, 38
neutral). Labels derive from construction: same fact-id + same stance →
entailment; same fact-id + opposing stance → contradiction; different
fact-ids → neutral. Fully deterministic and auditable — no LLM-generated
labels. Contradiction types covered: negation, antonym, numeric, year,
entity swap. Dev/test split is by scenario, stratified.

**Baseline.** TF-IDF cosine similarity with a tuned band (lo, hi):
contradiction iff similarity ∈ [lo, hi], tuned on dev — the standard naive
approach.

| System | Precision | Recall | F1 |
|---|---|---|---|
| citadel-rag (test, n=72) | 1.000 | 0.941 | **0.970** |
| TF-IDF baseline (test) | 0.444 | 0.706 | 0.546 |

**Kill-gate: PASS** — F1 gap **42.42pp** (bar: 10pp). Dev F1 0.875, tuned
thresholds τ_contra=0.5, τ_entail=0.5.

## 4. Limitations (honest)

- **One false negative** (test): "Absolute zero equals -273.15 C" vs "…minus
  150 degrees Celsius" — unit phrasing mismatch ("C" vs "degrees Celsius")
  plus short claims defeat the same-quantity gate. A cross-encoder NLI
  backend (documented extension point behind the same `score_pair`
  interface) would likely catch it.
- **Synthetic evaluation.** The 102-pair set is hand-constructed; real
  retrieval noise (OCR errors, tables, long documents) is not covered.
- **English only.** The lexicons and heuristics are English-specific.
- **Claim extraction is naive.** Sentence splitting misses cross-sentence
  coreference ("It was founded in 1928" loses its subject).

## 5. Related work

RAGAS (Faithfulness), trustrag, and hermes-cmx do claim-level citation
*verification* — none builds a cross-source contradiction graph or changes
answering behavior on disagreement. NLI-based contradiction detection
exists for single-pair classification (SNLI/MNLI fine-tunes); citadel-rag's
contribution is the graph + honest-answering contract, not the classifier.

## 6. Reproduce

```bash
pip install citadel-rag          # or: pip install -e .
PYTHONPATH=src python3 -m citadel_rag.eval results
python3 -m pytest tests/ -q     # 17 tests
PYTHONPATH=src python3 examples/quickstart.py
```

All numbers above are produced by `eval.py` and pinned by
`test_eval_kill_gate`.
