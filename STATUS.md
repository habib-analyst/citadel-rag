# citadel-rag — STATUS

## Project
Citation-contradiction detection for RAG. PIVOTED invention (2026-10-12):
claim-level NLI + verbatim citations are covered (RAGAS, trustrag). The gap:
when retrieved sources DISAGREE, systems silently pick one. citadel-rag
builds a contradiction graph across sources and surfaces disagreement honestly.

## Pipeline
- [x] Setup + START logged
- [ ] claims.py (atomic claim extraction)
- [ ] nli.py (deterministic entailment heuristic + calibration)
- [ ] graph.py (contradiction graph)
- [ ] answer.py (honest answering)
- [ ] eval.py (labeled test set ~100 pairs / ~30 scenarios + baseline)
- [ ] Run eval, check kill-gate (F1 ≥ baseline + 10pp)
- [ ] tests (pytest)
- [ ] paper/PAPER.md
- [ ] README.md (branded)
- [ ] examples/quickstart.py
- [ ] LinkedIn draft
- [ ] push payload staged
- [ ] DONE → report to parent

## Lifeline
If killed: resume at first unchecked box. Eval writes results/summary.json.
DONE marker: results/PIPELINE_DONE.
