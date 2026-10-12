# citadel-rag

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/citadel-rag.svg)](https://github.com/habib-analyst/citadel-rag/stargazers)

**When your retrieved sources disagree, most RAG systems silently pick a side. citadel-rag refuses — it shows you both sides, with citations, and tells you it can't decide.**

## 30-second why

RAG pipelines cite sources, but they never check whether the sources *agree with each other*. On contested topics — dosages, dates, statistics — contradictory sources are the norm, and today's systems resolve them by accident (whichever chunk came first wins). citadel-rag builds a contradiction graph over atomic claims and answers honestly: agreement → cited answer; disagreement → both sides shown, confidence downgraded, never silently resolved.

## Quickstart

```bash
pip install citadel-rag
```

```python
from citadel_rag import answer_honestly

ans = answer_honestly(
    "How tall is the Eiffel Tower?",
    {
        "wikipedia": "The Eiffel Tower is 330 metres tall. It was completed in 1889.",
        "blog": "The Eiffel Tower is 210 metres tall. It is located in Paris.",
    },
)
print(ans.render())
```

Expected output:

```
Q: How tall is the Eiffel Tower?

SOURCES DISAGREE (shown, not resolved):
  - The Eiffel Tower is 330 metres tall [wikipedia]
  - The Eiffel Tower is 210 metres tall [blog]
  (contradiction score 1.00)

Confidence: low (sources disagree — both sides shown)
```

When sources agree, you get a normal cited answer with high confidence instead.

## Measured results

Contradiction detection on 102 labeled claim-pairs across 34 multi-source scenarios (dev 30 / test 72; labels from construction, fully auditable):

| System | Precision | Recall | F1 |
|---|---|---|---|
| **citadel-rag** (test) | 1.000 | 0.941 | **0.970** |
| TF-IDF similarity baseline (tuned) | 0.444 | 0.706 | 0.546 |

Kill-gate: F1 must beat the baseline by ≥10pp → **passed at +42.42pp**. One documented false negative (unit phrasing "C" vs "degrees Celsius" on short claims). Zero model downloads, zero APIs — a deterministic, calibrated NLI heuristic (negation flips, antonym clashes, numeric/year mismatches with same-quantity gating, minimal-pair entity swaps).

## How it works

1. **Extract** atomic claims per source (`claims.py`) — deterministic, no models.
2. **Score** every cross-source claim pair (`nli.py`) — four inspectable signals, thresholds tuned on a dev split.
3. **Build** the contradiction graph (`graph.py`) — nodes are claims, edges are scored NLI relations.
4. **Answer honestly** (`answer.py`) — key claims on contradicted edges trigger the disagreement path; otherwise a cited answer.

A cross-encoder NLI backend can replace the heuristic behind the same `score_pair` interface.

## Citation

```bibtex
@software{rehman2026citadelrag,
  author = {Rehman, Habib Ur},
  title = {citadel-rag: Citation-Contradiction Detection for RAG},
  year = {2026},
  url = {https://github.com/habib-analyst/citadel-rag}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
