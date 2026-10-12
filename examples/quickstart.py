"""Quickstart: contradiction detection + honest answering.

Run:  PYTHONPATH=src python3 examples/quickstart.py
"""

from citadel_rag import build_graph, answer_honestly
from citadel_rag.nli import tune_thresholds
from citadel_rag.eval import SCENARIOS, make_pairs, DEV_IDS

# Calibrate thresholds on the dev split (deterministic, no downloads)
dev = [s for s in SCENARIOS if s["id"] in DEV_IDS]
th = tune_thresholds(make_pairs(dev))
print(f"calibrated thresholds: contra>={th.tau_contra}, entail>={th.tau_entail}\n")

# --- Case 1: sources DISAGREE -> disagreement is surfaced, not resolved ---
ans = answer_honestly(
    "How tall is the Eiffel Tower?",
    {
        "wikipedia": "The Eiffel Tower is 330 metres tall. It was completed in 1889.",
        "blog": "The Eiffel Tower is 210 metres tall. It is located in Paris.",
    },
    th,
)
print(ans.render())

print("\n" + "=" * 60 + "\n")

# --- Case 2: sources AGREE -> normal cited answer ---
ans2 = answer_honestly(
    "What is the capital of France?",
    {
        "wikipedia": "Paris is the capital of France. It has 2.1 million people.",
        "travel-guide": "The capital of France is Paris. The city is known for art.",
    },
    th,
)
print(ans2.render())
