"""Tests for citadel-rag. All deterministic — no network, no models."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from citadel_rag.claims import extract_claims
from citadel_rag.nli import (
    score_pair, classify, tune_thresholds, Thresholds, NLIRelation,
    tokenize, content_tokens,
)
from citadel_rag.graph import build_graph
from citadel_rag.answer import answer_honestly
from citadel_rag.eval import SCENARIOS, make_pairs, TEST_IDS, run_eval


def test_extract_claims_basic():
    claims = extract_claims("Paris is the capital of France. It has 2.1 million people.")
    assert claims == ["Paris is the capital of France", "It has 2.1 million people"]


def test_extract_claims_empty():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_extract_claims_dedup():
    claims = extract_claims("Dogs bark loudly at night. Dogs bark loudly at night.")
    assert claims == ["Dogs bark loudly at night"]


def test_nli_negation_contradiction():
    s = score_pair("Statins reduce the risk of heart attack.",
                   "Statins do not reduce the risk of heart attack.")
    assert s["contradiction"] >= 0.75, s


def test_nli_numeric_contradiction():
    s = score_pair("The Eiffel Tower is 330 metres tall.",
                   "The Eiffel Tower is 210 metres tall.")
    assert s["contradiction"] >= 0.75, s


def test_nli_year_contradiction():
    s = score_pair("Humans first landed on the Moon in 1969.",
                   "Humans first landed on the Moon in 1972.")
    assert s["contradiction"] >= 0.75, s


def test_nli_year_different_events_no_fire():
    # ended-1945 vs began-1939: different facts, must NOT contradict
    s = score_pair("World War II ended in 1945.", "World War II began in 1939.")
    assert s["contradiction"] < 0.5, s


def test_nli_same_quantity_gate():
    # 71% of surface vs 97% of water: different quantities, must NOT fire
    s = score_pair("Water covers 71 percent of Earth's surface.",
                   "Oceans hold 97 percent of Earth's water.")
    assert s["contradiction"] < 0.5, s


def test_nli_antonym_contradiction():
    s = score_pair("Smoking increases the risk of lung cancer.",
                   "Smoking decreases the risk of lung cancer.")
    assert s["contradiction"] >= 0.75, s


def test_nli_entailment_paraphrase():
    s = score_pair("Paris is the capital of France.",
                   "The capital of France is Paris.")
    assert s["entailment"] >= 0.75, s
    assert classify("Paris is the capital of France.",
                    "The capital of France is Paris.") == NLIRelation.ENTAILMENT


def test_nli_neutral():
    s = score_pair("Paris is the capital of France.",
                   "The Nile flows through Egypt.")
    assert classify("Paris is the capital of France.",
                    "The Nile flows through Egypt.") == NLIRelation.NEUTRAL


def test_graph_cross_source_only():
    g = build_graph({"s1": ["Paris is the capital of France."],
                     "s2": ["Lyon is the capital of France."]})
    assert len(g.nodes) == 2
    assert len(g.edges) == 1  # only the cross-source pair
    assert g.has_disagreement()


def test_graph_no_self_edges():
    g = build_graph({"s1": ["Cats sleep.", "Dogs bark."]})
    assert len(g.edges) == 0  # same source: no edges


def test_answer_surfaces_disagreement():
    ans = answer_honestly(
        "What is the capital of France?",
        {"src-a": "Paris is the capital of France.",
         "src-b": "Lyon is the capital of France."})
    assert len(ans.disagreements) >= 1
    assert "disagree" in ans.confidence
    rendered = ans.render()
    assert "SOURCES DISAGREE" in rendered
    assert "[src-a]" in rendered and "[src-b]" in rendered


def test_answer_agreement_cites():
    ans = answer_honestly(
        "What is the capital of France?",
        {"src-a": "Paris is the capital of France.",
         "src-b": "The capital of France is Paris."})
    assert len(ans.disagreements) == 0
    assert len(ans.answer_claims) >= 1
    assert ans.confidence in ("high", "medium")


def test_eval_pair_counts():
    pairs = make_pairs(SCENARIOS)
    assert len(pairs) == 102, len(pairs)
    test_pairs = make_pairs([s for s in SCENARIOS if s["id"] in TEST_IDS])
    assert len(test_pairs) == 72, len(test_pairs)


def test_eval_kill_gate():
    summary = run_eval()
    assert summary["kill_gate_pass"] is True
    assert summary["f1_gap_pp"] >= 10.0
    # pin the measured numbers so regressions are caught
    assert summary["system_test"]["f1"] >= 0.85, summary["system_test"]
