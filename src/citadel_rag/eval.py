"""Labeled evaluation for contradiction detection.

~30 synthetic multi-source scenarios (half contradictory), ~100+ labeled
claim-pairs. Labels are derived from construction: claims sharing a fact_id
with the same stance -> entailment; same fact_id with different stance ->
contradiction; different fact_ids -> neutral.

Deterministic and fully auditable — no LLM generation, no hidden labels.
"""

from __future__ import annotations

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .nli import Thresholds, classify, tune_thresholds

# Each scenario: question + sources, each claim = (text, fact_id, stance).
# Same fact_id + same stance  -> entailment (paraphrases)
# Same fact_id + other stance -> contradiction
# Different fact_id           -> neutral
SCENARIOS: list[dict] = [
    # ---------------- CONTRADICTORY (17) ----------------
    {"id": "geo-capital-fr", "question": "What is the capital of France?",
     "sources": {
         "s1": [("Paris is the capital of France.", "cap-fr", "pro")],
         "s2": [("The capital of France is Paris.", "cap-fr", "pro")],
         "s3": [("Lyon is the capital of France.", "cap-fr", "con")]}},
    {"id": "sci-boiling", "question": "At what temperature does water boil?",
     "sources": {
         "s1": [("Water boils at 100 degrees Celsius at sea level.", "boil", "pro")],
         "s2": [("At sea level, water reaches boiling point at 100 C.", "boil", "pro")],
         "s3": [("Water boils at 90 degrees Celsius at sea level.", "boil", "con")]}},
    {"id": "hist-penicillin", "question": "When was penicillin discovered?",
     "sources": {
         "s1": [("Alexander Fleming discovered penicillin in 1928.", "pen", "pro")],
         "s2": [("Penicillin was discovered by Fleming in 1928.", "pen", "pro")],
         "s3": [("Penicillin was discovered in 1935.", "pen", "con")]}},
    {"id": "med-statin", "question": "Do statins reduce heart attack risk?",
     "sources": {
         "s1": [("Statins reduce the risk of heart attack.", "statin", "pro")],
         "s2": [("Statins do not reduce the risk of heart attack.", "statin", "con")],
         "s3": [("Doctors prescribe statins to lower cardiac risk.", "statin-rx", "other")]}},
    {"id": "geo-population", "question": "What is the population of Paris?",
     "sources": {
         "s1": [("Paris has a population of 2.1 million people.", "pop", "pro")],
         "s2": [("About 2.1 million people live in Paris.", "pop", "pro")],
         "s3": [("Paris has a population of 3.4 million people.", "pop", "con")]}},
    {"id": "med-smoking", "question": "Does smoking affect lung cancer risk?",
     "sources": {
         "s1": [("Smoking increases the risk of lung cancer.", "smoke", "pro")],
         "s2": [("Smoking decreases the risk of lung cancer.", "smoke", "con")],
         "s3": [("Lung cancer is linked to tobacco use.", "smoke-link", "other")]}},
    {"id": "geo-eiffel", "question": "How tall is the Eiffel Tower?",
     "sources": {
         "s1": [("The Eiffel Tower is 330 metres tall.", "eiffel", "pro")],
         "s2": [("The Eiffel Tower stands 330 meters high.", "eiffel", "pro")],
         "s3": [("The Eiffel Tower is 210 metres tall.", "eiffel", "con")]}},
    {"id": "med-vaccine", "question": "Are vaccines safe?",
     "sources": {
         "s1": [("Vaccines are safe and effective.", "vax", "pro")],
         "s2": [("Vaccines are unsafe.", "vax", "con")],
         "s3": [("Vaccines are effective at preventing disease.", "vax-eff", "other")]}},
    {"id": "hist-moon", "question": "When did humans first land on the Moon?",
     "sources": {
         "s1": [("Humans first landed on the Moon in 1969.", "moon", "pro")],
         "s2": [("The first Moon landing happened in 1969.", "moon", "pro")],
         "s3": [("Humans first landed on the Moon in 1972.", "moon", "con")]}},
    {"id": "med-exercise", "question": "Is exercise good for heart health?",
     "sources": {
         "s1": [("Regular exercise is beneficial for heart health.", "ex", "pro")],
         "s2": [("Regular exercise is harmful for heart health.", "ex", "con")],
         "s3": [("Cardiologists recommend physical activity.", "ex-rx", "other")]}},
    {"id": "sci-lightspeed", "question": "What is the speed of light?",
     "sources": {
         "s1": [("Light travels at 299792 kilometres per second.", "light", "pro")],
         "s2": [("The speed of light is 299792 km per second.", "light", "pro")],
         "s3": [("Light travels at 150000 kilometres per second.", "light", "con")]}},
    {"id": "med-antibiotics", "question": "Do antibiotics work against viruses?",
     "sources": {
         "s1": [("Antibiotics do not work against viruses.", "abx", "pro")],
         "s2": [("Antibiotics work against viruses.", "abx", "con")],
         "s3": [("Antibiotics treat bacterial infections.", "abx-b", "other")]}},
    {"id": "geo-nile", "question": "What is the longest river in the world?",
     "sources": {
         "s1": [("The Nile is the longest river in the world.", "nile", "pro")],
         "s2": [("The Amazon is the longest river in the world.", "nile", "con")],
         "s3": [("The Nile flows through Egypt.", "nile-eg", "other")]}},
    {"id": "med-insulin", "question": "Does type 1 diabetes require insulin?",
     "sources": {
         "s1": [("Type 1 diabetes requires lifelong insulin therapy.", "t1d", "pro")],
         "s2": [("Type 1 diabetes does not require insulin.", "t1d", "con")],
         "s3": [("Insulin regulates blood sugar.", "ins", "other")]}},
    {"id": "geo-wall", "question": "How long is the Great Wall of China?",
     "sources": {
         "s1": [("The Great Wall of China is 21196 kilometres long.", "wall", "pro")],
         "s2": [("The Great Wall stretches 21196 km in length.", "wall", "pro")],
         "s3": [("The Great Wall of China is 8850 kilometres long.", "wall", "con")]}},
    {"id": "sci-absolute-zero", "question": "What is absolute zero?",
     "sources": {
         "s1": [("Absolute zero is minus 273.15 degrees Celsius.", "az", "pro")],
         "s2": [("Absolute zero equals -273.15 C.", "az", "pro")],
         "s3": [("Absolute zero is minus 150 degrees Celsius.", "az", "con")]}},
    {"id": "med-aspirin-bleed", "question": "Does aspirin affect bleeding risk?",
     "sources": {
         "s1": [("Aspirin increases the risk of bleeding.", "bleed", "pro")],
         "s2": [("Aspirin decreases the risk of bleeding.", "bleed", "con")],
         "s3": [("Bleeding risk rises with blood thinners.", "bleed-t", "other")]}},
    # ---------------- AGREEMENT (17) ----------------
    {"id": "sci-photo", "question": "What is photosynthesis?",
     "sources": {
         "s1": [("Photosynthesis converts sunlight into chemical energy.", "photo", "pro")],
         "s2": [("Plants use photosynthesis to turn light into energy.", "photo", "pro")],
         "s3": [("Chlorophyll absorbs light during photosynthesis.", "chloro", "other")]}},
    {"id": "sci-gravity", "question": "What is gravity?",
     "sources": {
         "s1": [("Gravity is the force that attracts objects toward each other.", "grav", "pro")],
         "s2": [("Gravity pulls objects with mass toward one another.", "grav", "pro")],
         "s3": [("Newton described gravity in 1687.", "newton", "other")]}},
    {"id": "sci-dna", "question": "What is the shape of DNA?",
     "sources": {
         "s1": [("DNA has a double helix structure.", "dna", "pro")],
         "s2": [("The DNA molecule is shaped like a double helix.", "dna", "pro")],
         "s3": [("DNA carries genetic information.", "dna-info", "other")]}},
    {"id": "geo-pacific", "question": "What is the largest ocean?",
     "sources": {
         "s1": [("The Pacific Ocean is the largest ocean on Earth.", "pac", "pro")],
         "s2": [("The Pacific is the biggest ocean in the world.", "pac", "pro")],
         "s3": [("The Pacific covers about one third of Earth.", "pac-size", "other")]}},
    {"id": "med-temp", "question": "What is normal human body temperature?",
     "sources": {
         "s1": [("Normal human body temperature is 37 degrees Celsius.", "temp", "pro")],
         "s2": [("A healthy body temperature is about 37 C.", "temp", "pro")],
         "s3": [("Fever is defined as temperature above 38 C.", "fever", "other")]}},
    {"id": "med-aspirin", "question": "What does aspirin do?",
     "sources": {
         "s1": [("Aspirin relieves pain and reduces inflammation.", "asp", "pro")],
         "s2": [("Aspirin is used for pain relief and as an anti-inflammatory.", "asp", "pro")],
         "s3": [("Aspirin can irritate the stomach lining.", "asp-se", "other")]}},
    {"id": "geo-everest", "question": "What is the highest mountain?",
     "sources": {
         "s1": [("Mount Everest is the highest mountain in the world.", "ev", "pro")],
         "s2": [("Everest is the tallest mountain on Earth.", "ev", "pro")],
         "s3": [("Everest lies in the Himalayas.", "ev-him", "other")]}},
    {"id": "sci-year", "question": "How many days are in a year?",
     "sources": {
         "s1": [("A common year has 365 days.", "yr", "pro")],
         "s2": [("There are 365 days in a regular year.", "yr", "pro")],
         "s3": [("Leap years have 366 days.", "leap", "other")]}},
    {"id": "sci-neuron", "question": "How do neurons communicate?",
     "sources": {
         "s1": [("Neurons communicate through synapses.", "neur", "pro")],
         "s2": [("Synapses are the junctions where neurons signal each other.", "neur", "pro")],
         "s3": [("The brain has about 86 billion neurons.", "brain-n", "other")]}},
    {"id": "geo-tokyo", "question": "What is the population of Tokyo?",
     "sources": {
         "s1": [("Tokyo has about 14 million residents.", "tok", "pro")],
         "s2": [("Around 14 million people live in Tokyo.", "tok", "pro")],
         "s3": [("Tokyo hosted the 2020 Olympics.", "tok-ol", "other")]}},
    {"id": "hist-ww2", "question": "When did World War II end?",
     "sources": {
         "s1": [("World War II ended in 1945.", "ww2", "pro")],
         "s2": [("The Second World War finished in 1945.", "ww2", "pro")],
         "s3": [("World War II began in 1939.", "ww2-start", "other")]}},
    {"id": "sci-water", "question": "How much of Earth is covered by water?",
     "sources": {
         "s1": [("Water covers 71 percent of Earth's surface.", "wat", "pro")],
         "s2": [("About 71% of the planet is covered by water.", "wat", "pro")],
         "s3": [("Oceans hold 97 percent of Earth's water.", "ocean-w", "other")]}},
    {"id": "tech-python", "question": "Who created Python?",
     "sources": {
         "s1": [("Python was created by Guido van Rossum.", "py", "pro")],
         "s2": [("Guido van Rossum invented the Python language.", "py", "pro")],
         "s3": [("Python was first released in 1991.", "py-rel", "other")]}},
    {"id": "med-heart", "question": "What does the heart do?",
     "sources": {
         "s1": [("The heart pumps blood through the body.", "heart", "pro")],
         "s2": [("Blood is circulated by the pumping heart.", "heart", "pro")],
         "s3": [("The heart has four chambers.", "heart-ch", "other")]}},
    {"id": "geo-sahara", "question": "What is the largest hot desert?",
     "sources": {
         "s1": [("The Sahara is the largest hot desert in the world.", "sah", "pro")],
         "s2": [("The Sahara is the biggest hot desert on Earth.", "sah", "pro")],
         "s3": [("The Sahara spans North Africa.", "sah-na", "other")]}},
    {"id": "sci-mitochondria", "question": "What do mitochondria do?",
     "sources": {
         "s1": [("Mitochondria produce energy for the cell.", "mito", "pro")],
         "s2": [("Cells get their energy from mitochondria.", "mito", "pro")],
         "s3": [("Mitochondria have their own DNA.", "mito-dna", "other")]}},
    {"id": "hist-rome", "question": "When was Rome founded?",
     "sources": {
         "s1": [("Rome was founded in 753 BC.", "rome", "pro")],
         "s2": [("The founding of Rome dates to 753 BC.", "rome", "pro")],
         "s3": [("Romulus is the legendary founder of Rome.", "romulus", "other")]}},
]

DEV_IDS = {"geo-capital-fr", "sci-boiling", "hist-penicillin", "med-statin",
           "geo-population", "sci-photo", "sci-gravity", "sci-dna",
           "geo-pacific", "med-temp"}
TEST_IDS = {s["id"] for s in SCENARIOS} - DEV_IDS


def make_pairs(scenarios: list[dict]) -> list[tuple[str, str, str, str]]:
    """All cross-source claim pairs with gold labels.

    Returns (claim_a, claim_b, gold_label, scenario_id).
    """
    pairs: list[tuple[str, str, str, str]] = []
    for sc in scenarios:
        flat: list[tuple[str, str, str, str]] = []  # text, fact, stance, src
        for src, claims in sc["sources"].items():
            for text, fact, stance in claims:
                flat.append((text, fact, stance, src))
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                ta, fa, sa, srca = flat[i]
                tb, fb, sb, srcb = flat[j]
                if srca == srcb:
                    continue
                if fa == fb:
                    gold = "contradiction" if sa != sb else "entailment"
                else:
                    gold = "neutral"
                pairs.append((ta, tb, gold, sc["id"]))
    return pairs


def prf(gold: list[str], pred: list[str], label: str = "contradiction") -> dict:
    tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
    fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n": len(gold)}


def evaluate_system(pairs: list[tuple[str, str, str, str]],
                    th: Thresholds) -> dict:
    gold = [g for _, _, g, _ in pairs]
    pred = [classify(a, b, th).value for a, b, _, _ in pairs]
    return prf(gold, pred)


def baseline_tfidf(dev_pairs: list[tuple[str, str, str, str]],
                    test_pairs: list[tuple[str, str, str, str]]) -> dict:
    """Naive baseline: TF-IDF cosine similarity in a tuned band -> contradiction.

    Tunes (lo, hi) on dev to maximize contradiction F1, reports on test.
    """
    def sims(pairs):
        texts_a = [a for a, _, _, _ in pairs]
        texts_b = [b for _, b, _, _ in pairs]
        vec = TfidfVectorizer().fit(texts_a + texts_b)
        xa = vec.transform(texts_a)
        xb = vec.transform(texts_b)
        return [float(cosine_similarity(xa[i], xb[i])[0, 0])
                for i in range(len(pairs))]

    dev_s, test_s = sims(dev_pairs), sims(test_pairs)
    dev_gold = [g for _, _, g, _ in dev_pairs]
    best = (0.0, 0.0, -1.0)
    for lo in [round(x * 0.05, 2) for x in range(0, 20)]:
        for hi in [round(x * 0.05, 2) for x in range(1, 21)]:
            if hi <= lo:
                continue
            pred = ["contradiction" if lo <= s <= hi else "neutral"
                    for s in dev_s]
            f1 = prf(dev_gold, pred)["f1"]
            if f1 > best[2]:
                best = (lo, hi, f1)
    lo, hi, dev_f1 = best
    test_gold = [g for _, _, g, _ in test_pairs]
    test_pred = ["contradiction" if lo <= s <= hi else "neutral"
                 for s in test_s]
    test = prf(test_gold, test_pred)
    return {"lo": lo, "hi": hi, "dev_f1": dev_f1, "test": test}


def run_eval(out_dir: str | None = None) -> dict:
    dev = [s for s in SCENARIOS if s["id"] in DEV_IDS]
    test = [s for s in SCENARIOS if s["id"] in TEST_IDS]
    dev_pairs = make_pairs(dev)
    test_pairs = make_pairs(test)

    th = tune_thresholds(dev_pairs)
    sys_dev = evaluate_system(dev_pairs, th)
    sys_test = evaluate_system(test_pairs, th)
    base = baseline_tfidf(dev_pairs, test_pairs)

    label_counts: dict[str, int] = {}
    for _, _, g, _ in test_pairs:
        label_counts[g] = label_counts.get(g, 0) + 1

    summary = {
        "n_scenarios": len(SCENARIOS),
        "n_dev_pairs": len(dev_pairs),
        "n_test_pairs": len(test_pairs),
        "test_label_counts": label_counts,
        "tuned_thresholds": {"tau_contra": th.tau_contra,
                             "tau_entail": th.tau_entail},
        "system_dev_f1": round(sys_dev["f1"], 4),
        "system_test": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in sys_test.items()},
        "baseline_tfidf": {
            "lo": base["lo"], "hi": base["hi"],
            "dev_f1": round(base["dev_f1"], 4),
            "test": {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in base["test"].items()},
        },
    }
    f1_sys = sys_test["f1"]
    f1_base = base["test"]["f1"]
    summary["f1_gap_pp"] = round((f1_sys - f1_base) * 100, 2)
    summary["kill_gate_pass"] = summary["f1_gap_pp"] >= 10.0

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else None
    s = run_eval(out)
    print(json.dumps(s, indent=1))
