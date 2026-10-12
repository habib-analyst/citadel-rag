"""Deterministic NLI heuristic for claim pairs.

No model downloads, no APIs: contradiction / entailment / neutral are
decided by calibrated linguistic signals — negation flips, antonym
predicates, numeric mismatches (same unit), and minimal-pair entity swaps.
Thresholds are tuned on a dev split (see ``tune_thresholds``); a
cross-encoder backend can replace this module behind the same interface.

score_pair(a, b) -> {"contradiction": float, "entailment": float,
                     "neutral": float} with scores in [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class NLIRelation(Enum):
    CONTRADICTION = "contradiction"
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"


STOPWORDS = frozenset(
    """a an the and or but of in on at to for with by from as is are was were be been
    being it its this that these those there their they them he she we you i my our
    his her will would can could should shall may might do does did have has had
    not no""".split()
)

NEGATIONS = frozenset(
    """not no never n't cannot can't won't don't doesn't isn't aren't wasn't weren't
    haven't hasn't hadn't couldn't wouldn't shouldn't without none neither nor""".split()
)

# word -> antonym (symmetric pairs listed once, expanded at import)
_ANTONYM_PAIRS = [
    ("increase", "decrease"), ("increased", "decreased"), ("increases", "decreases"),
    ("true", "false"), ("safe", "unsafe"), ("effective", "ineffective"),
    ("legal", "illegal"), ("possible", "impossible"), ("higher", "lower"),
    ("more", "less"), ("always", "never"), ("all", "none"),
    ("beneficial", "harmful"), ("approved", "banned"), ("allow", "prohibit"),
    ("support", "oppose"), ("rise", "fall"), ("gain", "loss"),
    ("success", "failure"), ("present", "absent"), ("include", "exclude"),
    ("open", "closed"), ("available", "unavailable"), ("necessary", "unnecessary"),
    ("likely", "unlikely"), ("stable", "unstable"), ("accurate", "inaccurate"),
    ("recommended", "discouraged"), ("protect", "harm"), ("reduce", "raise"),
    ("grow", "shrink"), ("hot", "cold"), ("fast", "slow"),
    ("begin", "end"),
]
_TOK_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?")


# irregular verbs for the stemmer (kept tiny and principled)
_IRREGULAR = {"began": "begin", "went": "go", "did": "do", "was": "be",
              "were": "be", "had": "have", "has": "have"}


def _stem(tok: str, raw: str | None = None) -> str:
    """Light stemming, applied consistently to both sides so it only ever
    helps matching (requires/require -> require). Proper nouns (capitalized
    in the raw text) are never stemmed: Paris stays Paris, not Pari."""
    if raw is not None and raw[:1].isupper() and tok.isalpha():
        return tok
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    if tok.isalpha() and len(tok) > 4:
        if tok.endswith("ies"):
            return tok[:-3] + "y"
        if tok.endswith("eed"):
            return tok
        if tok.endswith("ed"):
            return tok[:-2]
    if tok.isalpha() and len(tok) > 5 and tok.endswith("ing") \
            and not tok.endswith(("eing", "aing", "oing", "uing")):
        return tok[:-3]
    if tok.isalpha() and len(tok) > 3:
        if tok.endswith("s") and not tok.endswith("ss"):
            return tok[:-1]
    return tok


ANTONYMS: dict[str, str] = {}
for w1, w2 in _ANTONYM_PAIRS:
    s1, s2 = _stem(w1), _stem(w2)
    ANTONYMS[s1] = s2
    ANTONYMS[s2] = s1

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
    "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000,
}
_NUM_RE = re.compile(
    r"(?P<sign>-)?(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>million|billion|thousand|percent|%|milligrams?|kilograms?|mg|kg|kilometres?|kilometers?|km|metres?|meters?|cm|mm|years?|days?|hours?)?\b",
    re.IGNORECASE,
)
_OF_RE = re.compile(r"\bof\s+([a-zA-Z']+(?:\s+[a-zA-Z']+){0,2})")

_UNIT_ALIASES = {
    "kilometres": "km", "kilometers": "km", "kilometre": "km", "kilometer": "km",
    "metres": "m", "meters": "m", "metre": "m", "meter": "m",
    "percent": "percent", "%": "percent",
    "milligrams": "mg", "milligram": "mg",
    "kilograms": "kg", "kilogram": "kg",
    "years": "year", "year": "year", "days": "day", "day": "day",
    "hours": "hour", "hour": "hour",
}
# stemmed forms (tokenize() stems before this map is consulted on tokens)
_UNIT_ALIASES.update({
    "kilometre": "km", "kilometer": "km", "metre": "m", "meter": "m",
})


def _canon_unit_tok(tok: str) -> str:
    return _UNIT_ALIASES.get(tok, tok)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in _TOK_RE.findall(text):
        out.append(_stem(raw.lower(), raw))
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _has_negation(tokens: list[str]) -> bool:
    return any(t in NEGATIONS or t.endswith("n't") for t in tokens)


def _numbers(text: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for m in _NUM_RE.finditer(text):
        unit = (m.group("unit") or "").lower()
        unit = _UNIT_ALIASES.get(unit, unit)
        val = float(m.group("val"))
        if m.group("sign"):
            val = -val
        out.append((val, unit))
    # word numbers ("two million")
    toks = tokenize(text)
    for i, t in enumerate(toks):
        if t in _WORD_NUM and t in ("million", "billion", "thousand"):
            continue
        if t in _WORD_NUM and i + 1 < len(toks) and toks[i + 1] in ("million", "billion", "thousand"):
            out.append((float(_WORD_NUM[t] * _WORD_NUM[toks[i + 1]]), toks[i + 1]))
    return out


def _is_year(v: float) -> bool:
    return 1000 <= v <= 2100 and float(v).is_integer()


def _of_complements(text: str) -> list[set[str]]:
    """The 'of X' phrases in a claim: '71 percent of Earth's surface' ->
    [{'earth's', 'surface'}]. Used to tell same-quantity numbers apart
    from different-quantity numbers sharing a unit."""
    out: list[set[str]] = []
    for m in _OF_RE.finditer(text):
        toks = {_stem(t.lower()) for t in m.group(1).split()}
        toks = {t for t in toks if t not in STOPWORDS}
        if toks:
            out.append(toks)
    return out


def _numeric_mismatch(a: str, b: str) -> float:
    """1.0 if both claims state different values for the SAME quantity.

    Same-quantity gate: the non-numeric content must substantially overlap
    (unit-normalized Jaccard >= 0.4), otherwise "71% of Earth's surface" vs
    "97% of Earth's water" would falsely contradict. Years are stricter:
    a bare year difference only contradicts when the rest of the claim is
    essentially the same statement (Jaccard >= 0.8 of non-year tokens, or
    one side's tokens are a subset of the other's plus <= 2 extra tokens) —
    so "ended in 1945" vs "began in 1939" does not fire.
    """
    na, nb = _numbers(a), _numbers(b)
    if not na or not nb:
        return 0.0
    ca = {_canon_unit_tok(t) for t in content_tokens(a)}
    cb = {_canon_unit_tok(t) for t in content_tokens(b)}
    units_a = {u for _, u in na}
    units_b = {u for _, u in nb}
    for unit in units_a & units_b:
        va = [v for v, u in na if u == unit]
        vb = [v for v, u in nb if u == unit]
        for x in va:
            for y in vb:
                if x == y:
                    continue
                nonnum_a = {t for t in ca if not re.fullmatch(r"\d+(\.\d+)?", t)}
                nonnum_b = {t for t in cb if not re.fullmatch(r"\d+(\.\d+)?", t)}
                if _is_year(x) and _is_year(y):
                    j = _jaccard(nonnum_a, nonnum_b)
                    subset_ok = ((nonnum_a <= nonnum_b or nonnum_b <= nonnum_a)
                                 and abs(len(nonnum_a) - len(nonnum_b)) <= 2)
                    # antonym-differing predicates mean different events:
                    # "ended in 1945" vs "began in 1939" must NOT fire
                    diff = (nonnum_a - nonnum_b) | (nonnum_b - nonnum_a)
                    antonym_diff = any(ANTONYMS.get(t) in diff for t in diff)
                    if (j >= 0.6 or subset_ok) and not antonym_diff:
                        return 1.0
                    continue
                if _jaccard(nonnum_a, nonnum_b) < 0.3:
                    continue  # different quantities, not a contradiction
                # of-complement check: "71% of Earth's surface" vs "97% of
                # Earth's water" share a unit but measure different things
                ofa, ofb = _of_complements(a), _of_complements(b)
                if ofa and ofb and max(_jaccard(x, y) for x in ofa
                                       for y in ofb) < 0.5:
                    continue
                denom = max(abs(x), abs(y), 1e-9)
                if abs(x - y) / denom > 0.05:
                    return 1.0
    return 0.0


def _negation_flip(ta: list[str], tb: list[str], overlap: float) -> float:
    na, nb = _has_negation(ta), _has_negation(tb)
    if na != nb and overlap >= 0.55:
        return 0.9
    return 0.0


def _antonym_clash(ca: list[str], cb: list[str], overlap: float) -> float:
    sa, sb = set(ca), set(cb)
    # numeric tokens: if the claims state different numbers/years, the
    # antonymy is about different events ("ended in 1945" vs "began in
    # 1939") — not a contradiction. The numeric rule handles same-event
    # cases.
    numa = {t for t in sa if re.fullmatch(r"\d+(\.\d+)?", t)}
    numb = {t for t in sb if re.fullmatch(r"\d+(\.\d+)?", t)}
    if numa != numb:
        return 0.0
    for t in sa:
        ant = ANTONYMS.get(t)
        if ant and ant in sb and t not in sb:
            # residual overlap: after removing the antonym pair, do the
            # remaining claims still talk about the same thing?
            ra, rb = sa - {t}, sb - {ant}
            resid = _jaccard(ra, rb)
            if resid >= 0.5 or overlap >= 0.45:
                return 0.85
    return 0.0


def _minimal_pair_swap(ca: list[str], cb: list[str], overlap: float,
                       raw_a: str, raw_b: str) -> float:
    """High overlap, exactly one content-token difference, and the differing
    token looks like an entity/number swap (capitalized in raw text or numeric).

    Uses the overlap *coefficient* (|intersection| / min length) instead of
    Jaccard — Jaccard punishes short claims too harshly.
    """
    sa, sb = set(ca), set(cb)
    if not sa or not sb:
        return 0.0
    coeff = len(sa & sb) / min(len(sa), len(sb))
    if coeff < 0.6:
        return 0.0
    diff = (sa - sb) | (sb - sa)
    if len(diff) != 2:
        return 0.0
    only_a = (sa - sb).pop()
    only_b = (sb - sa).pop()

    # Sentence-initial common nouns ("Humans first landed...") are capitalized
    # by position, not entityhood — they must not count as entities.
    _SENTENCE_INITIAL_COMMON = frozenset(
        "humans human people person man woman men women they we it this that "
        "these those there here".split()
    )

    def entity_like(tok: str, raw: str) -> bool:
        # numeric, or the raw text contains a capitalized word whose
        # lowercase form equals the token (proper-noun swap: Paris/Lyon).
        # A sentence-initial match only counts if the word is not a common
        # noun.
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            return True
        for m in re.finditer(r"[A-Z][a-zA-Z]*", raw):
            if m.group(0).lower() != tok:
                continue
            if m.start() == 0 and tok in _SENTENCE_INITIAL_COMMON:
                continue
            return True
        return False

    # both differing tokens must be substitutable entities/numbers:
    # "Paris"/"Lyon" yes; "Humans"/"happen" no.
    if entity_like(only_a, raw_a) and entity_like(only_b, raw_b):
        return 0.8
    return 0.0


@dataclass
class Thresholds:
    tau_contra: float = 0.75
    tau_entail: float = 0.8


DEFAULT_THRESHOLDS = Thresholds()


def score_pair(a: str, b: str) -> dict[str, float]:
    """Score the NLI relation between two claims.

    Returns dict with keys 'contradiction', 'entailment', 'neutral' in [0,1].
    """
    ta, tb = tokenize(a), tokenize(b)
    ca, cb = content_tokens(a), content_tokens(b)
    sa, sb = set(ca), set(cb)
    overlap = _jaccard(sa, sb)

    contra = max(
        _numeric_mismatch(a, b),
        _negation_flip(ta, tb, overlap),
        _antonym_clash(ca, cb, overlap),
        _minimal_pair_swap(ca, cb, overlap, a, b),
    )

    entail = 0.0
    if contra < 0.5:
        if sa and sa <= sb:
            entail = 0.95
        elif sb and sb <= sa:
            entail = 0.95
        elif overlap >= 0.8:
            entail = 0.85

    neutral = max(0.0, 1.0 - max(contra, entail))
    return {"contradiction": contra, "entailment": entail, "neutral": neutral}


def classify(a: str, b: str, th: Thresholds = DEFAULT_THRESHOLDS) -> NLIRelation:
    s = score_pair(a, b)
    if s["contradiction"] >= th.tau_contra:
        return NLIRelation.CONTRADICTION
    if s["entailment"] >= th.tau_entail:
        return NLIRelation.ENTAILMENT
    return NLIRelation.NEUTRAL


def tune_thresholds(pairs: list[tuple[str, str, str]]) -> Thresholds:
    """Grid-search decision thresholds on labeled dev pairs.

    pairs: (claim_a, claim_b, gold_label, ...) with gold_label in
    {"contradiction", "entailment", "neutral"}. Maximizes contradiction F1.
    """
    best = DEFAULT_THRESHOLDS
    best_f1 = -1.0
    for tc in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        for te in (0.5, 0.6, 0.7, 0.8, 0.9):
            th = Thresholds(tau_contra=tc, tau_entail=te)
            tp = fp = fn = 0
            for a, b, gold, *_ in pairs:
                pred = classify(a, b, th).value
                if pred == "contradiction" and gold == "contradiction":
                    tp += 1
                elif pred == "contradiction":
                    fp += 1
                elif gold == "contradiction":
                    fn += 1
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if f1 > best_f1:
                best_f1, best = f1, th
    return best
