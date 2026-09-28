"""Step 5: combine the signals into one 0-100 risk score with a plain-English reason.

Per chunk:
  1. Weighted average of the fast signals (heuristics, similarity, classifier).
     Signals that are unavailable are skipped and the other weights re-balanced.
     If train/train_fusion.py has produced runtime/fusion_weights.json AND the same
     signals are available now, the learned logistic-regression model is used instead.
  2. (Removed) There is no strong-signal floor: the score is purely the weighted/learned combination.
  3. If the score is in the unclear band, the LLM judge is asked and can move it partway.

Per document:
  risk = highest chunk score (an email is as dangerous as its worst sentence),
  raised to a minimum if the pre-processor found hiding tricks.
  Media/QR threats get no special case: the rules signal carries them (see preprocess.qr_rules_score)
  and the fused score is whatever the learned weights (or the default weights) make of it.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Callable, Optional

from .config import RUNTIME_DIR, SETTINGS
from .types import Chunk, ChunkResult, Finding, Level, SignalScore

FAST_SIGNALS = ("heuristics", "similarity", "classifier")
WEIGHTS_FILE = RUNTIME_DIR / "fusion_weights.json"
# Learned weights from fewer examples than this are ignored: with tiny data they are noise.
MIN_TRAIN_FOR_LEARNED = 100

# Minimum document risk when a hiding trick is found, even if no sentence looks like an attack
FINDING_FLOORS = {
    "unicode_tags": 45, "exfil_image": 45, "base64": 30,
    "hidden_css": 25, "html_comment": 20, "homoglyph": 20, "zero_width": 15, "bidi_control": 15,
}


@lru_cache(maxsize=1)
def learned_model() -> Optional[dict]:
    if WEIGHTS_FILE.exists():
        try:
            return json.loads(WEIGHTS_FILE.read_text())
        except json.JSONDecodeError:
            return None
    return None


def reload_learned_model() -> None:
    learned_model.cache_clear()


def fast_score(signals: list[SignalScore], use_learned: bool = True) -> float:
    by_name = {s.name: s for s in signals if s.available and s.name in FAST_SIGNALS}
    if not by_name:
        return 0.0

    model = learned_model() if use_learned else None
    if model and model.get("n_train", 0) >= MIN_TRAIN_FOR_LEARNED and set(model["signals"]) == set(by_name):
        z = model["intercept"] + sum(model["coef"][n] * by_name[n].score for n in model["signals"])
        p = 1.0 / (1.0 + math.exp(-z))
        p0 = 1.0 / (1.0 + math.exp(-model["intercept"]))       # probability with zero evidence
        score = max(0.0, (p - p0) / (1.0 - p0))                  # evidence above that baseline
    else:
        total_w = sum(SETTINGS.weights[n] for n in by_name)
        score = sum(SETTINGS.weights[n] * by_name[n].score for n in by_name) / total_w

    # No floors, caps or buckets: the score is exactly the weighted / learned combination above.
    # min/max only guard the 0-1 range against floating-point or extrapolation noise.
    return min(1.0, max(0.0, score))


def fast_score_from_values(values: dict[str, float], use_learned: bool = False) -> float:
    """Same maths as fast_score() but from plain numbers (used by training/eval scripts)."""
    return fast_score([SignalScore(name=n, score=v) for n, v in values.items()], use_learned=use_learned)


def combine_chunk(chunk: Chunk, signals: list[SignalScore],
                  judge_fn: Optional[Callable[[Chunk], SignalScore]] = None) -> ChunkResult:
    score = fast_score(signals)
    judged = False
    lo, hi = SETTINGS.judge_band
    if judge_fn is not None and lo <= score <= hi:
        verdict = judge_fn(chunk)
        signals = signals + [verdict]
        if verdict.available:
            score = (1 - SETTINGS.judge_weight) * score + SETTINGS.judge_weight * verdict.score
            judged = True
    return ChunkResult(chunk=chunk, signals=signals, combined=round(score, 4), judged=judged)


def document_risk(chunk_results: list[ChunkResult], findings: list[Finding]) -> int:
    risk = max((c.combined for c in chunk_results), default=0.0) * 100
    for f in findings:
        risk = max(risk, FINDING_FLOORS.get(f.kind, 0))
    return int(round(risk))


def level_for(risk: int) -> Level:
    if risk >= SETTINGS.quarantine_at:
        return "quarantine"
    if risk >= SETTINGS.warn_at:
        return "warn"
    return "allow"


def explain(level: Level, risk: int, top: Optional[ChunkResult], findings: list[Finding]) -> str:
    if level == "allow" and not findings:
        return f"No signs of hidden instructions (risk {risk}/100)."
    parts = []
    verb = {"quarantine": "Quarantined", "warn": "Warning", "allow": "Allowed"}[level]
    parts.append(f"{verb} (risk {risk}/100).")
    if top is not None and top.combined * 100 >= SETTINGS.warn_at:
        where = "hidden text" if top.chunk.hidden else "the message"
        snippet = top.chunk.text if len(top.chunk.text) <= 140 else top.chunk.text[:137] + "..."
        parts.append(f'Suspicious sentence in {where}: "{snippet}"')
        reasons = []
        for s in top.signals:
            if not s.available:
                continue
            for r in s.reasons:
                if r not in reasons:
                    reasons.append(r)
        if reasons:
            parts.append("Why: " + "; ".join(reasons[:4]) + ".")
    tricks = sorted({f.detail for f in findings if f.kind in FINDING_FLOORS or f.kind in ("attribute_text", "media_alert")})
    if tricks:
        parts.append("Hiding tricks found: " + "; ".join(tricks[:3]) + ".")
    return " ".join(parts)
