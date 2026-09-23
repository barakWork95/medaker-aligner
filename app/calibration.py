"""
Threshold calibration from real recordings.

The aligner has five acoustic thresholds (see app/aligner.py). Given per-word / per-phone
statistics gathered from fixtures whose reading is known to be correct (default) or annotated
(omitted words, expected Temani deviations), this module picks values that:

  MIN_WORD_POSTERIOR  sits below the 5th percentile of spoken-word posteriors (×0.8), so real
                      words are never dropped; clamped to [0.03, 0.3].
  SQUEEZED_S          sits below the 5th percentile of spoken-word durations (×0.8), clamped
                      to [0.06, 0.25].
  OMIT_RELATIVE       stays at 0.4 unless annotated omitted words exist, in which case it is
                      raised just enough to catch them (up to 0.7) without dropping spoken words.
  CONTRAST_MARGIN     ≥ the 95th percentile of (contrast − expected) mean posterior over phones
                      read correctly (+0.02), so ≤ 5 % false Temani issues; clamped [0.05, 0.4].
  MARKER_MIN          ≤ the 5th percentile of the marker letter's peak posterior over correctly
                      read digraphs (×0.8), clamped [0.03, 0.3].

Everything is pure so it is testable without the model; the aligner reads the resulting
calibration.json at import time (app/aligner.py → load_calibration()).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_THRESHOLDS = {
    "MIN_WORD_S": 0.05,
    "SQUEEZED_S": 0.15,
    "MIN_WORD_POSTERIOR": 0.15,
    "OMIT_RELATIVE": 0.4,
    "LOW_CONFIDENCE_MEDIAN": 0.1,
    "CONTRAST_MARGIN": 0.15,
    "MARKER_MIN": 0.12,
}


@dataclass
class WordStat:
    file: str
    word: int
    duration: float
    posterior: float
    spoken: bool  # False when the manifest says the reader omitted it


@dataclass
class ContrastStat:
    file: str
    word: int
    ipa: str
    kind: str  # "pair" | "marker"
    expected_mean: float  # pair: expected letter mean posterior; marker: marker letter PEAK posterior
    contrast_mean: float
    correct: bool  # True when the reader is known to have pronounced it the Temani way


@dataclass
class CalibrationResult:
    thresholds: dict[str, float]
    stats: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _pct(values: list[float], p: float) -> Optional[float]:
    return float(np.percentile(values, p)) if values else None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def calibrate(word_stats: list[WordStat], contrast_stats: list[ContrastStat], base: Optional[dict[str, float]] = None) -> CalibrationResult:
    th = dict(base or DEFAULT_THRESHOLDS)
    stats: dict[str, float] = {}
    warnings: list[str] = []

    spoken = [w for w in word_stats if w.spoken]
    omitted = [w for w in word_stats if not w.spoken]
    if spoken:
        p5_post = _pct([w.posterior for w in spoken], 5)
        p5_dur = _pct([w.duration for w in spoken], 5)
        stats.update(spoken_words=len(spoken), posterior_p5=p5_post, posterior_median=_pct([w.posterior for w in spoken], 50), duration_p5=p5_dur)
        th["MIN_WORD_POSTERIOR"] = round(clamp(p5_post * 0.8, 0.03, 0.3), 3)
        th["SQUEEZED_S"] = round(clamp(p5_dur * 0.8, 0.06, 0.25), 3)
        median = _pct([w.posterior for w in spoken], 50) or 0.0
        if median < th["LOW_CONFIDENCE_MEDIAN"]:
            warnings.append("median spoken-word posterior %.2f is below the low-confidence level — are the fixtures real speech?" % median)
    else:
        warnings.append("no spoken words in the fixtures; word thresholds left unchanged")

    if omitted and spoken:
        # raise OMIT_RELATIVE until every annotated omission is caught, without touching spoken words
        median = _pct([w.posterior for w in spoken], 50) or 1.0
        need = max(w.posterior for w in omitted) / median if median else 0
        min_spoken_rel = min(w.posterior for w in spoken) / median if median else 1
        candidate = clamp(need + 0.05, th["OMIT_RELATIVE"], 0.7)
        if candidate < min_spoken_rel:
            th["OMIT_RELATIVE"] = round(candidate, 3)
        else:
            warnings.append("annotated omitted words overlap spoken-word posteriors; OMIT_RELATIVE not raised")
        caught = sum(1 for w in omitted if w.duration < th["SQUEEZED_S"] and w.posterior < th["MIN_WORD_POSTERIOR"] and w.posterior < th["OMIT_RELATIVE"] * median)
        stats.update(omitted_words=len(omitted), omitted_caught=caught)

    pairs_ok = [c for c in contrast_stats if c.kind == "pair" and c.correct]
    markers_ok = [c for c in contrast_stats if c.kind == "marker" and c.correct]
    if pairs_ok:
        diffs = [c.contrast_mean - c.expected_mean for c in pairs_ok]
        p95 = _pct(diffs, 95)
        th["CONTRAST_MARGIN"] = round(clamp(p95 + 0.02, 0.05, 0.4), 3)
        stats.update(pair_phones=len(pairs_ok), pair_diff_p95=p95)
    if markers_ok:
        p5 = _pct([c.expected_mean for c in markers_ok], 5)
        th["MARKER_MIN"] = round(clamp(p5 * 0.8, 0.03, 0.3), 3)
        stats.update(marker_phones=len(markers_ok), marker_peak_p5=p5)
    wrong = [c for c in contrast_stats if not c.correct]
    if wrong:
        detected = 0
        for c in wrong:
            if c.kind == "pair":
                detected += c.contrast_mean > c.expected_mean + th["CONTRAST_MARGIN"]
            else:
                detected += c.expected_mean < th["MARKER_MIN"] and c.contrast_mean > th["MARKER_MIN"]
        stats.update(annotated_deviations=len(wrong), deviations_detected=detected)
    return CalibrationResult(thresholds=th, stats=stats, warnings=warnings)


def save_calibration(result: CalibrationResult, path: Path, report: Optional[list[dict]] = None) -> None:
    payload = {"thresholds": result.thresholds, "stats": result.stats, "warnings": result.warnings, "files": report or []}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path) -> dict[str, float]:
    """Thresholds from calibration.json merged over the defaults (missing file → defaults)."""
    th = dict(DEFAULT_THRESHOLDS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for k, v in (data.get("thresholds") or {}).items():
            if k in th and isinstance(v, (int, float)):
                th[k] = float(v)
    except (OSError, ValueError):
        pass
    return th


def as_dict(result: CalibrationResult) -> dict:
    return asdict(result)
