"""
Synthesise verse-shaped sample audio (a harmonic burst per syllable, gaps between words) for
schema/timestamp tests — a port of medaker's src/lib/speech/synth.ts. NOT speech: the model's
posteriors on it are meaningless. For real scores record yourself (README).

    python scripts/make_samples.py            → samples/genesis-1-1.wav
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "samples"


def synthesize(expected: list[dict], sr: int = 16000, unit_ms: int = 150, gap_ms: int = 35, word_gap_ms: int = 140, f0: float = 130.0) -> np.ndarray:
    chunks: list[np.ndarray] = []

    def push(seconds: float, voiced: bool, hz: float = f0) -> None:
        n = int(round(seconds * sr))
        c = np.zeros(n, dtype=np.float32)
        if voiced and n:
            t = np.arange(n) / sr
            env = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
            c = (0.5 * env * (np.sin(2 * np.pi * hz * t) + 0.5 * np.sin(4 * np.pi * hz * t) + 0.25 * np.sin(6 * np.pi * hz * t)) / 1.75).astype(np.float32)
        chunks.append(c)

    push(0.15, False)
    for wi, w in enumerate(expected):
        sylls = w["syllables"]
        for si, s in enumerate(sylls):
            push(s["weight"] * unit_ms / 1000, True, f0 * (1.12 if s["stressed"] else 1.0))
            if si < len(sylls) - 1:
                push(gap_ms / 1000, False)
        if wi < len(expected) - 1:
            push((word_gap_ms * (2 if w.get("disjunctive") else 1)) / 1000, False)
    push(0.15, False)
    return np.concatenate(chunks)


def make_genesis_1_1(out: Path = SAMPLES / "genesis-1-1.wav") -> Path:
    expected = json.loads((SAMPLES / "genesis-1-1.expected.json").read_text(encoding="utf-8"))
    sf.write(out, synthesize(expected), 16000)
    return out


if __name__ == "__main__":
    print(make_genesis_1_1())
