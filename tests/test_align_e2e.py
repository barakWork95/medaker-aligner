"""
End-to-end alignment with the real MMS model (opt-in: MEDAKER_ALIGNER_E2E=1).
Uses the synthetic Genesis 1:1 sample — verse-shaped audio, not speech — so it verifies the
response schema, monotonic non-overlapping timestamps and that every word is placed; the
posterior scores are meaningless on tones. Drop a real recording as samples/genesis-1-1.wav
to see real scores (see README).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SAMPLES, requires_model


def _load_or_make_sample() -> tuple[np.ndarray, int]:
    wav = SAMPLES / "genesis-1-1.wav"
    if not wav.exists():
        from scripts.make_samples import make_genesis_1_1

        make_genesis_1_1(wav)
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


@requires_model
def test_align_genesis_1_1_schema_and_monotonic_timestamps(genesis_1_1_expected):
    from app.aligner import get_aligner

    wave, sr = _load_or_make_sample()
    words, warnings = get_aligner().align(wave, sr, genesis_1_1_expected)
    assert len(words) == len(genesis_1_1_expected)
    duration = len(wave) / sr
    placed = [w for w in words if w.start is not None]
    assert len(placed) >= 6
    last_end = 0.0
    for w in placed:
        assert 0 <= w.start < w.end <= duration + 0.05
        assert w.start >= last_end - 1e-6  # words in order, no overlap
        last_end = w.end
        assert w.syllables and w.phones
        assert w.syllables[0].start >= w.start - 1e-6 and w.syllables[-1].end <= w.end + 1e-6
        for a, b in zip(w.phones, w.phones[1:]):
            assert b.start >= a.end - 1e-6
        assert w.phonetic is not None and 0 <= w.phonetic.score <= 1


@requires_model
def test_http_endpoint(genesis_1_1_expected):
    import base64
    import io

    from fastapi.testclient import TestClient

    from app.main import app

    wave, sr = _load_or_make_sample()
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    body = {
        "contractVersion": 1,
        "tradition": "temani",
        "verse": {"ref": {"book": "Genesis", "chapter": 1, "verse": 1}, "text": "…"},
        "expected": [w.model_dump() for w in genesis_1_1_expected],
        "profile": None,
        "audio": {"mimeType": "audio/wav", "durationMs": int(len(wave) / sr * 1000), "base64": base64.b64encode(buf.getvalue()).decode()},
    }
    with TestClient(app) as client:
        r = client.post("/api/align", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["contractVersion"] == 1
        assert data["engine"] == "mms-fa-torchaudio"
        assert len(data["words"]) == 7
        assert client.get("/health").json()["ok"] is True


@requires_model
def test_calibration_pipeline_on_a_dropped_file(tmp_path, monkeypatch):
    """Drop a file named by verse into a folder → evaluate() infers, aligns and reports; calibrate() tunes."""
    import shutil

    from app.calibration import calibrate
    from scripts.calibrate_fixtures import evaluate

    wave, sr = _load_or_make_sample()
    folder = tmp_path / "real_audio"
    folder.mkdir()
    shutil.copy(SAMPLES / "genesis-1-1.wav", folder / "genesis_1_1_test.wav")
    # expectations: reuse the committed sample fixture as the cache so no frontend/network is needed
    import app.fixtures.registry as reg

    cache_dir = tmp_path / "expected"
    cache_dir.mkdir()
    fixture = json.loads((SAMPLES / "genesis-1-1.expected.json").read_text(encoding="utf-8"))
    (cache_dir / "Genesis.1.1.json").write_text(json.dumps({"ref": "Genesis 1:1", "text": " ".join(w["pointed"] for w in fixture), "expected": fixture}), encoding="utf-8")
    monkeypatch.setattr(reg, "expected_cache_path", lambda ref: cache_dir / (ref.replace(" ", "_").replace(":", ".") + ".json"))

    word_stats, contrast_stats, report = evaluate(folder)
    assert len(report) == 1 and report[0]["ref"] == "Genesis 1:1" and report[0]["placed"] == 7
    assert len(word_stats) == 7 and all(w.spoken for w in word_stats)
    assert any(c.kind == "pair" for c in contrast_stats) and any(c.kind == "marker" for c in contrast_stats)
    result = calibrate(word_stats, contrast_stats)
    assert set(result.thresholds) >= {"MIN_WORD_POSTERIOR", "CONTRAST_MARGIN", "MARKER_MIN"}
    assert any("real speech" in w for w in result.warnings)  # synthetic tones → low-confidence guard fires
