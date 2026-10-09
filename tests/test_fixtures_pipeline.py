import json
from pathlib import Path

from app.calibration import DEFAULT_THRESHOLDS, ContrastStat, WordStat, calibrate, load_calibration, save_calibration
from app.fixtures.registry import expected_cache_path, load_manifest, normalize_hebrew, ref_from_filename, resolve_expected, resolve_fixtures


def test_ref_from_filename():
    assert ref_from_filename("genesis_1_1_reader1.wav") == "Genesis 1:1"
    assert ref_from_filename("GENESIS_19_16.m4a") == "Genesis 19:16"
    assert ref_from_filename("i_samuel_3_2_dan.webm") == "I Samuel 3:2"
    assert ref_from_filename("1_kings_18_39.wav") == "I Kings 18:39"
    assert ref_from_filename("song_of_songs_2_4.wav") == "Song of Songs 2:4"
    assert ref_from_filename("psalms_119_176_x_y.mp3") == "Psalms 119:176"
    assert ref_from_filename("recording.wav") is None
    assert ref_from_filename("narnia_1_1.wav") is None


def test_manifest_shape():
    m = load_manifest()
    assert "פרשת-בראשית-שני.mp3" in m  # keys are NFC-normalised
    assert m["פרשת-בראשית-שני.mp3"]["ref"] == "Genesis 2:4-14"
    assert m["פרשת-בשלח-רביעי.mp3"]["ref"] == "Exodus 14:26-15:26"


def test_expand_ref_single_and_same_chapter_ranges():
    from app.fixtures.registry import expand_ref

    assert expand_ref("Genesis 1:1") == ["Genesis 1:1"]
    assert expand_ref("Genesis 2:4-6") == ["Genesis 2:4", "Genesis 2:5", "Genesis 2:6"]
    assert expand_ref("I Samuel 3:2-3") == ["I Samuel 3:2", "I Samuel 3:3"]


def test_expand_ref_cross_chapter_uses_chapter_lengths(monkeypatch):
    import app.fixtures.registry as reg

    monkeypatch.setattr(reg, "chapter_lengths", lambda book: [31, 25, 24])  # fake Genesis shape
    refs = reg.expand_ref("Genesis 1:30-2:2")
    assert refs == ["Genesis 1:30", "Genesis 1:31", "Genesis 2:1", "Genesis 2:2"]


def test_resolve_fixtures_from_a_temp_dir_with_cached_expectations(tmp_path, genesis_1_1_expected, monkeypatch):
    # cache the expectations so no network / frontend is needed
    import app.fixtures.registry as reg

    monkeypatch.setattr(reg, "EXPECTED_DIR", tmp_path / "expected")
    (tmp_path / "expected").mkdir()
    cache = expected_cache_path("Genesis 1:1")
    text = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃"
    (tmp_path / "expected" / cache.name).write_text(json.dumps({"ref": "Genesis 1:1", "text": normalize_hebrew(text), "expected": [w.model_dump() for w in genesis_1_1_expected]}), encoding="utf-8")
    monkeypatch.setattr(reg, "expected_cache_path", lambda ref: tmp_path / "expected" / cache.name)

    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "genesis_1_1_reader1.wav").write_bytes(b"RIFF")
    (audio / "genesis_1_1_reader2.m4a").write_bytes(b"x")
    (audio / "notes.txt").write_text("ignored")
    (audio / "manifest.json").write_text(json.dumps({"files": {"genesis_1_1_reader2.m4a": {"reader": "r2", "omittedWords": [3], "issues": {"5": ["w→v"]}}, "genesis_1_1_reader3.wav": {"skip": True}}}), encoding="utf-8")
    (audio / "genesis_1_1_reader3.wav").write_bytes(b"RIFF")

    fixtures = resolve_fixtures(audio, allow_network=False, allow_frontend=False)
    assert [f.path.name for f in fixtures] == ["genesis_1_1_reader1.wav", "genesis_1_1_reader2.m4a"]
    assert fixtures[0].ref == "Genesis 1:1" and len(fixtures[0].expected) == 7
    assert fixtures[1].reader == "r2" and fixtures[1].omitted_words == [3] and fixtures[1].issues == {5: ["w→v"]}
    assert fixtures[0].verse_key == "Genesis.1.1"
    # a cached verse resolves without network or frontend
    t, e = resolve_expected("Genesis 1:1", None, allow_network=False, allow_frontend=False)
    assert len(e) == 7 and t.startswith("בְּרֵאשִׁ")


def test_normalize_hebrew_matches_the_frontend_rules():
    assert normalize_hebrew("<big>בְּ</big>רֵאשִׁ֖ית&thinsp;<b>׀</b> עַל־ פְּנֵ֣י͏") == "בְּרֵאשִׁ֖ית ׀ עַל־פְּנֵ֣י".encode().decode()  # NFD equal


def test_calibrate_sets_word_thresholds_below_the_spoken_distribution():
    words = [WordStat("a.wav", i, duration=0.3 + 0.05 * (i % 4), posterior=0.5 + 0.02 * (i % 10), spoken=True) for i in range(40)]
    words += [WordStat("a.wav", 99, duration=0.04, posterior=0.05, spoken=False)]
    r = calibrate(words, [])
    th = r.thresholds
    assert th["MIN_WORD_POSTERIOR"] < min(w.posterior for w in words if w.spoken)
    assert 0.03 <= th["MIN_WORD_POSTERIOR"] <= 0.3
    assert th["SQUEEZED_S"] < min(w.duration for w in words if w.spoken)
    assert r.stats["omitted_caught"] == 1
    assert r.stats["spoken_words"] == 40
    assert not r.warnings


def test_calibrate_contrast_thresholds_bound_false_positives():
    pairs = [ContrastStat("a.wav", i, "w", "pair", expected_mean=0.6, contrast_mean=0.6 - 0.3 + 0.01 * (i % 20), correct=True) for i in range(40)]
    markers = [ContrastStat("a.wav", i, "θ", "marker", expected_mean=0.4 + 0.01 * (i % 10), contrast_mean=0.5, correct=True) for i in range(20)]
    wrong = [ContrastStat("b.wav", 1, "w", "pair", expected_mean=0.1, contrast_mean=0.7, correct=False), ContrastStat("b.wav", 2, "θ", "marker", expected_mean=0.02, contrast_mean=0.6, correct=False)]
    r = calibrate([WordStat("a.wav", 0, 0.3, 0.5, True)], pairs + markers + wrong)
    th = r.thresholds
    diffs = [c.contrast_mean - c.expected_mean for c in pairs]
    false_positives = sum(1 for d in diffs if d > th["CONTRAST_MARGIN"])
    assert false_positives <= len(diffs) * 0.05
    assert th["MARKER_MIN"] < min(c.expected_mean for c in markers)
    assert r.stats["deviations_detected"] == 2


def test_calibration_round_trip_and_defaults(tmp_path):
    path = tmp_path / "calibration.json"
    assert load_calibration(path) == DEFAULT_THRESHOLDS  # missing file → defaults
    r = calibrate([WordStat("a.wav", i, 0.3, 0.5, True) for i in range(10)], [])
    save_calibration(r, path, report=[{"file": "a.wav"}])
    loaded = load_calibration(path)
    assert loaded["MIN_WORD_POSTERIOR"] == r.thresholds["MIN_WORD_POSTERIOR"]
    assert set(loaded) == set(DEFAULT_THRESHOLDS)
    path.write_text("{not json")
    assert load_calibration(path) == DEFAULT_THRESHOLDS
    path.write_text(json.dumps({"thresholds": {"MIN_WORD_POSTERIOR": "bad", "UNKNOWN": 1}}))
    assert load_calibration(path) == DEFAULT_THRESHOLDS
