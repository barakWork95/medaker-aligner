"""
Evaluate the aligner on every recording in app/fixtures/real_audio/ and tune its thresholds.

    python scripts/calibrate_fixtures.py             # evaluate + write app/fixtures/calibration.json
    python scripts/calibrate_fixtures.py --dry-run   # evaluate and print, do not write
    python scripts/calibrate_fixtures.py --dir path  # another folder of recordings

Per file: verse from the manifest or the filename, pointed text from the manifest or Sefaria,
Temani expectations from the frontend exporter (cached in app/fixtures/expected/), then the
ungated alignment statistics. The calibration (app/calibration.py) turns the pooled statistics
into thresholds and the report lists every file with placed words, coverage, confidence and
the Temani issues the CURRENT thresholds would raise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.calibration import ContrastStat, WordStat, calibrate, save_calibration  # noqa: E402
from app.fixtures.registry import CALIBRATION_FILE, REAL_AUDIO_DIR, resolve_fixtures  # noqa: E402


def evaluate(directory: Path) -> tuple[list[WordStat], list[ContrastStat], list[dict]]:
    import soundfile as sf

    from app.aligner import get_aligner
    from app.audio import decode_base64_audio
    from app.contract import ExpectedWord

    aligner = get_aligner()
    fixtures = resolve_fixtures(directory)
    if not fixtures:
        print(f"no recordings in {directory} (drop .wav/.m4a/.mp3/.webm files there)")
        return [], [], []
    word_stats: list[WordStat] = []
    contrast_stats: list[ContrastStat] = []
    report: list[dict] = []
    for fx in fixtures:
        t0 = time.time()
        data = fx.path.read_bytes()
        if fx.path.suffix.lower() == ".wav":
            x, sr = sf.read(fx.path, dtype="float32", always_2d=True)
            wave = x.mean(axis=1)
            if sr != 16000:
                import base64

                wave, sr = decode_base64_audio(base64.b64encode(data).decode(), "audio/wav")
        else:
            import base64

            wave, sr = decode_base64_audio(base64.b64encode(data).decode(), "audio/" + fx.path.suffix.lstrip("."))
        expected = [ExpectedWord(**w) for w in fx.expected]
        raw = aligner.raw_stats(wave, sr, expected)
        for w in raw["words"]:
            word_stats.append(WordStat(fx.path.name, w["word"], w["duration"], w["posterior"], spoken=w["word"] not in fx.omitted_words))
        for c in raw["contrasts"]:
            annotated = fx.issues.get(c["word"], [])
            contrast_stats.append(ContrastStat(fx.path.name, c["word"], c["ipa"], c["kind"], c["expected"], c["contrast"], correct=not any(c["ipa"] in a or a.startswith(c["ipa"]) for a in annotated)))
        words, warnings = aligner.align(wave, sr, expected)
        placed = [w for w in words if w.start is not None]
        duration = len(wave) / sr
        covered = sum(w.end - w.start for w in placed)
        entry = {
            "file": fx.path.name,
            "ref": fx.ref,
            "reader": fx.reader,
            "seconds": round(duration, 2),
            "words": len(words),
            "placed": len(placed),
            "coverage": round(covered / duration, 2) if duration else 0,
            "medianPosterior": round(float(sorted(w["posterior"] for w in raw["words"])[len(raw["words"]) // 2]), 3) if raw["words"] else None,
            "issues": {w.index: w.phonetic.issues for w in placed if w.phonetic and w.phonetic.issues},
            "warnings": warnings,
            "alignMs": int((time.time() - t0) * 1000),
        }
        report.append(entry)
        print(f"{fx.path.name:40s} {fx.ref:16s} {entry['placed']:2d}/{entry['words']:<2d} words  cov {entry['coverage']:.2f}  post {entry['medianPosterior']}  {len(entry['issues'])} words w/ issues  {entry['alignMs']} ms")
        for wi, issues in entry["issues"].items():
            print(f"    word {wi} ({expected[wi].display}): " + "; ".join(issues))
        for w in warnings:
            print(f"    ! {w}")
    return word_stats, contrast_stats, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REAL_AUDIO_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="write calibration.json even when the fixtures look like non-speech")
    args = ap.parse_args()
    word_stats, contrast_stats, report = evaluate(args.dir)
    if not report:
        return
    result = calibrate(word_stats, contrast_stats)
    print("\nthresholds:")
    for k, v in result.thresholds.items():
        print(f"  {k:22s} {v}")
    print("stats:", json.dumps(result.stats, ensure_ascii=False))
    for w in result.warnings:
        print("  !", w)
    if args.dry_run:
        print("(dry run — calibration.json not written)")
        return
    low_confidence = any("real speech" in w for w in result.warnings)
    if low_confidence and not args.force:
        print("\nREFUSING to write calibration.json: the fixtures' confidence is too low for real speech (synthetic/noisy audio?). Use --force to override.")
        sys.exit(2)
    save_calibration(result, CALIBRATION_FILE, report)
    print(f"\nwrote {CALIBRATION_FILE.relative_to(ROOT)} — restart the server to apply")


if __name__ == "__main__":
    main()
