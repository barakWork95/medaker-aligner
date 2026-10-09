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


COLLAPSE_POSTERIOR = 0.08
COLLAPSE_SECONDS_PER_WORD = 0.25


def greedy_peek(aligner, wave, sr: int, seconds: float = 10.0) -> tuple[str, str]:
    """What the model hears (uroman letters) in the first and last `seconds` — to label a file."""
    import torch

    inv = {v: k for k, v in aligner._dictionary.items()}

    def decode(x):
        em = aligner.emission(x, sr)
        ids = em.argmax(dim=-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                out.append(inv.get(i, "?"))
            prev = i
        return "".join(out)

    n = int(seconds * sr)
    return decode(wave[:n]), decode(wave[-n:])


def evaluate(directory: Path, peek: bool = False) -> tuple[list[WordStat], list[ContrastStat], list[dict]]:
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

            mime = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "webm": "audio/webm", "ogg": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac", "mp4": "audio/mp4"}.get(fx.path.suffix.lstrip(".").lower(), "application/octet-stream")
            wave, sr = decode_base64_audio(base64.b64encode(data).decode(), mime)
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
        posts = {w["word"]: w["posterior"] for w in raw["words"]}
        by_index = {w.index: w for w in words}
        per_verse = []
        collapsed: list[str] = []
        for vu in fx.verses:
            idxs = [vu.word_offset + w["index"] for w in vu.expected]
            vwords = [by_index[i] for i in idxs if i in by_index]
            vplaced = [w for w in vwords if w.start is not None]
            vposts = sorted(posts[i] for i in idxs if i in posts)
            vstart = min(w.start for w in vplaced) if vplaced else None
            vend = max(w.end for w in vplaced) if vplaced else None
            median_post = vposts[len(vposts) // 2] if vposts else None
            spw = (vend - vstart) / len(idxs) if vplaced and idxs else None
            # a verse squeezed into a sliver at near-zero confidence was not in the audio
            is_collapsed = bool(median_post is not None and median_post < COLLAPSE_POSTERIOR and spw is not None and spw < COLLAPSE_SECONDS_PER_WORD)
            if is_collapsed:
                collapsed.append(vu.ref)
            per_verse.append({
                "ref": vu.ref,
                "words": len(idxs),
                "placed": len(vplaced),
                "start": round(vstart, 2) if vstart is not None else None,
                "end": round(vend, 2) if vend is not None else None,
                "medianPosterior": round(median_post, 3) if median_post is not None else None,
                "secondsPerWord": round(spw, 2) if spw is not None else None,
                "collapsed": is_collapsed,
                "issues": sum(1 for w in vplaced if w.phonetic and w.phonetic.issues),
            })
        if collapsed:
            warnings = list(warnings) + [f"{len(collapsed)} verse(s) collapsed to near-zero confidence ({collapsed[0]} … {collapsed[-1]}): the recording probably does not contain them — check the manifest range"]
        if peek:
            head, tail = greedy_peek(aligner, wave, sr)
            entry_peek = {"first10s": head, "last10s": tail}
        else:
            entry_peek = None
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
            "verses": per_verse,
            "peek": entry_peek,
            "warnings": warnings,
            "alignMs": int((time.time() - t0) * 1000),
        }
        # words of collapsed verses must not tune the thresholds
        collapsed_words = {vu.word_offset + w["index"] for vu in fx.verses if vu.ref in collapsed for w in vu.expected}
        if collapsed_words:
            word_stats[:] = [ws for ws in word_stats if not (ws.file == fx.path.name and ws.word in collapsed_words)]
            contrast_stats[:] = [cs for cs in contrast_stats if not (cs.file == fx.path.name and cs.word in collapsed_words)]
        report.append(entry)
        print(f"{fx.path.name:32s} {fx.ref:20s} {entry['placed']:3d}/{entry['words']:<3d} words  cov {entry['coverage']:.2f}  post {entry['medianPosterior']}  {len(entry['issues'])} words w/ issues  {entry['alignMs']/1000:.0f} s")
        if len(per_verse) > 1:
            for v in per_verse:
                flag = "  ← COLLAPSED (not in audio?)" if v["collapsed"] else ""
                print(f"    {v['ref']:18s} {v['placed']:2d}/{v['words']:<2d}  {v['start']}–{v['end']} s  post {v['medianPosterior']}  {v['secondsPerWord']} s/word  issues {v['issues']}{flag}")
        if entry_peek:
            print(f"    hears (first 10 s): {entry_peek['first10s']}")
            print(f"    hears (last 10 s):  {entry_peek['last10s']}")
        shown = 0
        cstats = {(c["word"], c["ipa"]): c for c in raw["contrasts"]}
        for wi, issues in entry["issues"].items():
            if shown >= 12:
                print(f"    … {len(entry['issues']) - shown} more words with issues")
                break
            detail = "  ".join(f"[{ipa}: expected {c['expected']:.2f} vs contrast {c['contrast']:.2f}]" for (w, ipa), c in cstats.items() if w == wi)
            print(f"    word {wi} ({expected[wi].display}): " + "; ".join(issues) + "  " + detail)
            shown += 1
        for w in warnings:
            print(f"    ! {w}")
    return word_stats, contrast_stats, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REAL_AUDIO_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="write calibration.json even when the fixtures look like non-speech")
    ap.add_argument("--peek", action="store_true", help="also print what the model hears in the first/last 10 s of each file (helps labelling)")
    args = ap.parse_args()
    word_stats, contrast_stats, report = evaluate(args.dir, peek=args.peek)
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
