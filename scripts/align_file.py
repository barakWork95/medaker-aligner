"""
Align a recording against a verse and print the /api/align response.

    python scripts/align_file.py samples/genesis-1-1.wav samples/genesis-1-1.expected.json
    python scripts/align_file.py rec.m4a --ref "Genesis 1:3"            # text from Sefaria, expectations from the frontend
    python scripts/align_file.py app/fixtures/real_audio/genesis_1_1_reader1.wav   # verse inferred from the filename
    python scripts/align_file.py my.webm expected.json --server http://localhost:8000   # via HTTP

Explicit expected.json comes from the frontend:
    cd ../medaker && npx vite-node --config vitest.config.mts scripts/export-expected.ts "<pointed verse>" > expected.json
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("expected", nargs="?", help="expected.json; omit to resolve from --ref or the filename")
    ap.add_argument("--ref", help='verse reference, e.g. "Genesis 1:1" (text fetched from Sefaria, expectations from the frontend)')
    ap.add_argument("--server", help="POST to a running server instead of aligning in-process")
    args = ap.parse_args()
    data = Path(args.audio).read_bytes()
    mime = mimetypes.guess_type(args.audio)[0] or "audio/wav"
    if args.expected:
        expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    else:
        from app.fixtures.registry import ref_from_filename, resolve_expected

        ref = args.ref or ref_from_filename(Path(args.audio).name)
        if not ref:
            ap.error("give expected.json, --ref, or name the file <book>_<chapter>_<verse>_*.ext")
        _, expected = resolve_expected(ref, None)
        print(f"# {ref}: {len(expected)} words", file=sys.stderr)
    body = {
        "contractVersion": 1,
        "tradition": "temani",
        "verse": {"ref": {"book": "?", "chapter": 0, "verse": 0}, "text": " ".join(w["pointed"] for w in expected)},
        "expected": expected,
        "profile": None,
        "audio": {"mimeType": mime, "durationMs": 0, "base64": base64.b64encode(data).decode()},
    }
    if args.server:
        import urllib.request

        req = urllib.request.Request(args.server.rstrip("/") + "/api/align", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
    else:
        from app.aligner import ENGINE_ID, get_aligner
        from app.audio import decode_base64_audio
        from app.contract import AlignRequest

        req_model = AlignRequest(**body)
        wave, sr = decode_base64_audio(req_model.audio.base64, req_model.audio.mimeType)
        words, warnings = get_aligner().align(wave, sr, req_model.expected)
        out = {"contractVersion": 1, "engine": ENGINE_ID, "words": [w.model_dump() for w in words], "warnings": warnings}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n%-14s %8s %8s %6s  issues" % ("word", "start", "end", "score"), file=sys.stderr)
    for w, e in zip(out["words"], expected):
        if w["start"] is None:
            print("%-14s %8s %8s %6s" % (e["display"], "—", "—", "—"), file=sys.stderr)
        else:
            ph = w.get("phonetic") or {}
            print("%-14s %8.2f %8.2f %6.2f  %s" % (e["display"], w["start"], w["end"], ph.get("score", 0), "; ".join(ph.get("issues", []))), file=sys.stderr)


if __name__ == "__main__":
    main()
