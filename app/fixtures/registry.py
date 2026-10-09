"""
Real-audio fixture registry.

  app/fixtures/real_audio/<book>_<chapter>_<verse>_<anything>.<ext>   ← drop files here
  app/fixtures/real_audio/manifest.json                               ← optional overrides
  app/fixtures/expected/<Book>.<c>.<v>.json                           ← cached Temani expectations
  app/fixtures/calibration.json                                       ← tuned thresholds (output)

resolve_fixtures() returns one Fixture per audio file with its verse ref, pointed text and the
`expected` array. Text comes from the manifest or Sefaria (WLC); expectations come from the
frontend exporter (`../medaker/scripts/export-expected.ts`, run with vite-node) or the cache.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

FIXTURES_DIR = Path(__file__).resolve().parent
REAL_AUDIO_DIR = FIXTURES_DIR / "real_audio"
EXPECTED_DIR = FIXTURES_DIR / "expected"
MANIFEST = REAL_AUDIO_DIR / "manifest.json"
CALIBRATION_FILE = FIXTURES_DIR / "calibration.json"
AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".webm", ".ogg", ".flac", ".aac", ".mp4"}

SEFARIA_VERSION = "Tanach with Ta'amei Hamikra"
FRONTEND_DIR = Path(os.environ.get("MEDAKER_FRONTEND", FIXTURES_DIR.parents[2] / "medaker"))

BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "I Samuel", "II Samuel", "I Kings", "II Kings",
    "Isaiah", "Jeremiah", "Ezekiel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Psalms", "Proverbs", "Job", "Song of Songs", "Ruth", "Lamentations", "Ecclesiastes", "Esther", "Daniel",
    "Ezra", "Nehemiah", "I Chronicles", "II Chronicles",
]
_BOOK_SLUGS = {re.sub(r"[^a-z0-9]+", "_", b.lower()).strip("_"): b for b in BOOKS}
_BOOK_SLUGS.update({"1_samuel": "I Samuel", "2_samuel": "II Samuel", "1_kings": "I Kings", "2_kings": "II Kings", "1_chronicles": "I Chronicles", "2_chronicles": "II Chronicles", "song": "Song of Songs"})


@dataclass
class VerseUnit:
    ref: str  # "Genesis 2:4"
    text: str
    expected: list[dict]  # word indices local to the verse
    word_offset: int  # index of this verse's first word in the fixture's concatenated expected


@dataclass
class Fixture:
    path: Path
    ref: str  # "Genesis 1:1" or a range "Genesis 2:4-19" / "Exodus 14:26-15:26"
    text: str  # pointed text of all verses, space-joined
    expected: list[dict]  # concatenated, word indices re-numbered across verses
    verses: list[VerseUnit] = field(default_factory=list)
    reader: Optional[str] = None
    notes: Optional[str] = None
    omitted_words: list[int] = field(default_factory=list)
    issues: dict[int, list[str]] = field(default_factory=dict)

    @property
    def verse_key(self) -> str:
        book, cv = self.ref.rsplit(" ", 1)
        return f"{book.replace(' ', '_')}.{cv.replace(':', '.')}"

    @property
    def is_range(self) -> bool:
        return len(self.verses) > 1


def expand_ref(ref: str) -> list[str]:
    """
    "Genesis 1:1" → ["Genesis 1:1"]; "Genesis 2:4-19" → 2:4 … 2:19;
    "Exodus 14:26-15:26" → 14:26 … 15:26 (chapter lengths from Sefaria's shape API).
    """
    m = re.match(r"^(.+?) (\d+):(\d+)(?:-(?:(\d+):)?(\d+))?$", ref.strip())
    if not m:
        raise ValueError(f"bad ref {ref!r}")
    book, c1, v1, c2, v2 = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
    if v2 is None:
        return [f"{book} {c1}:{v1}"]
    c2 = int(c2) if c2 else c1
    v2 = int(v2)
    if c2 == c1:
        return [f"{book} {c1}:{v}" for v in range(v1, v2 + 1)]
    lengths = chapter_lengths(book)
    out = []
    for c in range(c1, c2 + 1):
        start = v1 if c == c1 else 1
        end = v2 if c == c2 else lengths[c - 1]
        out.extend(f"{book} {c}:{v}" for v in range(start, end + 1))
    return out


_shape_cache: dict[str, list[int]] = {}


def chapter_lengths(book: str) -> list[int]:
    if book not in _shape_cache:
        url = f"https://www.sefaria.org/api/shape/{urllib.parse.quote(book)}"
        with urllib.request.urlopen(url, timeout=30) as r:
            _shape_cache[book] = json.load(r)[0]["chapters"]
    return _shape_cache[book]


def ref_from_filename(name: str) -> Optional[str]:
    """genesis_1_1_reader1.wav → "Genesis 1:1"; i_samuel_3_2_x.m4a → "I Samuel 3:2"."""
    stem = Path(name).stem.lower()
    # try the longest book slug that prefixes the stem (handles "1_kings", "song_of_songs", …)
    for slug in sorted(_BOOK_SLUGS, key=len, reverse=True):
        m = re.match(rf"^{re.escape(slug)}_(\d+)_(\d+)(?:_.*)?$", stem)
        if m:
            return f"{_BOOK_SLUGS[slug]} {int(m.group(1))}:{int(m.group(2))}"
    return None


def normalize_hebrew(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s).replace("&thinsp;", " ").replace("&nbsp;", " ").replace("͏", "")
    s = re.sub(r"־\s+", "־", s)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFD", s)).strip()


def fetch_verse_text(ref: str) -> str:
    """Pointed WLC text of a verse from Sefaria API v3."""
    tref = ref.replace(" ", ".").replace(":", ".")
    url = f"https://www.sefaria.org/api/v3/texts/{urllib.parse.quote(tref)}?version=hebrew%7C{urllib.parse.quote(SEFARIA_VERSION)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    versions = data.get("versions") or []
    v = next((x for x in versions if x.get("versionTitle") == SEFARIA_VERSION), versions[0] if versions else None)
    if not v:
        raise RuntimeError(f"Sefaria returned no text for {ref}")
    text = v["text"]
    if isinstance(text, list):
        text = " ".join(text)
    return normalize_hebrew(text)


def export_expected(text: str, frontend: Path = FRONTEND_DIR) -> list[dict]:
    """Run the frontend's Temani exporter (owner of the phonetic rules)."""
    script = frontend / "scripts" / "export-expected.ts"
    if not script.exists():
        raise RuntimeError(f"frontend exporter not found at {script}; set MEDAKER_FRONTEND or commit the expected JSON into {EXPECTED_DIR}")
    proc = subprocess.run(["npx", "vite-node", "--config", "vitest.config.mts", "scripts/export-expected.ts", text], cwd=frontend, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"export-expected failed: {proc.stderr[-400:]}")
    return json.loads(proc.stdout)


def load_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    """Manifest entries keyed by NFC-normalised filename (macOS may hand us NFD names)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files", {}) if isinstance(data, dict) else {}
    return {unicodedata.normalize("NFC", k): v for k, v in files.items()}


def expected_cache_path(ref: str) -> Path:
    book, cv = ref.rsplit(" ", 1)
    return EXPECTED_DIR / f"{book.replace(' ', '_')}.{cv.replace(':', '.')}.json"


def resolve_expected(ref: str, text: Optional[str], *, allow_network: bool = True, allow_frontend: bool = True) -> tuple[str, list[dict]]:
    """Return (text, expected) for a verse, using the cache first."""
    cache = expected_cache_path(ref)
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if not text or normalize_hebrew(text) == cached.get("text"):
            return cached["text"], cached["expected"]
    if not text:
        if not allow_network:
            raise RuntimeError(f"no text for {ref} and network disabled")
        text = fetch_verse_text(ref)
    text = normalize_hebrew(text)
    if not allow_frontend:
        raise RuntimeError(f"no cached expectations for {ref}")
    expected = export_expected(text)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"ref": ref, "text": text, "expected": expected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return text, expected


def audio_files(directory: Path = REAL_AUDIO_DIR) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTS) if directory.exists() else []


def resolve_fixtures(directory: Path = REAL_AUDIO_DIR, manifest: Optional[dict[str, dict]] = None, **kw) -> list[Fixture]:
    manifest = load_manifest(directory / "manifest.json") if manifest is None else manifest
    out: list[Fixture] = []
    for path in audio_files(directory):
        entry = manifest.get(unicodedata.normalize("NFC", path.name), {})
        if entry.get("skip"):
            continue
        ref = entry.get("ref") or ref_from_filename(path.name)
        if not ref:
            print(f"! {path.name}: cannot infer the verse — add a manifest entry with \"ref\"")
            continue
        refs = expand_ref(ref)
        verses: list[VerseUnit] = []
        expected: list[dict] = []
        texts: list[str] = []
        for vref in refs:
            vtext, vexp = resolve_expected(vref, entry.get("text") if len(refs) == 1 else None, **kw)
            offset = len(expected)
            verses.append(VerseUnit(ref=vref, text=vtext, expected=vexp, word_offset=offset))
            for w in vexp:
                w = dict(w)
                w["index"] = offset + w["index"]
                expected.append(w)
            texts.append(vtext)
        text = " ".join(texts)
        out.append(
            Fixture(
                path=path,
                ref=ref,
                text=text,
                expected=expected,
                verses=verses,
                reader=entry.get("reader"),
                notes=entry.get("notes"),
                omitted_words=[int(i) for i in entry.get("omittedWords", [])],
                issues={int(k): list(v) for k, v in entry.get("issues", {}).items()},
            )
        )
    return out
