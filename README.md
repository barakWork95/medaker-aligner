# medaker-aligner

Forced-alignment server for **Medaker (מד׳כר)** — implements `POST /api/align` (contract v1,
see `medaker/src/lib/speech/api-contract.ts`) and returns word / syllable / phoneme timestamps
plus Temani pronunciation checks for a recorded verse.

Engine: **Meta MMS forced aligner** via `torchaudio.pipelines.MMS_FA` — a multilingual
character-level CTC model that aligns any romanised transcript. The client already sends the
Temani expectations (IPA per syllable); `app/temani_adapter.py` renders them as the letters the
model knows (ו→w, ק→g, גּ→j, ג→gh, ת→th, ד→dh, כ→kh, ח→h, ע/א→', קמץ→o, סגול→a …) and the aligner
maps the character spans back to phones, syllables and words. Per-frame posteriors give a
confidence per phone/word and let the server test the key Temani contrasts against the audio
(W vs V, G vs K, TH vs T, DH vs D, GH vs G, KH vs K, J vs G) → Hebrew issue strings.

```
app/contract.py        pydantic models = api-contract.ts (AlignRequest / AlignResponse)
app/temani_adapter.py  IPA → MMS letters, with word/syllable/phone bookkeeping + contrast table
app/audio.py           base64 audio (webm/opus, mp4/aac, wav …) → 16 kHz mono (ffmpeg / libsndfile)
app/aligner.py         MMS_FA: emission → forced_align → spans → words/syllables/phones/scores/issues
app/main.py            FastAPI: POST /api/align, GET /health, CORS
main.py                uvicorn launcher (port 8000)
scripts/make_samples.py  synthetic verse-shaped WAV for schema tests (not speech)
scripts/align_file.py    align a file in-process or through a running server
samples/               Genesis 1:1 expected-words fixture (+ generated wav)
tests/                 adapter + contract (fast) · e2e with the model (opt-in)
```

## Run locally

Requirements: Python 3.10+, `ffmpeg` on PATH (`brew install ffmpeg` / `apt install ffmpeg`)
for browser recordings (WAV works without it).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt        # torch 2.5.1 CPU wheels ≈ 200 MB
python main.py                             # http://localhost:8000  (docs: /docs)
```

The first start downloads the MMS_FA weights (~1.2 GB) into `~/.cache/torch`; later starts
load in a few seconds. Set `PRELOAD_MODEL=0` to load lazily on the first request instead.
`ALLOWED_ORIGINS` (comma-separated) controls CORS; the default allows `localhost:3100/3101/3000`
and `https://barakwork95.github.io`.

Check: `curl localhost:8000/health` → `{"ok":true,"engine":"mms-fa-torchaudio","modelLoaded":true}`.

## Hook up the frontend

The app picks the remote engine when a base URL is configured, and falls back to the local
engine (with a visible notice) if the server is unreachable.

- **Per device (no rebuild):** open the app, and in the browser console run
  `localStorage.setItem('medaker.alignmentApi', 'http://localhost:8000')`, then record a verse
  and press "ניתוח הקריאה". The result header says **מנוע שרת** and the word detail shows
  **התאמה פונטית N%** with any Temani issues. `localStorage.removeItem('medaker.alignmentApi')`
  reverts.
- **Build-time:** `NEXT_PUBLIC_ALIGNMENT_API=https://your-host npm run build` in `medaker/`
  (for GitHub Pages: set it as a repository variable and pass it in the deploy workflow).

Phones on the same Wi-Fi: run the frontend with `npm run dev -- --hostname 0.0.0.0`, open
`http://<mac-ip>:3100`, and set the override to `http://<mac-ip>:8000` — add that origin to
`ALLOWED_ORIGINS`. Note that browsers only expose the microphone on `localhost` or HTTPS, so
for a real phone test either use the deployed HTTPS app with a tunnelled HTTPS server
(e.g. `cloudflared tunnel --url http://localhost:8000`) or an HTTPS dev setup.

## Test

```bash
pytest                                  # adapter + contract tests (no model)
MEDAKER_ALIGNER_E2E=1 pytest            # + real alignment on samples/genesis-1-1.wav
python scripts/make_samples.py          # (re)generate the synthetic sample
python scripts/align_file.py samples/genesis-1-1.wav samples/genesis-1-1.expected.json
python scripts/align_file.py rec.webm expected.json --server http://localhost:8000
```

`expected.json` for any verse comes from the frontend (it owns the Temani rules):

```bash
cd ../medaker && npx vite-node --config vitest.config.mts scripts/export-expected.ts "וַיֹּ֥אמֶר אֱלֹהִ֖ים יְהִ֣י א֑וֹר וַֽיְהִי־אֽוֹר׃" > ../medaker-aligner/expected.json
```

The synthetic sample is verse-shaped tones, not speech: it proves the schema and monotonic
timestamps, but posteriors are near zero and the server adds a "low acoustic confidence"
warning. For meaningful scores record yourself (see `samples/README.md`).

## Docker

```bash
docker build -t medaker-aligner .                    # bakes the 1.2 GB model into the image
docker build --build-arg PRELOAD=0 -t medaker-aligner .   # or download on first start
docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://barakwork95.github.io medaker-aligner
```

## How the response is produced

1. `tokenize_expected` → one MMS letter sequence per word, each letter tagged with its word,
   syllable and phone.
2. `MMS_FA` emission (20 ms frames) → `forced_align` (CTC Viterbi over the whole verse) →
   `merge_tokens` → one span + mean posterior per letter.
3. Letters are grouped back into phones (`phone`, `start`, `end`, `score`), syllables and words.
4. Omission rule: forced alignment always places every word; an unspoken word ends up as a
   sliver of low-confidence frames. A word is reported as `start: null` when its span is
   < 50 ms, or < 150 ms **and** its posterior is < 0.15 **and** < 0.4 × the utterance's median
   word posterior. A uniformly low-confidence recording is not emptied out; it gets a warning.
5. `phonetic.score` = mean posterior − 0.2 per detected Temani issue (clamped to 0..1);
   `issues` come from the contrast checks over the phone's frames (`CONTRAST_MARGIN` /
   `MARKER_MIN` in `app/aligner.py`). The frontend marks a word "minor" below 0.6.

Everything is CPU-only; a 5 s verse aligns in ≈ 0.5–1 s on a laptop.
