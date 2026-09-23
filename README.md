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
- **Build-time (deployed app):** set the GitHub repository variable `ALIGNMENT_API_URL`
  (Settings → Secrets and variables → Actions → Variables) to the server's HTTPS URL, e.g.
  `https://medaker-aligner.onrender.com`; the Pages workflow passes it as
  `NEXT_PUBLIC_ALIGNMENT_API_URL` and the deployed app uses the remote engine with no
  per-device setup. Locally: `NEXT_PUBLIC_ALIGNMENT_API_URL=… npm run dev` (see `.env.example`).
  The Settings sheet in the app shows the configured server and pings `/health`.

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

## Real-audio fixtures & threshold calibration (drop-in pipeline)

```
app/fixtures/real_audio/          ← drop recordings here: genesis_1_1_reader1.wav, i_samuel_3_2_dan.m4a …
app/fixtures/real_audio/manifest.json   optional per-file overrides (ref, text, reader, omittedWords, issues, skip)
app/fixtures/expected/            cached Temani expectations per verse (generated, committed)
app/fixtures/calibration.json     tuned thresholds (generated by the script, committed, loaded by the server)
```

1. Name the file `<book>_<chapter>_<verse>_<anything>.<ext>` (`.wav .m4a .mp3 .webm .ogg .flac`);
   book slugs are the Sefaria titles lower-cased with underscores (`genesis`, `i_samuel` or
   `1_samuel`, `song_of_songs`, `psalms` …). Or add a manifest entry with `"ref"`.
2. `python scripts/calibrate_fixtures.py`
   - verse text is fetched from Sefaria (WLC) unless the manifest gives `"text"`;
   - Temani expectations are generated by the frontend exporter (`../medaker`, or
     `MEDAKER_FRONTEND=/path`) and cached in `app/fixtures/expected/`;
   - every file is aligned; the report shows placed words, coverage, median posterior, the
     Temani issues the current thresholds raise, and warnings;
   - the pooled statistics tune the thresholds (`app/calibration.py`: word posterior/duration
     gates below the spoken distribution, contrast margins bounding false Temani issues to
     ≈ 5 %) and write `app/fixtures/calibration.json`, which `app/aligner.py` loads on start
     (`/health` shows `calibrated: true` and the values).
3. Restart the server (or rebuild the Docker image) and commit `calibration.json` +
   `expected/` + the recordings.

Annotate deliberate mistakes in the manifest to tune detection rather than only suppress
false positives: `"omittedWords": [3]` (reader skipped word 3), `"issues": {"5": ["w"]}`
(word 5's ו was read as V — the key is the IPA phoneme of the contrast: `w`, `g`, `dʒ`, `θ`,
`ð`, `ɣ`, `x`). The script reports how many annotated omissions/deviations the tuned
thresholds catch.

Safety: if the fixtures' median confidence is below the low-confidence level (synthetic
tones, noise), the script refuses to write `calibration.json` (use `--force` to override);
`--dry-run` prints without writing. `python scripts/align_file.py <file> --ref "Genesis 1:3"`
(or a verse-named file with no second argument) aligns one recording the same way.

## Memory (read this before choosing a host)

MMS_FA is a 315 M-parameter wav2vec2 model. What the server does about it
(`app/model_io.py`, `app/aligner.py`, `main.py`):

- **int8 dynamic quantisation of every Linear layer, done once** (`convert_to_int8`, at Docker
  build time or on a big machine) and saved as a 338 MB state dict. The float32 bundle is never
  built at runtime.
- **Streaming loader**: the module tree is created on the meta device (no float32 weights),
  each Linear gets a placeholder int8 layer, and the packed weights are set one layer at a time
  straight from the memory-mapped artifact — no second copy ever exists.
- **Chunked inference** (`MMS_CHUNK_S`, 6 s chunks with 0.5 s overlap, stitched at frame level)
  bounds activation memory regardless of clip length; tensors are freed and `gc.collect()`ed after
  every request.
- **One uvicorn worker**, one torch thread (`TORCH_THREADS=1`), `MALLOC_ARENA_MAX=2`, small
  concurrency limit — a second worker would double everything.

Measured (`/proc` RSS; x86 numbers from an amd64 container, `torch 2.5.1+cpu`, fbgemm/x86 engine):

| Stage | float32 bundle (before) | int8 streaming (now), x86 Linux | int8, macOS (qnnpack) |
|---|---|---|---|
| after `import torch` | 164 MB | 255 MB (181 anon) | 164 MB |
| model loaded | 1 905 MB, **peak 2 581 MB** | 995 MB, peak 995 MB — **569 MB anonymous** + 425 MB file-backed (mmap) | 318 MB |
| after a 6 s request | — | 1 094 MB (650 anon), peak 1 144 MB | 1 221 MB* |
| after a 12–20 s request | — | 1 140 MB (697 anon), peak 1 278 MB | 1 494 MB* |

\* qnnpack keeps dequantised float32 copies of the weights after the first forward; that is a
macOS-only artefact (dev machines), fbgemm on Linux does true int8 GEMM.

**Verdict for a 512 MB instance:** not achievable with this model. The packed int8 weights alone
are ≈ 315 MB and torch's runtime ≈ 180 MB before any request; under a real 512 MB cgroup the
process is OOM-killed while loading (reproduced with `docker run --memory=512m`). The changes
above cut the load peak by ~60 % and steady state by ~45 %, which makes **1 GB instances viable
and 2 GB comfortable**, but no loading trick shrinks 315 M parameters below the limit. Options:

1. **Hugging Face Spaces (Docker Space, free CPU basic: 2 vCPU / 16 GB RAM)** — the best free
   fit; push this repo as a Space, set `ALLOWED_ORIGINS`, use the Space URL as
   `ALIGNMENT_API_URL`.
2. **Google Cloud Run** with 2 GiB memory and min-instances 0 — pay-per-request, effectively free
   at this traffic; cold start ≈ 5 s with the int8 image.
3. **Render Standard (2 GB)** — the same image works unchanged.
4. A smaller aligner model (e.g. a ~95 M-parameter wav2vec2 with a character CTC head fine-tuned
   for romanised Hebrew) would fit 512 MB with these same loading tricks, but such a model would
   have to be trained; MMS is the only off-the-shelf multilingual character aligner.

Env knobs: `MMS_FA_PREFER_INT8` (1), `MMS_FA_INT8` (artifact path), `MMS_FA_INT8_URL` (download a
prebuilt artifact on first start — e.g. a GitHub release asset; MMS is CC-BY-NC 4.0, keep the
attribution), `MMS_FA_INT8_REQUIRED=1` (convert on first start; needs ≈ 3 GB), `MMS_CHUNK_S`,
`TORCH_THREADS`, `UVICORN_LIMIT_CONCURRENCY`. `/health` reports `modelVariant` and `peakRssMb`.

## Docker

```bash
docker build -t medaker-aligner .                    # converts + bakes the 338 MB int8 model (build needs ≈ 3 GB RAM)
docker build --build-arg PRELOAD=0 -t medaker-aligner .   # no model in the image → MMS_FA_INT8_URL or MMS_FA_INT8_REQUIRED=1 at runtime
docker run -p 8000:8000 --memory=1g -e ALLOWED_ORIGINS=https://barakwork95.github.io medaker-aligner
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
