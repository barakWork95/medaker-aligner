# Samples

- `genesis-1-1.expected.json` — the `expected` array for Genesis 1:1, exported from the
  frontend (`scripts/export-expected.ts`). Regenerate after changing the Temani rules.
- `genesis-1-1.wav` — generated on demand by `python scripts/make_samples.py` (synthetic,
  verse-shaped tones; good for schema/timestamp tests only). Git-ignored.

To test with real speech, record Genesis 1:1 in the Temani reading (any phone voice memo works),
convert to 16 kHz mono WAV (`ffmpeg -i rec.m4a -ac 1 -ar 16000 samples/genesis-1-1.wav`) and run
`python scripts/align_file.py samples/genesis-1-1.wav samples/genesis-1-1.expected.json`.
