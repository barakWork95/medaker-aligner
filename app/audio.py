"""Decode the request audio (base64; webm/opus, mp4/aac, ogg, wav …) to 16 kHz mono float32."""
from __future__ import annotations

import base64
import io
import shutil
import subprocess

import numpy as np
import soundfile as sf

TARGET_SR = 16000


class AudioDecodeError(RuntimeError):
    pass


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    # linear interpolation is fine for alignment input
    n = int(round(len(x) * target / sr))
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def decode_base64_audio(b64: str, mime_type: str) -> tuple[np.ndarray, int]:
    data = base64.b64decode(b64)
    if not data:
        raise AudioDecodeError("empty audio")
    # WAV / FLAC / OGG-vorbis: libsndfile handles them without ffmpeg
    if "wav" in mime_type or data[:4] == b"RIFF":
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return _resample(x.mean(axis=1), sr, TARGET_SR), TARGET_SR
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioDecodeError("ffmpeg is required to decode %s (install ffmpeg or send WAV)" % mime_type)
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "pipe:1"],
        input=data,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise AudioDecodeError("ffmpeg failed: %s" % proc.stderr.decode(errors="ignore")[-300:])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), TARGET_SR
