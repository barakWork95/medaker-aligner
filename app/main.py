"""FastAPI app — POST /api/align (contract v1), GET /health."""
from __future__ import annotations

import gc
import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .aligner import CALIBRATED, ENGINE_ID, THRESHOLDS, get_aligner
from .model_io import peak_rss_mb
from .audio import AudioDecodeError, decode_base64_audio
from .contract import AlignRequest, AlignResponse

log = logging.getLogger("medaker-aligner")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

DEFAULT_ORIGINS = ["http://localhost:3100", "http://localhost:3101", "http://localhost:3000", "https://barakwork95.github.io"]
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",") if o.strip()]

app = FastAPI(title="Medaker aligner", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST", "GET", "OPTIONS"], allow_headers=["*"])


@app.on_event("startup")
def _warm() -> None:
    if os.environ.get("PRELOAD_MODEL", "1") == "1":
        t0 = time.time()
        get_aligner().load()
        log.info("MMS_FA loaded in %.1fs", time.time() - t0)


@app.get("/health")
def health() -> dict:
    a = get_aligner()
    return {"ok": True, "engine": ENGINE_ID, "modelLoaded": a._model is not None, "modelVariant": a.variant, "peakRssMb": peak_rss_mb(), "calibrated": CALIBRATED, "thresholds": THRESHOLDS}


@app.post("/api/align", response_model=AlignResponse)
def align(req: AlignRequest) -> AlignResponse:
    t0 = time.time()
    try:
        wave, sr = decode_base64_audio(req.audio.base64, req.audio.mimeType)
    except AudioDecodeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if wave.size < sr // 10:
        raise HTTPException(status_code=400, detail="audio shorter than 100 ms")
    seconds = wave.size / sr
    try:
        words, warnings = get_aligner().align(wave, sr, req.expected)
    finally:
        del wave  # release the PCM buffer before the response is serialised
        gc.collect()
    log.info("aligned %s (%d words, %.1fs audio) in %.2fs", req.verse.ref, len(req.expected), seconds, time.time() - t0)
    return AlignResponse(engine=ENGINE_ID, words=words, warnings=warnings)
