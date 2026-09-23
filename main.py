"""Run the aligner locally: `python main.py` → http://localhost:8000 (docs at /docs).

Memory-conscious defaults (Render free tier = 512 MB): ONE uvicorn worker, no reload,
tiny concurrency (alignment is CPU-bound and holds the model; parallel requests would only
multiply activation memory). Override with PORT / HOST / UVICORN_LIMIT_CONCURRENCY.
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        reload=os.environ.get("RELOAD") == "1",
        limit_concurrency=int(os.environ.get("UVICORN_LIMIT_CONCURRENCY", "4")),
        backlog=64,
        timeout_keep_alive=5,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
