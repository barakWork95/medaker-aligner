"""Run the aligner locally: `python main.py` → http://localhost:8000 (docs at /docs)."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")), reload=os.environ.get("RELOAD") == "1")
