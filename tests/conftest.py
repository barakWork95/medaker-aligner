import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.contract import ExpectedWord  # noqa: E402

SAMPLES = ROOT / "samples"


@pytest.fixture
def genesis_1_1_expected() -> list[ExpectedWord]:
    data = json.loads((SAMPLES / "genesis-1-1.expected.json").read_text(encoding="utf-8"))
    return [ExpectedWord(**w) for w in data]


def e2e_enabled() -> bool:
    return os.environ.get("MEDAKER_ALIGNER_E2E") == "1"


requires_model = pytest.mark.skipif(not e2e_enabled(), reason="set MEDAKER_ALIGNER_E2E=1 to run the MMS model (downloads ~1.2 GB)")
