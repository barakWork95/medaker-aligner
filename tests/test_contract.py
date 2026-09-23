import json

import pytest
from pydantic import ValidationError

from app.contract import AlignRequest, AlignResponse, AlignWord, Phone, Phonetic, SyllableSpan


def test_request_round_trip(genesis_1_1_expected):
    req = AlignRequest(
        contractVersion=1,
        tradition="temani",
        verse={"ref": {"book": "Genesis", "chapter": 1, "verse": 1}, "text": "…"},
        expected=[w.model_dump() for w in genesis_1_1_expected],
        profile={"version": 1, "baselineF0": 130},
        audio={"mimeType": "audio/wav", "durationMs": 1000, "base64": "AAAA"},
    )
    assert req.expected[2].markId == "etnahta"
    assert req.expected[2].disjunctive is True


def test_request_rejects_wrong_version_or_tradition(genesis_1_1_expected):
    base = dict(verse={"ref": {"book": "Genesis", "chapter": 1, "verse": 1}, "text": "…"}, expected=[], audio={"mimeType": "audio/wav", "durationMs": 1, "base64": "AA"})
    with pytest.raises(ValidationError):
        AlignRequest(contractVersion=2, tradition="temani", **base)
    with pytest.raises(ValidationError):
        AlignRequest(contractVersion=1, tradition="ashkenazi", **base)


def test_response_matches_the_typescript_validator():
    """medaker's isAlignResponse() needs contractVersion 1 and words[].index/syllables/phones."""
    resp = AlignResponse(
        engine="test",
        words=[
            AlignWord(index=0, start=0.1, end=0.5, syllables=[SyllableSpan(index=0, start=0.1, end=0.5)], phones=[Phone(phone="b", start=0.1, end=0.2, score=0.9)], phonetic=Phonetic(score=0.9, issues=[])),
            AlignWord(index=1, start=None, end=None, syllables=[], phones=[], phonetic=None),
        ],
    )
    data = json.loads(resp.model_dump_json())
    assert data["contractVersion"] == 1
    assert data["words"][1]["start"] is None
    for w in data["words"]:
        assert {"index", "start", "end", "syllables", "phones"} <= set(w)
    with pytest.raises(ValidationError):
        Phonetic(score=1.5, issues=[])
