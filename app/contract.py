"""Pydantic models mirroring medaker/src/lib/speech/api-contract.ts (contract version 1)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

CONTRACT_VERSION = 1


class VerseRef(BaseModel):
    book: str
    chapter: int
    verse: int


class Verse(BaseModel):
    ref: VerseRef
    text: str


class ExpectedSyllable(BaseModel):
    index: int
    ipa: str
    stressed: bool
    weight: float


class ExpectedWord(BaseModel):
    index: int
    display: str
    pointed: str
    ipa: str
    roman: str
    syllables: list[ExpectedSyllable]
    markId: Optional[str] = None
    disjunctive: bool = False


class Audio(BaseModel):
    mimeType: str
    durationMs: int
    base64: str


class AlignRequest(BaseModel):
    contractVersion: Literal[1]
    tradition: Literal["temani"]
    verse: Verse
    expected: list[ExpectedWord]
    profile: Optional[dict] = None  # VoiceProfile — opaque to the server
    audio: Audio


class Phone(BaseModel):
    phone: str
    start: float
    end: float
    score: Optional[float] = None


class SyllableSpan(BaseModel):
    index: int
    start: float
    end: float


class Phonetic(BaseModel):
    score: float = Field(ge=0, le=1)
    issues: list[str] = []


class AlignWord(BaseModel):
    index: int
    start: Optional[float]
    end: Optional[float]
    syllables: list[SyllableSpan]
    phones: list[Phone]
    phonetic: Optional[Phonetic] = None


class AlignResponse(BaseModel):
    contractVersion: Literal[1] = CONTRACT_VERSION
    engine: str
    words: list[AlignWord]
    warnings: list[str] = []
