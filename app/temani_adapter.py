"""
Temani phonetic adapter: expected IPA (from the client's transcribeTemani) → tokens of the
MMS forced-alignment model.

MMS_FA is a character-level CTC model trained on uroman-ised text: its vocabulary is the
lowercase Latin letters plus the apostrophe (and a wildcard). Each Temani phoneme is therefore
rendered as one or more Latin letters the way uroman would spell the sound:

    w→w  j→y  ʔ→'  ʕ→'  ħ→h  θ→th  ð→dh  ɣ→gh  dʒ→j  x→kh  ʃ→sh  tˤ→t  sˤ→s
    ɔ→o  ø→o  ə→e  a→a e→e i→i u→u

Gemination (doubled consonants in the IPA) is collapsed: CTC alignment gains nothing from a
repeated letter, and it would only steal frames.

Every token remembers which word / syllable / IPA phoneme it came from so the aligner can group
character spans back into phones, syllables and words.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# IPA (multi-character symbols first so they match before their prefixes)
IPA_TO_LATIN: list[tuple[str, str]] = [
    ("dʒ", "j"),
    ("tˤ", "t"),
    ("sˤ", "s"),
    ("θ", "th"),
    ("ð", "dh"),
    ("ɣ", "gh"),
    ("x", "kh"),
    ("ʃ", "sh"),
    ("ħ", "h"),
    ("ʔ", "'"),
    ("ʕ", "'"),
    ("j", "y"),
    ("w", "w"),
    ("ɔ", "o"),
    ("ø", "o"),
    ("ə", "e"),
    ("a", "a"),
    ("e", "e"),
    ("i", "i"),
    ("u", "u"),
    ("o", "o"),
]

SINGLE_LETTERS = set("bdfghklmnprstvyz")

# Temani contrasts the server checks acoustically. key = expected IPA phoneme.
#   ("pair", expected_letter, contrast_letter): compare mean posteriors of the two letters.
#       Validated on an authentic Yemenite reader (Genesis 2:4-14, 2026-10-09): 0 % false flags,
#       contrast − expected posterior ≈ −0.7 … −0.9 → reliable.
#   ("marker", marker_letter, contrast_letter): the digraph's distinguishing letter must show up.
#       NOT reliable: the uroman letter "h" is not what the model lights up for a dental fricative,
#       so TH/DH/GH/KH flagged 56–67 % of an authentic reader's phones. Disabled unless
#       MEDAKER_MARKER_CONTRASTS=1 (kept for experimentation / a future model).
MARKER_CONTRASTS_ENABLED = os.environ.get("MEDAKER_MARKER_CONTRASTS", "0") == "1"

CONTRASTS: dict[str, tuple[str, str, str, str]] = {
    # ipa: (kind, expected/marker letter, contrast letter, Hebrew issue text)
    "w": ("pair", "w", "v", "ו נשמעה כ־V (בהגייה התימנית: W)"),
    "g": ("pair", "g", "k", "ק נשמעה כ־K (בהגייה התימנית: G)"),
    "dʒ": ("pair", "j", "g", "גּ דגושה נשמעה כ־G (בהגייה התימנית: J)"),
    "θ": ("marker", "h", "t", "ת רפה נשמעה כ־T (בהגייה התימנית: TH)"),
    "ð": ("marker", "h", "d", "ד רפה נשמעה כ־D (בהגייה התימנית: DH)"),
    "ɣ": ("marker", "h", "g", "ג רפה נשמעה כ־G (בהגייה התימנית: GH)"),
    "x": ("marker", "h", "k", "כ רפה נשמעה כ־K (בהגייה התימנית: KH)"),
}


@dataclass(frozen=True)
class Token:
    letter: str  # one MMS character
    word: int
    syllable: int
    phone_index: int  # index of the phone within the word
    ipa: str  # the IPA phoneme this letter belongs to


@dataclass(frozen=True)
class PhoneRef:
    word: int
    syllable: int
    phone_index: int
    ipa: str
    letters: str


def active_contrasts() -> dict[str, tuple[str, str, str, str]]:
    """The contrast checks in force (marker checks only when explicitly enabled)."""
    return {k: v for k, v in CONTRASTS.items() if v[0] == "pair" or MARKER_CONTRASTS_ENABLED}


def ipa_to_phones(ipa: str) -> list[tuple[str, str]]:
    """Split an IPA syllable into (ipa_symbol, latin_letters) pairs; unknown symbols are dropped."""
    out: list[tuple[str, str]] = []
    i = 0
    s = ipa.replace(".", "")
    while i < len(s):
        for sym, latin in IPA_TO_LATIN:
            if s.startswith(sym, i):
                out.append((sym, latin))
                i += len(sym)
                break
        else:
            ch = s[i]
            if ch in SINGLE_LETTERS:
                out.append((ch, ch))
            # anything else (stress marks, unknown) is skipped
            i += 1
    # collapse gemination: identical consecutive phones
    collapsed: list[tuple[str, str]] = []
    for p in out:
        if collapsed and collapsed[-1] == p and p[0] not in "aeiouɔøə":
            continue
        collapsed.append(p)
    return collapsed


def tokenize_expected(expected: list) -> tuple[list[list[Token]], list[PhoneRef]]:
    """
    expected: list of ExpectedWord (contract). Returns tokens grouped per word (in order) and
    the flat list of phones (one per IPA phoneme) in the same order.
    """
    words_tokens: list[list[Token]] = []
    phones: list[PhoneRef] = []
    for w in expected:
        tokens: list[Token] = []
        phone_index = 0
        for syl in w.syllables:
            for sym, latin in ipa_to_phones(syl.ipa):
                phones.append(PhoneRef(w.index, syl.index, phone_index, sym, latin))
                for letter in latin:
                    tokens.append(Token(letter, w.index, syl.index, phone_index, sym))
                phone_index += 1
        if not tokens:  # a word that produced no tokens (should not happen) — keep alignment sane
            tokens.append(Token("'", w.index, 0, 0, "ʔ"))
            phones.append(PhoneRef(w.index, 0, 0, "ʔ", "'"))
        words_tokens.append(tokens)
    return words_tokens, phones


def transcript_of(words_tokens: list[list[Token]]) -> list[str]:
    """Words as MMS-ready strings, e.g. ["bereshith", "bara", "'elohim"]."""
    return ["".join(t.letter for t in toks) for toks in words_tokens]
