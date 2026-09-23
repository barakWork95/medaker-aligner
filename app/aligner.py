"""
MMS forced alignment (torchaudio.pipelines.MMS_FA) for the Medaker contract.

Pipeline
  1. tokens  = temani_adapter.tokenize_expected(expected)           (per word, per phone)
  2. emission = MMS_FA model(waveform)                               (frames × vocab log-probs)
  3. path    = torchaudio.functional.forced_align(emission, tokens)  (CTC Viterbi)
  4. spans   = merge_tokens(path)  → one TokenSpan per character with a mean posterior
  5. group character spans into phones → syllables → words with timestamps in seconds
  6. scores  = mean posterior per phone / word; Temani contrast checks on the emission
  7. words with a negligible span or posterior are reported as not found (start = null)

The model is loaded lazily on first use (~1.2 GB download into the torch hub cache).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contract import AlignWord, ExpectedWord, Phone, Phonetic, SyllableSpan
from .temani_adapter import CONTRASTS, PhoneRef, Token, tokenize_expected

ENGINE_ID = "mms-fa-torchaudio"

# Omission rule (forced alignment always places every word; an unspoken word gets squeezed into
# a sliver of low-confidence frames): a word is reported as not found when its span is shorter
# than MIN_WORD_S, or when it is short (< SQUEEZED_S) AND its mean posterior is below
# MIN_WORD_POSTERIOR AND below OMIT_RELATIVE × the median word posterior of the utterance.
# The relative clause keeps a uniformly low-confidence recording (noise, non-speech, unusual
# voice) from being emptied out — it gets a warning instead.
MIN_WORD_S = 0.05
SQUEEZED_S = 0.15
MIN_WORD_POSTERIOR = 0.15
OMIT_RELATIVE = 0.4
LOW_CONFIDENCE_MEDIAN = 0.1
# contrast checks: the competing letter must win by this much (mean posterior) to raise an issue
CONTRAST_MARGIN = 0.15
MARKER_MIN = 0.12


@dataclass
class CharSpan:
    token: Token
    start_frame: int
    end_frame: int  # exclusive
    score: float


class MMSAligner:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._bundle = None
        self._model = None
        self._tokenizer = None
        self._dictionary: dict[str, int] = {}

    # ---- model ----
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchaudio

        self._bundle = torchaudio.pipelines.MMS_FA
        self._model = self._bundle.get_model(with_star=False).to(self.device).eval()
        self._dictionary = self._bundle.get_dict(star=None)
        self._torch = torch
        self._torchaudio = torchaudio

    @property
    def sample_rate(self) -> int:
        self.load()
        return self._bundle.sample_rate  # 16000

    # ---- alignment ----
    def align(self, waveform: np.ndarray, sr: int, expected: list[ExpectedWord]) -> tuple[list[AlignWord], list[str]]:
        self.load()
        torch = self._torch
        F = self._torchaudio.functional
        warnings: list[str] = []
        if sr != self.sample_rate:
            raise ValueError("waveform must be %d Hz" % self.sample_rate)

        words_tokens, phones = tokenize_expected(expected)
        flat: list[Token] = [t for toks in words_tokens for t in toks]
        letters = [t.letter for t in flat]
        unknown = sorted({ch for ch in letters if ch not in self._dictionary})
        if unknown:
            warnings.append("letters outside the model vocabulary were dropped: %s" % ",".join(unknown))
            keep = [i for i, ch in enumerate(letters) if ch in self._dictionary]
            flat = [flat[i] for i in keep]
            letters = [letters[i] for i in keep]
        if not flat:
            return [self._not_found(w) for w in expected], warnings + ["no alignable tokens"]

        wave = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            emission, _ = self._model(wave)
        emission = torch.log_softmax(emission, dim=-1)[0].cpu()  # (frames, vocab)
        targets = torch.tensor([[self._dictionary[ch] for ch in letters]], dtype=torch.int32)
        n_frames = emission.size(0)
        if targets.size(1) > n_frames:
            warnings.append("audio too short for the transcript (%d frames < %d tokens)" % (n_frames, targets.size(1)))
            return [self._not_found(w) for w in expected], warnings
        aligned, scores = F.forced_align(emission.unsqueeze(0), targets, blank=0)
        token_spans = F.merge_tokens(aligned[0], scores[0].exp())  # TokenSpan(token, start, end, score)
        ratio = waveform.shape[0] / n_frames / sr  # seconds per frame

        # token_spans are in transcript order, one per non-blank token
        char_spans: list[CharSpan] = []
        for span, tok in zip(token_spans, flat):
            char_spans.append(CharSpan(tok, span.start, span.end, float(span.score)))
        if len(char_spans) != len(flat):
            warnings.append("token/span count mismatch (%d vs %d)" % (len(char_spans), len(flat)))

        emission_np = emission.numpy()
        per_word = {w.index: [c for c in char_spans if c.token.word == w.index] for w in expected}
        posteriors = [float(np.mean([c.score for c in sp])) for sp in per_word.values() if sp]
        median_post = float(np.median(posteriors)) if posteriors else 0.0
        if median_post < LOW_CONFIDENCE_MEDIAN:
            warnings.append("low acoustic confidence (median word posterior %.2f): noisy audio, not speech, or a very unusual voice" % median_post)

        words_out: list[AlignWord] = []
        for w in expected:
            spans = per_word[w.index]
            if not spans:
                words_out.append(self._not_found(w))
                continue
            start = spans[0].start_frame * ratio
            end = spans[-1].end_frame * ratio
            word_post = float(np.mean([c.score for c in spans]))
            squeezed = end - start < SQUEEZED_S and word_post < MIN_WORD_POSTERIOR and word_post < OMIT_RELATIVE * median_post
            if end - start < MIN_WORD_S or squeezed:
                words_out.append(self._not_found(w))
                continue

            # phones: group char spans by phone_index
            phone_list: list[Phone] = []
            issues: list[str] = []
            by_phone: dict[int, list[CharSpan]] = {}
            for c in spans:
                by_phone.setdefault(c.token.phone_index, []).append(c)
            for pi in sorted(by_phone):
                group = by_phone[pi]
                p_start = group[0].start_frame
                p_end = group[-1].end_frame
                p_score = float(np.mean([c.score for c in group]))
                ipa = group[0].token.ipa
                phone_list.append(Phone(phone=ipa, start=round(p_start * ratio, 3), end=round(p_end * ratio, 3), score=round(p_score, 3)))
                issue = self._contrast_issue(ipa, emission_np, p_start, p_end)
                if issue and issue not in issues:
                    issues.append(issue)

            # syllables: first/last char of each syllable index
            syl_spans: list[SyllableSpan] = []
            by_syl: dict[int, list[CharSpan]] = {}
            for c in spans:
                by_syl.setdefault(c.token.syllable, []).append(c)
            for si in sorted(by_syl):
                g = by_syl[si]
                syl_spans.append(SyllableSpan(index=si, start=round(g[0].start_frame * ratio, 3), end=round(g[-1].end_frame * ratio, 3)))

            # phonetic score: posterior confidence, minus a penalty per detected Temani deviation
            score = max(0.0, min(1.0, word_post - 0.2 * len(issues)))
            words_out.append(
                AlignWord(
                    index=w.index,
                    start=round(start, 3),
                    end=round(end, 3),
                    syllables=syl_spans,
                    phones=phone_list,
                    phonetic=Phonetic(score=round(score, 3), issues=issues),
                )
            )
        return words_out, warnings

    # ---- helpers ----
    @staticmethod
    def _not_found(w: ExpectedWord) -> AlignWord:
        return AlignWord(index=w.index, start=None, end=None, syllables=[], phones=[], phonetic=None)

    def _contrast_issue(self, ipa: str, emission: np.ndarray, start: int, end: int) -> Optional[str]:
        spec = CONTRASTS.get(ipa)
        if not spec or end <= start:
            return None
        kind, expected_letter, contrast_letter, text = spec
        seg = emission[start:end]  # log-probs
        probs = np.exp(seg)
        e = self._dictionary.get(expected_letter)
        c = self._dictionary.get(contrast_letter)
        if e is None or c is None:
            return None
        if kind == "pair":
            if float(probs[:, c].mean()) > float(probs[:, e].mean()) + CONTRAST_MARGIN:
                return text
        else:  # marker: the digraph's 'h' must be evidenced somewhere in the span
            if float(probs[:, e].max()) < MARKER_MIN and float(probs[:, c].mean()) > MARKER_MIN:
                return text
        return None


_singleton: Optional[MMSAligner] = None


def get_aligner() -> MMSAligner:
    global _singleton
    if _singleton is None:
        _singleton = MMSAligner()
    return _singleton


def phones_for_expected(expected: list[ExpectedWord]) -> list[PhoneRef]:
    """Exposed for tests / debugging: the phone plan without running the model."""
    return tokenize_expected(expected)[1]


def seconds(frames: int, ratio: float) -> float:
    return round(frames * ratio, 3)


def is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))
