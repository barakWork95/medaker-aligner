from app.contract import ExpectedSyllable, ExpectedWord
from app.temani_adapter import CONTRASTS, ipa_to_phones, tokenize_expected, transcript_of


def word(index: int, sylls: list[str], stressed: int = -1) -> ExpectedWord:
    return ExpectedWord(
        index=index,
        display="x",
        pointed="x",
        ipa=".".join(sylls),
        roman="x",
        syllables=[ExpectedSyllable(index=i, ipa=s, stressed=(i == stressed), weight=1.0) for i, s in enumerate(sylls)],
    )


def test_ipa_to_latin_covers_temani_phonemes():
    assert ipa_to_phones("wa") == [("w", "w"), ("a", "a")]
    assert ipa_to_phones("gɔ") == [("g", "g"), ("ɔ", "o")]
    assert ipa_to_phones("dʒɔ") == [("dʒ", "j"), ("ɔ", "o")]
    assert ipa_to_phones("ʃiθ") == [("ʃ", "sh"), ("i", "i"), ("θ", "th")]
    assert ipa_to_phones("ðøʃ") == [("ð", "dh"), ("ø", "o"), ("ʃ", "sh")]
    assert ipa_to_phones("ħɔ") == [("ħ", "h"), ("ɔ", "o")]
    assert ipa_to_phones("ʕa") == [("ʕ", "'"), ("a", "a")]
    assert ipa_to_phones("tˤøv") == [("tˤ", "t"), ("ø", "o"), ("v", "v")]
    assert ipa_to_phones("sˤa") == [("sˤ", "s"), ("a", "a")]
    assert ipa_to_phones("xɔm") == [("x", "kh"), ("ɔ", "o"), ("m", "m")]
    assert ipa_to_phones("ɣal") == [("ɣ", "gh"), ("a", "a"), ("l", "l")]
    assert ipa_to_phones("jjø") == [("j", "y"), ("ø", "o")]  # gemination collapsed


def test_tokens_remember_word_syllable_and_phone():
    w = word(3, ["bə", "re", "ʃiθ"], stressed=2)
    tokens, phones = tokenize_expected([w])
    assert transcript_of(tokens) == ["bereshith"]
    assert [p.ipa for p in phones] == ["b", "ə", "r", "e", "ʃ", "i", "θ"]
    assert [t.syllable for t in tokens[0]] == [0, 0, 1, 1, 2, 2, 2, 2, 2]  # sh → 2 letters, th → 2 letters
    assert [t.phone_index for t in tokens[0]] == [0, 1, 2, 3, 4, 4, 5, 6, 6]
    assert all(t.word == 3 for t in tokens[0])


def test_genesis_1_1_transcript(genesis_1_1_expected):
    tokens, phones = tokenize_expected(genesis_1_1_expected)
    # qamats → o (Temani å), segol → a, ת rafe → th, ו → w, צ → s, ש → sh
    assert transcript_of(tokens) == ["bereshith", "boro", "'alohim", "'eth", "hashomayim", "we'eth", "ho'oras"]
    assert len(tokens) == 7
    assert [p.ipa for p in phones if p.word == 6] == ["h", "ɔ", "ʔ", "ɔ", "r", "a", "sˤ"]


def test_contrast_table_targets_the_temani_shifts():
    assert CONTRASTS["w"][:3] == ("pair", "w", "v")
    assert CONTRASTS["g"][:3] == ("pair", "g", "k")
    assert CONTRASTS["θ"][:3] == ("marker", "h", "t")
    for spec in CONTRASTS.values():
        assert "בהגייה התימנית" in spec[3]


def test_empty_word_gets_a_placeholder_token():
    w = word(0, ["…"])
    tokens, phones = tokenize_expected([w])
    assert transcript_of(tokens) == ["'"]
    assert phones[0].word == 0
