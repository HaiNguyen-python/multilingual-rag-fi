import pytest

from mlrag.text import FinnishAnalyzer, detect_language, normalise, stem, strip_diacritics, tokenize
from mlrag.translate import Glossary, GlossaryTranslator, LLMTranslator, load_glossary


def test_normalise_unifies_unicode_forms_and_dashes():
    decomposed = "ja\u0308te\u00adasema \u2013 Ka\u0308ytetyt"  # combining diaeresis, soft hyphen
    assert normalise(decomposed) == "jäteasema - käytetyt"


def test_tokenize_keeps_decimal_comma_prices_together():
    assert tokenize("Kertalippu maksaa 3,20 euroa (K18).") == [
        "kertalippu",
        "maksaa",
        "3,20",
        "euroa",
        "k",
        "18",
    ]


@pytest.mark.parametrize(
    "forms",
    [
        ["kortti", "kortin", "korttia", "kortilla"],
        ["lippu", "lipun"],  # consonant gradation pp/p
        ["vaippa", "vaipat"],  # Snowball leaves these one vowel apart; the trim joins them
        ["lääke", "lääkkeet", "lääkkeistä"],
    ],
)
def test_inflected_forms_share_one_stem(forms):
    assert len({stem(f) for f in forms}) == 1


def test_compound_split_needs_both_parts_to_be_known_words():
    analyzer = FinnishAnalyzer.from_texts(["Kirjasto on auki.", "Uusi kortti maksaa."])
    assert analyzer.split_compound("kirjastokortin") == (stem("kirjasto"), stem("kortti"))
    # "kirjas" is not a word, so no split; unknown words are left alone
    assert analyzer.split_compound("kalastuslupa") == ()


def test_inflection_is_not_mistaken_for_a_compound():
    # "alle" is a Finnish word, but "terveysasema + lle" is just the allative case.
    analyzer = FinnishAnalyzer.from_texts(["Alle 18-vuotiaat.", "Terveysasema on auki."])
    assert analyzer.split_compound("terveysasemalle") == ()


def test_terms_drop_stopwords_and_add_compound_parts():
    analyzer = FinnishAnalyzer.from_texts(["kirjasto", "kortti"])
    terms = analyzer.terms("Miten saan kirjastokortin?")
    assert stem("miten") not in terms
    assert {stem("kirjastokortin"), stem("kirjasto"), stem("kortti")} <= set(terms)


def test_strip_diacritics_handles_vietnamese_d_bar():
    assert strip_diacritics("Đặt lịch khám") == "Dat lich kham"


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Miten saan kirjastokortin?", "fi"),
        ("Paljonko vuorokausilippu maksaa?", "fi"),
        ("How do I get a library card?", "en"),
        ("Làm sao để làm thẻ thư viện?", "vi"),
        ("lam sao de lam the thu vien", "vi"),  # Vietnamese typed without diacritics
        ("", "en"),
    ],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


# --- glossary translator ---


def _translator() -> GlossaryTranslator:
    return GlossaryTranslator(
        Glossary(
            {
                "en": {"library card": "kirjastokortti", "card": "kortti", "the": "", "how": ""},
                "vi": {"thẻ thư viện": "kirjastokortti", "thẻ": "kortti", "làm sao": ""},
            }
        )
    )


def test_longest_phrase_wins_and_stopwords_are_dropped():
    t = _translator().to_finnish("How do I renew the library card?", "en")
    assert t.text == "do i renew kirjastokortti"
    assert t.unknown == ("do", "i", "renew")
    assert t.coverage == 0.25


def test_numbers_count_as_understood():
    t = _translator().to_finnish("card 30", "en")
    assert t.text == "kortti 30" and t.coverage == 1.0


def test_vietnamese_without_diacritics_matches_multi_syllable_phrases_only():
    t = _translator()
    assert t.to_finnish("lam sao the thu vien", "vi").text == "kirjastokortti"
    # a single stripped syllable ("the") is ambiguous and is not translated
    assert t.to_finnish("the", "vi").unknown == ("the",)


def test_finnish_passes_through_unchanged():
    t = _translator().to_finnish("Miten saan kirjastokortin?", "fi")
    assert t.text == "Miten saan kirjastokortin?" and t.coverage == 1.0


def test_shipped_glossary_loads_and_normalises_keys_like_questions():
    glossary = load_glossary()
    assert glossary.lookup("driver s license", "en") == "ajokortti"  # from "driver's license"
    assert glossary.lookup("on", "en") == ""  # YAML would read a bare `on` as True
    t = GlossaryTranslator(glossary).to_finnish("Giấy phép đỗ xe cho cư dân giá bao nhiêu?", "vi")
    assert "asukaspysäköintitunnus" in t.text and "maksaa" in t.text


def test_reverse_glossary_maps_finnish_back_to_the_first_source_phrase():
    assert _translator().glossary.reverse("en")["kirjastokortti"] == "library card"


def test_llm_translator_calls_the_model_only_for_foreign_questions():
    class FakeClient:
        name = "fake"
        calls: list[str] = []

        def complete(self, system: str, user: str) -> str:
            self.calls.append(system)
            return "Paljonko kausilippu maksaa?"

    client = FakeClient()
    translator = LLMTranslator(client)
    assert translator.to_finnish("How much is a season ticket?", "en").text.startswith("Paljonko")
    assert "English" in client.calls[0]
    translator.to_finnish("Paljonko?", "fi")
    assert len(client.calls) == 1
