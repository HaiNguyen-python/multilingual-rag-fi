"""Text normalisation, Finnish analysis and language detection.

Finnish is hard for keyword search for two reasons:
- inflection: one noun has many forms (kortti, kortin, kortilla, korttia ...);
- compounds: words are glued together (kirjasto + kortti = kirjastokortti).
A query for "kortti" should still find "kirjastokortin". The analyser below stems every word with
the Snowball Finnish stemmer and adds the stems of compound parts it can recognise.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

import snowballstemmer

# A number with an optional decimal comma ("3,20") or a run of letters. Hyphens and spaces split.
TOKEN_RE = re.compile(r"\d+(?:[,.]\d+)?|[^\W\d_]+")

FI_STOPWORDS = frozenset(
    """
    ja tai sekä on ovat oli ei en et emme se ne että kun jos niin kuin mutta myös vain
    voi voit voiko voinko saa saako saanko minä minun minulla minulle minut mua sinä sinun
    sinulla sinulle hän me te he mitä mikä miten kuinka missä mihin mistä milloin kuka
    montako paljonko onko pitää pitääkö täytyy tämä tuo siitä sen niitä niiden joka jotka
    jonka joita joissa joilla kanssa olla jo eikä esimerkiksi kuten jotain mitään oma oman
    omat teen tehdä tehdään
    """.split()
)

_STEMMER = snowballstemmer.stemmer("finnish")


def normalise(text: str) -> str:
    """Unicode NFC, lower case, unify dashes and drop soft hyphens.

    NFC matters for Finnish and Vietnamese: "ä" typed as "a" + combining diaeresis would
    otherwise never match the precomposed "ä" in the documents.
    """
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    text = re.sub(r"[‐-―−]", "-", text)
    return text.lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalise(text))


def strip_diacritics(text: str) -> str:
    """'thẻ thư viện' -> 'the thu vien'. Used for Vietnamese typed without diacritics."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_VOWELS = frozenset("aeiouyäö")


@lru_cache(maxsize=50_000)
def stem(word: str) -> str:
    """Snowball Finnish stem plus a final-vowel trim. Numbers are returned unchanged.

    Snowball does not undo consonant gradation, so different forms of one word can end up one
    vowel apart: vaippa -> "vaip" but vaipat -> "vaipa"; lääke -> "lääke" but lääkkeet ->
    "lääk". Dropping one final vowel from stems of five or more letters joins these pairs.
    The cost is a few merges of related words (maksu / maksaa), which is fine for search.
    """
    if word[0].isdigit():
        return word
    stemmed = str(_STEMMER.stemWord(word))
    if len(stemmed) >= 5 and stemmed[-1] in _VOWELS:
        return stemmed[:-1]
    return stemmed


class FinnishAnalyzer:
    """Turns Finnish text into index terms: stems plus stems of recognised compound parts.

    Compound splitting needs a lexicon of known stems, built from the corpus itself. A split is
    accepted only when both parts are known words, so made-up splits like "kirjas + tokortti"
    never happen.
    """

    min_prefix = 4
    min_suffix = 3

    def __init__(self, lexicon: Iterable[str] = ()) -> None:
        self.lexicon = frozenset(lexicon)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> FinnishAnalyzer:
        words = {t for text in texts for t in tokenize(text) if not t[0].isdigit()}
        return cls(stem(w) for w in words if len(w) >= 3 and w not in FI_STOPWORDS)

    def _known(self, part: str) -> bool:
        # Stopwords and very short stems would make many junk splits (e.g. a tail "om").
        if part in FI_STOPWORDS or len(stem(part)) < 3:
            return False
        return part in self.lexicon or stem(part) in self.lexicon

    @lru_cache(maxsize=20_000)  # noqa: B019 - analyser lives as long as the index
    def split_compound(self, word: str) -> tuple[str, ...]:
        """Return the stems of the parts of a compound, or () if it is not a known compound.

        Tries the longest known prefix first and recurses on the rest, so three-part words
        (e.g. "kirjasto + kortti + hakemus") are split too. The cut must leave a real word stem
        after it, not just a case ending: "terveysasema + lle" is an inflected form, not a
        compound with the word "alle".
        """
        last_cut = min(len(word) - self.min_suffix, len(stem(word)) - 2)
        for cut in range(last_cut, self.min_prefix - 1, -1):
            head, tail = word[:cut], word[cut:]
            if not self._known(head):
                continue
            if self._known(tail):
                return (stem(head), stem(tail))
            rest = self.split_compound(tail)
            if rest:
                return (stem(head), *rest)
        return ()

    def terms(self, text: str) -> list[str]:
        out: list[str] = []
        for token in tokenize(text):
            if token in FI_STOPWORDS:
                continue
            out.append(stem(token))
            if not token[0].isdigit() and len(token) >= self.min_prefix + self.min_suffix:
                out.extend(self.split_compound(token))
        return out


# --- language detection -------------------------------------------------------------------

_VI_CHARS = frozenset("ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
_FI_CHARS = frozenset("äöå")
_VI_WORDS = frozenset(
    "không khong của cua được duoc tôi toi làm lam sao bao nhiêu nhieu ở đâu dau thì thi "
    "có co là gì gi nào nao".split()
)
_FI_WORDS = frozenset(
    "ja on ei miten kuinka mitä mikä missä mihin mistä milloin voiko saako onko paljonko "
    "montako minun minulla pitää pitääkö jos kun että".split()
)
_EN_WORDS = frozenset(
    "the a an is are do does how what where when which can i my to of for much is "
    "should will with".split()
)
LANGUAGES = ("fi", "en", "vi")


def detect_language(text: str) -> str:
    """Guess fi / en / vi from letters and function words.

    Vietnamese-only letters (ơ, ư, đ, tone marks) and Finnish ä/ö/å are strong signals; short
    function words decide the rest. Empty input falls back to English; ties prefer the more
    specific signal (vi, then fi, then en).
    """
    lowered = normalise(text)
    words = tokenize(lowered)
    scores = {
        "vi": sum(ch in _VI_CHARS for ch in lowered) + 2 * sum(w in _VI_WORDS for w in words),
        "fi": sum(ch in _FI_CHARS for ch in lowered) + 2 * sum(w in _FI_WORDS for w in words),
        "en": 2 * sum(w in _EN_WORDS for w in words) + 1,
    }
    return max(("vi", "fi", "en"), key=lambda lang: scores[lang])
