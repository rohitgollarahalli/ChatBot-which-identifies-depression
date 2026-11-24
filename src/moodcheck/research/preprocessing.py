"""Text cleaning: lowercase, strip punctuation, drop stop words, Porter-stem."""

from __future__ import annotations

import hashlib
import re
import string

import nltk

# Bump when the cleaning steps change in a way that alters the tokens, so saved
# models trained with the old steps are rejected.
PREPROCESSING_VERSION = "1"

# Minimal stop-word list used only if the NLTK corpus cannot be downloaded
# (e.g. offline / CI). The full NLTK list is preferred when available.
_FALLBACK_STOPWORDS = {
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "it",
    "they",
    "them",
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "of",
    "at",
    "by",
    "for",
    "with",
    "to",
    "from",
    "in",
    "on",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "this",
    "that",
    "these",
    "those",
    "as",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "not",
    "no",
    "nor",
    "s",
    "t",
    "don",
    "now",
}

# Resolved once per process.
_stop_words: frozenset[str] | None = None


def _get_stop_words() -> frozenset[str]:
    """NLTK's English stop words (downloaded if needed), or the fallback list offline."""
    global _stop_words
    if _stop_words is None:
        try:
            from nltk.corpus import stopwords

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            _stop_words = frozenset(stopwords.words("english"))
        except Exception:
            _stop_words = frozenset(_FALLBACK_STOPWORDS)
    return _stop_words


def has_full_stop_words() -> bool:
    """True if the full NLTK stop-word list is in use rather than the offline fallback."""
    return _get_stop_words() != frozenset(_FALLBACK_STOPWORDS)


class Preprocessor:
    """Cleans and normalizes raw text into a list of stemmed tokens."""

    def __init__(self) -> None:
        from nltk.stem import PorterStemmer

        self._stemmer = PorterStemmer()
        self._stop_words = _get_stop_words()

    def tokens(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[‘’“”]", "", text)  # smart quotes
        text = re.sub(r"[0-9]", "", text)  # digits
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        # preserve_line: the text is one line with punctuation already removed,
        # so no sentence-splitting model (punkt) is needed.
        return [
            self._stemmer.stem(tok)
            for tok in nltk.word_tokenize(text, preserve_line=True)
            if tok and tok not in self._stop_words
        ]

    def clean(self, text: str) -> str:
        """Return a normalized, space-joined string ready for vectorizing."""
        return " ".join(self.tokens(text))

    def fingerprint(self) -> str:
        """Identify the cleaning rules: the stop words plus the preprocessing version."""
        payload = PREPROCESSING_VERSION + "\n" + "\n".join(sorted(self._stop_words))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
