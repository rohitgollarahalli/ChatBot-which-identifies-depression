import nltk
import pytest

from moodcheck.research import preprocessing
from moodcheck.research.preprocessing import Preprocessor


@pytest.fixture(autouse=True)
def no_punkt(monkeypatch):
    def guarded(name, *args, **kwargs):
        assert name != "punkt", "punkt must not be needed"
        raise OSError("no network in tests")

    monkeypatch.setattr(nltk, "download", guarded)


def test_tokens_are_lowercased_stemmed_and_filtered():
    pre = Preprocessor()
    assert pre.tokens("I was Running, and 123 dogs' bones!") == ["run", "dog", "bone"]
    assert pre.clean("  Feeling   SAD ") == "feel sad"
    assert pre.clean("") == ""


def test_smart_quotes_removed():
    assert Preprocessor().clean("“sadness” isn’t") == Preprocessor().clean("sadness isnt")


def test_no_punkt_lookup(monkeypatch):
    real_find = nltk.data.find

    def find(resource, *args, **kwargs):
        assert "punkt" not in resource
        return real_find(resource, *args, **kwargs)

    monkeypatch.setattr(nltk.data, "find", find)
    monkeypatch.setattr(preprocessing, "_stop_words", None)
    assert Preprocessor().tokens("hello world again")


def test_offline_fallback_stop_words(monkeypatch):
    monkeypatch.setattr(preprocessing, "_stop_words", None)
    monkeypatch.setattr(nltk.data, "find", lambda *a, **k: (_ for _ in ()).throw(LookupError()))
    pre = Preprocessor()
    assert not preprocessing.has_full_stop_words()
    assert pre.clean("the cat") == "cat"


def test_fingerprint_changes_with_stop_words_and_version(monkeypatch):
    base = Preprocessor().fingerprint()
    assert base == Preprocessor().fingerprint()
    monkeypatch.setattr(preprocessing, "PREPROCESSING_VERSION", "other")
    assert Preprocessor().fingerprint() != base
    monkeypatch.undo()
    monkeypatch.setattr(preprocessing, "_stop_words", frozenset({"zzz"}))
    assert Preprocessor().fingerprint() != base
