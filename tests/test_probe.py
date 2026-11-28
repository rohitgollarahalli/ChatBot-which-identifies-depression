from moodcheck.research.probe import load_probe_sentences
from moodcheck.safety import is_crisis


def test_probe_has_sixty_unique_sentences():
    sentences = load_probe_sentences()
    assert len(sentences) == 60
    assert len(set(sentences)) == 60


def test_probe_sentences_are_not_crisis_language():
    assert [s for s in load_probe_sentences() if is_crisis(s)] == []


def test_probe_sentences_are_five_to_twenty_five_words():
    assert [s for s in load_probe_sentences() if not 5 <= len(s.split()) <= 25] == []


def test_probe_skips_comments_and_blanks():
    assert not any(s.startswith("#") or not s.strip() for s in load_probe_sentences())
