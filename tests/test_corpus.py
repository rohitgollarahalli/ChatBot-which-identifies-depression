from moodcheck.research.corpus import (
    corpus_sha256,
    load_dataset,
    load_depressed_docs,
    load_not_depressed_docs,
    split_sentences,
)


def test_corpus_counts():
    assert len(load_depressed_docs()) == 423
    assert len(load_not_depressed_docs()) == 150


def test_split_sentences():
    assert split_sentences("One.  Two!\nThree? four") == ["One.", "Two!", "Three?", "four"]
    assert split_sentences("   ") == []


def test_dataset_is_balanced_and_deterministic():
    docs, labels = load_dataset(str.lower)
    assert labels.count("depressed") == labels.count("not_depressed")
    assert (docs, labels) == load_dataset(str.lower)


def test_dataset_drops_documents_that_clean_to_nothing():
    docs, _ = load_dataset(lambda d: "" if "the" in d else d, balance=False)
    assert all("the" not in d for d in docs)


def test_corpus_hash_is_stable():
    assert corpus_sha256() == corpus_sha256()
    assert len(corpus_sha256()) == 64
