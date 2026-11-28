import csv
import io
from importlib import resources

import pytest

from moodcheck.research.corpus import (
    Dataset,
    UserDataError,
    load_dataset,
    load_depressed_docs,
    load_neutral_docs,
    load_not_depressed_docs,
    read_corpus_text,
    split_sentences,
    user_data_sha256,
)

DATA = resources.files("moodcheck.research") / "data"


def test_corpus_counts():
    assert len(load_depressed_docs()) == 424
    assert len(load_not_depressed_docs()) == 154
    assert len(load_neutral_docs()) == 300


def test_happydb_file_parses_to_one_record_per_moment():
    text = read_corpus_text("not_depressed.txt")
    id_lines = [
        line
        for line in text.splitlines()
        if len(line) > 5 and line[:5].isdigit() and line[5] == ","
    ]
    records = list(csv.reader(io.StringIO(text, newline=None)))
    assert len(records) == len(id_lines) == 154
    assert all(len(r[0]) == 5 and r[0].isdigit() for r in records)
    assert all(len(r) == 9 for r in records)


def test_split_sentences():
    assert split_sentences("One.  Two!\nThree? four") == ["One.", "Two!", "Three?", "four"]
    assert split_sentences("   ") == []


def test_dataset_is_balanced_and_deterministic():
    ds = load_dataset(str.lower)
    assert isinstance(ds, Dataset)
    assert ds.labels.count("depressed") == ds.labels.count("not_depressed")
    assert len(ds.documents) == len(ds.labels) == len(ds.groups) == len(ds.sources)
    assert ds == load_dataset(str.lower)


def test_dataset_unbalanced_has_all_sources_and_counts():
    ds = load_dataset(str.lower, balance=False)
    assert ds.sources.count("reddit_depression") == 424
    assert ds.sources.count("happydb") == 154
    assert ds.sources.count("goemotions_neutral") == 300
    assert set(ds.labels) == {"depressed", "not_depressed"}
    for source, label in zip(ds.sources, ds.labels, strict=True):
        assert label == ("depressed" if source == "reddit_depression" else "not_depressed")


def test_group_ids_are_prefixed_and_namespaced_by_source():
    ds = load_dataset(str.lower, balance=False)
    prefix = {"reddit_depression": "dep:", "happydb": "happy:", "goemotions_neutral": "neutral:"}
    for source, group in zip(ds.sources, ds.groups, strict=True):
        assert group.startswith(prefix[source])
    dep_groups = {g for s, g in zip(ds.sources, ds.groups, strict=True) if s == "reddit_depression"}
    assert len(dep_groups) == 19
    happy = [g for s, g in zip(ds.sources, ds.groups, strict=True) if s == "happydb"]
    neutral = [g for s, g in zip(ds.sources, ds.groups, strict=True) if s == "goemotions_neutral"]
    assert len(set(happy)) == 154
    assert len(set(neutral)) == 300


def test_dataset_drops_documents_that_clean_to_nothing_and_stays_aligned():
    ds = load_dataset(lambda d: "" if "the" in d else d, balance=False)
    assert all("the" not in d for d in ds.documents)
    assert len(ds.documents) == len(ds.labels) == len(ds.groups) == len(ds.sources)


def test_corpus_hash_covers_all_three_files(monkeypatch):
    import moodcheck.research.corpus as corpus

    original = corpus.corpus_sha256()
    assert len(original) == 64
    real = corpus._read_bytes
    monkeypatch.setattr(
        corpus, "_read_bytes", lambda n: real(n) + (b"x" if n == "neutral_goemotions.tsv" else b"")
    )
    assert corpus.corpus_sha256() != original


def test_license_file_ships_with_neutral_corpus():
    text = (DATA / "LICENSE-GoEmotions.txt").read_text(encoding="utf-8")
    assert "neutral_goemotions.tsv" in text
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text


# --- bring your own data -------------------------------------------------


def _write(tmp_path, content: str | bytes, name="user.csv"):
    path = tmp_path / name
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return path


def test_user_csv_loads_only_user_data(tmp_path):
    path = _write(
        tmp_path,
        "text,label,group\nI feel empty,depressed,p1\nI feel fine,not_depressed,p2\n"
        "More sad words,depressed,p1\nNice day,not_depressed,\n",
    )
    ds = load_dataset(str.lower, balance=False, data_path=path)
    # Output is ordered by label (depressed first), input order kept within a label.
    assert ds.documents == ["i feel empty", "more sad words", "i feel fine", "nice day"]
    assert ds.labels == ["depressed", "depressed", "not_depressed", "not_depressed"]
    assert ds.sources == ["user"] * 4
    assert ds.groups[0] == ds.groups[1] == "user:p1"
    assert len(set(ds.groups)) == 3  # the empty group got a unique id of its own


def test_user_csv_without_group_column_gives_unique_groups(tmp_path):
    path = _write(tmp_path, "text,label\na b,depressed\nc d,not_depressed\n")
    ds = load_dataset(str, balance=False, data_path=path)
    assert len(set(ds.groups)) == 2


def test_user_csv_balance_downsamples(tmp_path):
    rows = "".join(f"d{i},depressed\n" for i in range(5)) + "n,not_depressed\n"
    ds = load_dataset(str, data_path=_write(tmp_path, "text,label\n" + rows))
    assert ds.labels.count("depressed") == ds.labels.count("not_depressed") == 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("text,other\na,b\n", "label"),
        ("label,other\ndepressed,b\n", "text"),
        ("", "empty"),
        ("text,label\n", "no rows"),
        ("text,label\na,sad\nb,depressed\n", "sad"),
        ("text,label\na,depressed\nb,depressed\n", "both classes"),
        ("text,label\n,depressed\nb,not_depressed\n", "text"),
        (b"text,label\n\xff\xfe,depressed\n", "UTF-8"),
    ],
)
def test_user_csv_validation_errors(tmp_path, content, message):
    path = _write(tmp_path, content)
    with pytest.raises(UserDataError, match=message):
        load_dataset(str, data_path=path)


def test_user_csv_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(UserDataError, match="not found|cannot read"):
        load_dataset(str, data_path=tmp_path / "nope.csv")


def test_user_data_hash_is_of_the_file_bytes(tmp_path):
    import hashlib

    path = _write(tmp_path, "text,label\na,depressed\nb,not_depressed\n")
    assert user_data_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_goemotions_mask_tokens_are_removed_on_load():
    docs = load_neutral_docs()
    assert not any("[NAME]" in d or "[RELIGION]" in d for d in docs)
    assert len(docs) == 300
