"""Loading the two labelled corpora shipped inside the package.

* ``depressed.txt``     free-text posts by people describing feelings of
  depression, split into sentences (one document each).
* ``not_depressed.txt`` happy-moment entries from the HappyDB dataset, stored
  as CSV; only the cleaned moment text (column 4) is used.
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
from collections.abc import Callable
from importlib import resources

from moodcheck.research.config import DEPRESSED_LABEL, NOT_DEPRESSED_LABEL

RANDOM_SEED = 42

# Index of the cleaned happy-moment text in the HappyDB CSV rows.
HAPPYDB_TEXT_COLUMN = 4

_PACKAGE = "moodcheck.research"
DEPRESSED_FILE = "depressed.txt"
NOT_DEPRESSED_FILE = "not_depressed.txt"


def _read_bytes(name: str) -> bytes:
    return (resources.files(_PACKAGE) / "data" / name).read_bytes()


def read_corpus_text(name: str) -> str:
    return _read_bytes(name).decode("utf-8", errors="ignore")


def corpus_sha256() -> str:
    """Hash of both corpora; changes whenever the training data changes."""
    digest = hashlib.sha256()
    for name in (DEPRESSED_FILE, NOT_DEPRESSED_FILE):
        digest.update(_read_bytes(name))
    return digest.hexdigest()


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences (regex-based, no external data needed)."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def load_depressed_docs() -> list[str]:
    """Depressed corpus: free text -> one document per sentence."""
    return split_sentences(read_corpus_text(DEPRESSED_FILE))


def load_not_depressed_docs() -> list[str]:
    """Not-depressed corpus: HappyDB CSV -> one document per moment text."""
    docs: list[str] = []
    text = read_corpus_text(NOT_DEPRESSED_FILE)
    for row in csv.reader(io.StringIO(text, newline=None)):
        if len(row) > HAPPYDB_TEXT_COLUMN:
            moment = row[HAPPYDB_TEXT_COLUMN].strip()
            if moment:
                docs.append(moment)
    return docs


def load_dataset(clean: Callable[[str], str], balance: bool = True) -> tuple[list[str], list[str]]:
    """Return (documents, labels), with classes balanced by down-sampling."""
    by_label: dict[str, list[str]] = {DEPRESSED_LABEL: [], NOT_DEPRESSED_LABEL: []}
    for raw_docs, label in (
        (load_depressed_docs(), DEPRESSED_LABEL),
        (load_not_depressed_docs(), NOT_DEPRESSED_LABEL),
    ):
        by_label[label] = [c for c in (clean(d) for d in raw_docs) if c]

    if balance:
        smallest = min(len(docs) for docs in by_label.values())
        rng = random.Random(RANDOM_SEED)
        by_label = {label: rng.sample(docs, smallest) for label, docs in by_label.items()}

    documents: list[str] = []
    labels: list[str] = []
    for label, docs in by_label.items():
        documents.extend(docs)
        labels.extend([label] * len(docs))
    return documents, labels
