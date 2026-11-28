"""Loading the labelled corpora shipped inside the package, or a user's own CSV.

* ``depressed.txt``     free-text posts by people describing feelings of
  depression, split into sentences (one document each). Blank lines separate
  blocks; each block is one leakage group.
* ``not_depressed.txt`` happy-moment entries from the HappyDB dataset, stored
  as CSV; only the cleaned moment text (column 4) is used.
* ``neutral_goemotions.tsv`` neutral Reddit comments from GoEmotions (a filtered
  sample, see ``scripts/build_neutral_subset.py``), also labelled
  ``not_depressed``.

Researchers with licensed data can train on their own CSV instead
(``load_dataset(..., data_path=...)``); nothing from it is bundled or stored.
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from moodcheck.research.config import DEPRESSED_LABEL, NOT_DEPRESSED_LABEL

RANDOM_SEED = 42

# Index of the cleaned happy-moment text in the HappyDB CSV rows.
HAPPYDB_TEXT_COLUMN = 4

_PACKAGE = "moodcheck.research"
DEPRESSED_FILE = "depressed.txt"
NOT_DEPRESSED_FILE = "not_depressed.txt"
NEUTRAL_FILE = "neutral_goemotions.tsv"
BUNDLED_FILES = (DEPRESSED_FILE, NOT_DEPRESSED_FILE, NEUTRAL_FILE)

SOURCE_REDDIT = "reddit_depression"
SOURCE_HAPPYDB = "happydb"
SOURCE_NEUTRAL = "goemotions_neutral"
SOURCE_USER = "user"


class UserDataError(ValueError):
    """The user-supplied CSV is missing, unreadable or invalid."""


@dataclass(frozen=True)
class Dataset:
    """Aligned lists: one entry per document."""

    documents: list[str]
    labels: list[str]
    groups: list[str]  # documents sharing a group must not be split across train and test
    sources: list[str]  # reddit_depression | happydb | goemotions_neutral | user


def _read_bytes(name: str) -> bytes:
    return (resources.files(_PACKAGE) / "data" / name).read_bytes()


def read_corpus_text(name: str) -> str:
    return _read_bytes(name).decode("utf-8", errors="ignore")


def corpus_sha256() -> str:
    """Hash of the bundled corpora; changes whenever the training data changes."""
    digest = hashlib.sha256()
    for name in BUNDLED_FILES:
        digest.update(_read_bytes(name))
    return digest.hexdigest()


def user_data_sha256(path: Path) -> str:
    """Hash of a user's data file (its raw bytes), for model versioning."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise UserDataError(f"cannot read {path}: {exc.strerror or exc}") from exc


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences (regex-based, no external data needed)."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _depressed_records() -> list[tuple[str, str]]:
    """(group, sentence) pairs; each blank-line-separated block is one group."""
    text = read_corpus_text(DEPRESSED_FILE)
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [
        (f"dep:{i}", sentence)
        for i, block in enumerate(blocks)
        for sentence in split_sentences(block)
    ]


def _happydb_records() -> list[tuple[str, str]]:
    """(group, moment text) pairs from the HappyDB CSV; one group per moment id."""
    records: list[tuple[str, str]] = []
    text = read_corpus_text(NOT_DEPRESSED_FILE)
    for row in csv.reader(io.StringIO(text, newline=None)):
        if len(row) > HAPPYDB_TEXT_COLUMN:
            moment = row[HAPPYDB_TEXT_COLUMN].strip()
            if moment:
                records.append((f"happy:{row[0].strip()}", moment))
    return records


def _neutral_records() -> list[tuple[str, str]]:
    """(group, comment text) pairs from the GoEmotions subset (header: id, text)."""
    lines = read_corpus_text(NEUTRAL_FILE).splitlines()
    records: list[tuple[str, str]] = []
    for line in lines[1:]:
        comment_id, _, comment = line.partition("\t")
        if comment.strip():
            records.append((f"neutral:{comment_id}", comment.strip()))
    return records


def load_depressed_docs() -> list[str]:
    """Depressed corpus: free text -> one document per sentence."""
    return [doc for _, doc in _depressed_records()]


def load_not_depressed_docs() -> list[str]:
    """HappyDB happy moments: CSV -> one document per moment text."""
    return [doc for _, doc in _happydb_records()]


def load_neutral_docs() -> list[str]:
    """GoEmotions neutral comments: one document per comment."""
    return [doc for _, doc in _neutral_records()]


def _user_records(path: Path) -> list[tuple[str, str, str]]:
    """(group, text, label) rows from a user CSV; raises UserDataError if invalid."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise UserDataError(f"data file not found: {path}") from None
    except OSError as exc:
        raise UserDataError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UserDataError(
            f"{path} is not valid UTF-8 ({exc.reason} at byte {exc.start})"
        ) from exc
    if not content.strip():
        raise UserDataError(f"{path} is empty; expected a header 'text,label[,group]' and rows")

    reader = csv.DictReader(io.StringIO(content, newline=None))
    fields = reader.fieldnames or []
    missing = [c for c in ("text", "label") if c not in fields]
    if missing:
        raise UserDataError(
            f"{path} is missing required column(s): {', '.join(missing)} (found: {fields})"
        )

    rows: list[tuple[str, str, str]] = []
    for number, row in enumerate(reader, start=2):  # line 1 is the header
        text = (row.get("text") or "").strip()
        label = (row.get("label") or "").strip()
        if label not in (DEPRESSED_LABEL, NOT_DEPRESSED_LABEL):
            raise UserDataError(
                f"{path} row {number}: unknown label {label!r}; "
                f"expected exactly {DEPRESSED_LABEL!r} or {NOT_DEPRESSED_LABEL!r}"
            )
        if not text:
            raise UserDataError(f"{path} row {number}: text is empty")
        group = (row.get("group") or "").strip() or f"row{number}"
        rows.append((f"user:{group}", text, label))
    if not rows:
        raise UserDataError(f"{path} has a header but no rows")
    if {label for _, _, label in rows} != {DEPRESSED_LABEL, NOT_DEPRESSED_LABEL}:
        raise UserDataError(
            f"{path} must contain both classes ({DEPRESSED_LABEL!r} and {NOT_DEPRESSED_LABEL!r})"
        )
    return rows


def _bundled_rows() -> list[tuple[str, str, str, str]]:
    """(group, text, label, source) for every bundled document."""
    rows: list[tuple[str, str, str, str]] = []
    for records, label, source in (
        (_depressed_records(), DEPRESSED_LABEL, SOURCE_REDDIT),
        (_happydb_records(), NOT_DEPRESSED_LABEL, SOURCE_HAPPYDB),
        (_neutral_records(), NOT_DEPRESSED_LABEL, SOURCE_NEUTRAL),
    ):
        rows.extend((group, text, label, source) for group, text in records)
    return rows


def load_dataset(
    clean: Callable[[str], str], balance: bool = True, data_path: Path | None = None
) -> Dataset:
    """Load, clean and (by default) class-balance a dataset.

    With ``data_path=None`` the three bundled corpora are used; otherwise only
    the user's CSV (columns ``text``, ``label``, optional ``group``). Documents
    that clean to an empty string are dropped. Balancing down-samples the
    larger class with a fixed seed, so the result is deterministic.
    """
    if data_path is None:
        rows = _bundled_rows()
    else:
        rows = [(g, t, lab, SOURCE_USER) for g, t, lab in _user_records(data_path)]

    by_label: dict[str, list[tuple[str, str, str]]] = {DEPRESSED_LABEL: [], NOT_DEPRESSED_LABEL: []}
    for group, text, label, source in rows:
        cleaned = clean(text)
        if cleaned:
            by_label[label].append((cleaned, group, source))

    if balance:
        smallest = min(len(items) for items in by_label.values())
        rng = random.Random(RANDOM_SEED)
        by_label = {label: rng.sample(items, smallest) for label, items in by_label.items()}

    documents: list[str] = []
    labels: list[str] = []
    groups: list[str] = []
    sources: list[str] = []
    for label, items in by_label.items():
        for text, group, source in items:
            documents.append(text)
            labels.append(label)
            groups.append(group)
            sources.append(source)
    return Dataset(documents, labels, groups, sources)
