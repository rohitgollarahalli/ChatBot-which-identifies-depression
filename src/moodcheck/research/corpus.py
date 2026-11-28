"""Loading the labelled corpora shipped inside the package, or a user's own CSV.

* ``depressed.txt``     free-text posts by people describing feelings of
  depression, split into sentences (one document each). Blank lines separate
  blocks; each block is one leakage group.
* ``not_depressed.txt`` happy-moment entries from the HappyDB dataset, stored
  as CSV; only the cleaned moment text (column 4) is used.
* ``neutral_goemotions.tsv`` neutral Reddit comments from GoEmotions (a filtered
  sample, see ``scripts/build_neutral_subset.py``), also labelled
  ``not_depressed``.

``load_dataset(..., data_path=...)`` can read a researcher's own CSV instead;
nothing from it is bundled or stored. (Training and evaluation do not use that
option or the leakage groups yet.)
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
        data = _read_bytes(name)
        # Frame each file (name, length) so moving bytes between files changes the hash.
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
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


# GoEmotions replaces names and religions with these tokens. No other corpus has
# them, so a model could learn that the word "name" means not_depressed; replace
# them with ordinary words on load (the bundled file itself stays as GoEmotions has it).
_GOEMOTIONS_MASKS = (("[NAME]", "someone"), ("[RELIGION]", "religion"))


def _neutral_records() -> list[tuple[str, str]]:
    """(group, comment text) pairs from the GoEmotions subset (header: id, text)."""
    lines = read_corpus_text(NEUTRAL_FILE).splitlines()
    records: list[tuple[str, str]] = []
    for line in lines[1:]:
        comment_id, _, comment = line.partition("\t")
        for mask, word in _GOEMOTIONS_MASKS:
            comment = comment.replace(mask, word)
        comment = comment.strip()
        if comment:
            records.append((f"neutral:{comment_id}", comment))
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


_REQUIRED_HEADER = "text,label[,group]"
_ECHOABLE_LABEL = re.compile(r"^[A-Za-z_]{1,20}$")


def _user_records(path: Path) -> list[tuple[str, str, str]]:
    """(group, text, label) rows from a user CSV; raises UserDataError if invalid.

    The file is meant for gated clinical data, so no error message contains cell
    or header content (only a label value that looks like a label, and line numbers).
    Header names are matched after ``strip().lower()``; label values are stripped
    but otherwise matched exactly. Line numbers are the physical line on which the
    record ends.
    """
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
        raise UserDataError(f"{path} is empty; expected a header '{_REQUIRED_HEADER}' and rows")

    reader = csv.reader(io.StringIO(content, newline=None))
    try:
        header = [h.strip().lower() for h in next(reader)]
    except csv.Error:
        raise UserDataError(f"{path}: the header line could not be parsed as CSV") from None
    if len(set(header)) != len(header):
        raise UserDataError(f"{path}: the header has duplicate column names")
    if "text" not in header or "label" not in header:
        raise UserDataError(
            f"{path}: expected the header '{_REQUIRED_HEADER}' but found {len(header)} column(s) "
            "that do not include both 'text' and 'label'"
        )
    text_at, label_at = header.index("text"), header.index("label")
    group_at = header.index("group") if "group" in header else None

    rows: list[tuple[str, str, str]] = []
    while True:
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error:
            raise UserDataError(
                f"{path} line {reader.line_num}: the row could not be parsed as CSV "
                "(for example a field that is too large)"
            ) from None
        line = reader.line_num
        if not row:  # blank line
            continue
        if len(row) != len(header):
            raise UserDataError(f"{path} line {line} has the wrong number of columns")
        text = row[text_at].strip()
        label = row[label_at].strip()
        if label not in (DEPRESSED_LABEL, NOT_DEPRESSED_LABEL):
            shown = (
                repr(label)
                if _ECHOABLE_LABEL.match(label)
                else "an unrecognised label value (not shown)"
            )
            raise UserDataError(
                f"{path} line {line}: unknown label {shown}; "
                f"expected exactly {DEPRESSED_LABEL!r} or {NOT_DEPRESSED_LABEL!r}"
            )
        if not text:
            raise UserDataError(f"{path} line {line}: text is empty")
        explicit = row[group_at].strip() if group_at is not None else ""
        group = f"user:{explicit}" if explicit else f"user-auto:{line}"
        rows.append((group, text, label))
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
    clean: Callable[[str], str], balance: bool = True, data_path: Path | str | None = None
) -> Dataset:
    """Load, clean and (by default) class-balance a dataset.

    With ``data_path=None`` the three bundled corpora are used; otherwise only
    the user's CSV (columns ``text``, ``label``, optional ``group``). Documents
    that clean to an empty string are dropped, as are exact duplicates within a
    label after cleaning (the first occurrence is kept) and any cleaned text that
    appears under both labels (all copies). Raises ``UserDataError`` (user data)
    or ``ValueError`` (bundled data) if a class ends up with no documents.
    Balancing down-samples the larger class with a fixed seed, so the result is
    deterministic.
    """
    user = data_path is not None
    if data_path is None:
        rows = _bundled_rows()
    else:
        rows = [(g, t, lab, SOURCE_USER) for g, t, lab in _user_records(Path(data_path))]

    by_label: dict[str, dict[str, tuple[str, str]]] = {
        DEPRESSED_LABEL: {},
        NOT_DEPRESSED_LABEL: {},
    }
    for group, text, label, source in rows:
        cleaned = clean(text)
        if cleaned:
            by_label[label].setdefault(cleaned, (group, source))
    ambiguous = by_label[DEPRESSED_LABEL].keys() & by_label[NOT_DEPRESSED_LABEL].keys()
    items_by_label: dict[str, list[tuple[str, str, str]]] = {
        label: [(t, g, s) for t, (g, s) in docs.items() if t not in ambiguous]
        for label, docs in by_label.items()
    }
    if not all(items_by_label.values()):
        message = "a class has no documents left after cleaning and de-duplication"
        raise UserDataError(message) if user else ValueError(message)

    if balance:
        smallest = min(len(items) for items in items_by_label.values())
        rng = random.Random(RANDOM_SEED)
        items_by_label = {
            label: rng.sample(items, smallest) for label, items in items_by_label.items()
        }

    documents: list[str] = []
    labels: list[str] = []
    groups: list[str] = []
    sources: list[str] = []
    for label, items in items_by_label.items():
        for text, group, source in items:
            documents.append(text)
            labels.append(label)
            groups.append(group)
            sources.append(source)
    return Dataset(documents, labels, groups, sources)
