"""The data card ships with the corpora and stays in step with them."""

import hashlib
import re
from importlib import resources
from pathlib import Path

from moodcheck.research.corpus import load_depressed_docs, load_not_depressed_docs

DATA = resources.files("moodcheck.research") / "data"
CORPORA = ("depressed.txt", "not_depressed.txt")


def _card() -> str:
    return (DATA / "README.md").read_text(encoding="utf-8")


def test_data_card_sits_next_to_the_corpora():
    # Wheel contents are checked in CI's package job.
    assert (DATA / "README.md").is_file()


def test_data_card_records_the_current_hash_of_each_corpus():
    card = _card()
    for name in CORPORA:
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        recorded = re.search(rf"^\| `{re.escape(name)}` \| `([0-9a-f]{{64}})` \|$", card, re.M)
        assert recorded, f"data card has no SHA-256 line for {name}"
        assert recorded.group(1) == actual, f"{name} changed; update the data card"


def test_data_card_document_counts_match_the_loader():
    row = f"| Documents after loading | {len(load_depressed_docs())} ("
    assert row in _card()
    assert f"| {len(load_not_depressed_docs())} (one per happy moment" in _card()


def test_project_readme_links_the_data_card():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    assert "src/moodcheck/research/data/README.md" in readme.read_text(encoding="utf-8")
