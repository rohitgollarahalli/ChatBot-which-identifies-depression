"""Rebuild ``neutral_goemotions.tsv``, the neutral-text corpus bundled with moodcheck.

Source: GoEmotions (Demszky et al., 2020; Google Research, Apache-2.0)
https://github.com/google-research/google-research/tree/master/goemotions

Selection (deterministic, so anyone can regenerate the exact bundled file):

1. Download ``data/train.tsv`` and ``data/emotions.txt`` from the GoEmotions
   directory (no header; train.tsv columns are text, comma-separated label ids,
   comment id).
2. Find the id of "neutral" in ``emotions.txt`` (line index = label id).
3. Keep rows whose label list is exactly that one id, whose text has at least
   6 words, and that ``moodcheck.safety.is_crisis`` does not flag.
4. Sort the candidates by comment id and take 300 with ``random.Random(42)``.
5. Write ``id<TAB>text`` with a header. GoEmotions' own ``[NAME]`` and
   ``[RELIGION]`` masking is kept; tabs and newlines inside text become spaces.

Usage::

    python scripts/build_neutral_subset.py WORKDIR [--output PATH] [--offline]

WORKDIR receives the downloaded files (created if missing). ``--offline``
skips downloading and uses files already in WORKDIR. The output defaults to
``src/moodcheck/research/data/neutral_goemotions.tsv``.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import urllib.request
from pathlib import Path

from moodcheck.safety import is_crisis

BASE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
)
SAMPLE_SIZE = 300
MIN_WORDS = 6
SEED = 42
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "moodcheck"
    / "research"
    / "data"
    / "neutral_goemotions.tsv"
)


def neutral_label_id(emotions_text: str) -> str:
    """Label id of "neutral": the line index in emotions.txt."""
    names = [line.strip() for line in emotions_text.splitlines()]
    return str(names.index("neutral"))


def select_neutral(train_tsv: str, neutral_id: str) -> list[tuple[str, str]]:
    """Return ``(comment_id, text)`` candidates sorted by comment id."""
    candidates: dict[str, str] = {}
    for line in train_tsv.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        text, labels, comment_id = parts
        if labels != neutral_id:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) < MIN_WORDS or is_crisis(text):
            continue
        candidates[comment_id] = text
    return sorted(candidates.items())


def sample(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return random.Random(SEED).sample(candidates, SAMPLE_SIZE)


def render(rows: list[tuple[str, str]]) -> str:
    return "id\ttext\n" + "".join(f"{cid}\t{text}\n" for cid, text in rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("workdir", type=Path, help="directory for the downloaded GoEmotions files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="use files already in WORKDIR")
    args = parser.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)
    for name in ("train.tsv", "emotions.txt"):
        target = args.workdir / name
        if not args.offline:
            urllib.request.urlretrieve(BASE_URL + name, target)  # noqa: S310 (fixed https URL)
    emotions = (args.workdir / "emotions.txt").read_text(encoding="utf-8")
    train = (args.workdir / "train.tsv").read_text(encoding="utf-8")

    neutral_id = neutral_label_id(emotions)
    candidates = select_neutral(train, neutral_id)
    rows = sample(candidates)
    args.output.write_text(render(rows), encoding="utf-8", newline="\n")
    print(
        f"neutral id {neutral_id}; {len(candidates)} candidates; wrote {len(rows)} rows to "
        f"{args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
