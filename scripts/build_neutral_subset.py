"""Rebuild ``neutral_goemotions.tsv``, the neutral-text corpus bundled with moodcheck.

Source: GoEmotions (Demszky et al., 2020; Google Research, Apache-2.0)
https://github.com/google-research/google-research/tree/2adf640a14f11025ae5a9d0ec493b78530d276d3/goemotions

Selection (deterministic, so anyone can regenerate the exact bundled file):

1. Download ``data/train.tsv`` and ``data/emotions.txt`` from the GoEmotions
   directory at a pinned google-research commit and verify their SHA-256 (the
   script fails on a mismatch; no header; train.tsv columns are text, comma-separated label ids,
   comment id).
2. Find the id of "neutral" in ``emotions.txt`` (line index = label id).
3. Keep rows whose label list is exactly that one id, whose text has at least
   6 words (mask tokens such as ``[NAME]`` count as words), and that
   ``moodcheck.safety.is_crisis`` does not flag. The selection therefore also
   depends on the current ``is_crisis`` patterns: changing them can change the sample.
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
import hashlib
import random
import re
import sys
import urllib.request
from pathlib import Path

from moodcheck.safety import is_crisis

# Pinned: latest google-research commit touching goemotions/data/train.tsv at the time of writing.
GOEMOTIONS_COMMIT = "2adf640a14f11025ae5a9d0ec493b78530d276d3"
BASE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    f"{GOEMOTIONS_COMMIT}/goemotions/data/"
)
EXPECTED_SHA256 = {
    "train.tsv": "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5",
    "emotions.txt": "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
}
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


def download(name: str, workdir: Path) -> None:
    """Fetch one file at the pinned commit; verify its hash; then move it into place."""
    target = workdir / name
    partial = workdir / (name + ".part")
    with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:  # noqa: S310 (fixed https URL)
        partial.write_bytes(response.read())
    verify(name, partial)
    partial.replace(target)


def verify(name: str, path: Path) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256[name]:
        raise SystemExit(
            f"{name}: SHA-256 {actual} does not match the pinned {EXPECTED_SHA256[name]}"
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
        if args.offline:
            verify(name, args.workdir / name)
        else:
            download(name, args.workdir)
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
