"""Split text into the units the classifier votes on."""

from __future__ import annotations

from moodcheck.research.corpus import split_sentences


def split_units(text: str) -> list[str]:
    """Split on lines first (one chat message per line), then into sentences.

    A message without final punctuation stays its own unit instead of merging
    with the next one. A single-line text (an old transcript, a pasted
    paragraph) is split on sentence punctuation only.
    """
    units: list[str] = []
    for line in text.splitlines():
        units.extend(split_sentences(line))
    return units
