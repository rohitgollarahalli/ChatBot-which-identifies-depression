"""Split text into the units the classifier votes on."""

from __future__ import annotations

import re

from moodcheck.research.corpus import split_sentences
from moodcheck.storage import TRANSCRIPT_HEADER

_NEWLINE = re.compile(r"\r\n|\r|\n")


def split_units(text: str) -> list[str]:
    """Split text into voting units.

    A transcript saved by ``moodcheck chat --save`` starts with
    ``TRANSCRIPT_HEADER`` and has one message per line: each message is split
    into sentences on its own, so a message without final punctuation stays a
    unit. Any other text is treated as prose, where a line break may just be
    wrapping, so it is split on sentence punctuation only.
    """
    lines = _NEWLINE.split(text)
    if lines and lines[0].strip() == TRANSCRIPT_HEADER:
        units: list[str] = []
        for line in lines[1:]:
            units.extend(split_sentences(line))
        return units
    return split_sentences(text)
