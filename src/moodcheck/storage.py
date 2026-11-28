"""Opt-in local transcript saving."""

from __future__ import annotations

import os
from pathlib import Path

# First line of a transcript saved by `moodcheck chat --save`. It tells
# `research predict` that each following line is one message.
TRANSCRIPT_HEADER = "# moodcheck transcript: one message per line"


class TranscriptSaveError(Exception):
    """The transcript could not be saved; no new file was left behind."""


def save_transcript(path: Path, text: str) -> None:
    """Write ``text`` to a new file readable only by the current user.

    Refuses to overwrite an existing ``path``. Raises ``TranscriptSaveError`` on
    any failure, after removing a partly written file.
    """
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise TranscriptSaveError(exc.strerror or str(exc)) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
    except (OSError, UnicodeError) as exc:
        path.unlink(missing_ok=True)
        raise TranscriptSaveError(str(exc)) from exc
