"""The neutral probe set: everyday sentences that say nothing about mood.

Used to see how the classifier treats ordinary text. The sentences were written
for this project and are MIT-licensed like the code.
"""

from __future__ import annotations

from importlib import resources


def load_probe_sentences() -> list[str]:
    """Return the probe sentences, skipping ``#`` comment lines and blank lines."""
    path = resources.files("moodcheck.research") / "data" / "probe_neutral.txt"
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]
