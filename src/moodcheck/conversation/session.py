"""The interactive chat loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from moodcheck.conversation.engine import Engine
from moodcheck.safety import CRISIS_MESSAGE, CRISIS_REMINDER, DISCLAIMER
from moodcheck.storage import save_transcript

# When saving, wait for this many words so the transcript has enough text to be useful.
MIN_WORDS = 31
EXIT_COMMAND = "good-bye"


def run(
    engine: Engine,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    save_path: Path | None = None,
) -> None:
    """Chat until the user types ``good-bye`` (or input ends).

    Nothing is written to disk unless ``save_path`` is given. Raises
    ``TranscriptSaveError`` if saving fails, after showing the crisis reminder if one is due.
    """
    transcript: list[str] = []
    crisis_seen = False
    write(DISCLAIMER + "\n")
    write("Hi! Please answer in full sentences rather than one or two words.")
    write("We're happy to help. :)")
    if save_path is None:
        write("Nothing you type is saved.")
    else:
        write(f"Your messages will be saved to {save_path} when you finish.")
    write(f"When you'd like to finish, just type '{EXIT_COMMAND}'.")
    write("\nHello. How are you feeling today?")

    try:
        while True:
            try:
                message = read("> ").strip()
            except (EOFError, KeyboardInterrupt):
                write("")
                break

            if message.lower() == EXIT_COMMAND:
                if save_path is None:
                    break
                if len(" ".join(transcript).split()) >= MIN_WORDS:
                    save_transcript(save_path, " ".join(transcript))
                    write(f"\nThank you. Your responses were saved to {save_path}.")
                    break
                if crisis_seen:
                    # Never hold someone in the conversation after a crisis message.
                    break
                write("Let's chat a little more so we can help. Tell me more about your problem.")
                continue

            transcript.append(message)
            reply = engine.respond(message)
            text = reply.text
            if reply.crisis:
                # Full message once; a short reminder after, so reassurances
                # ("I'm not suicidal") don't get the same wall of text.
                text = CRISIS_REMINDER if crisis_seen else CRISIS_MESSAGE
                crisis_seen = True
            write(text)
    finally:
        if crisis_seen:
            write(CRISIS_REMINDER)
