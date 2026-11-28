"""The interactive chat loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from moodcheck.conversation.engine import Engine, Reply
from moodcheck.safety import CRISIS_MESSAGE, CRISIS_REMINDER, DISCLAIMER, is_crisis
from moodcheck.storage import TRANSCRIPT_HEADER, save_transcript

# What an undecodable byte becomes when input is read with errors="replace".
_UNREADABLE = "\ufffd"

# When saving, wait for this many words so the transcript has enough text to be useful.
MIN_WORDS = 31
EXIT_COMMAND = "good-bye"


def _needs_crisis_reply(message: str) -> bool:
    """Screen a message, also trying the likely readings of undecodable characters."""
    readings = {message, message.replace(_UNREADABLE, ""), message.replace(_UNREADABLE, "'")}
    return any(is_crisis(reading) for reading in readings)


def _enough_words(transcript: list[str]) -> bool:
    return len(" ".join(transcript).split()) >= MIN_WORDS


def _finish(save_path: Path, transcript: list[str], write: Callable[[str], None]) -> None:
    if not _enough_words(transcript):
        write(f"Nothing was saved (fewer than {MIN_WORDS} words).")
        return
    # A header, then one message per line, so prediction can tell messages apart.
    # Other line-separator characters inside a message become spaces.
    lines = [" ".join(message.split()) for message in transcript]
    save_transcript(save_path, "\n".join([TRANSCRIPT_HEADER, *lines]) + "\n")
    write(f"\nThank you. Your responses were saved to {save_path}.")


def run(
    engine: Engine,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    save_path: Path | None = None,
) -> None:
    """Chat until the user types ``good-bye``, input ends, or they press Ctrl-C.

    Nothing is written to disk unless ``save_path`` is given. With it, the
    transcript is saved on ``good-bye`` or end of input once there are
    ``MIN_WORDS`` words; Ctrl-C discards it. An exit that saves nothing says so.
    Raises
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
        write(
            f"When you finish, your messages will be saved to {save_path} "
            f"if there are at least {MIN_WORDS} words. Press Ctrl-C to leave without saving."
        )
    write(f"When you'd like to finish, just type '{EXIT_COMMAND}'.")
    write("\nHello. How are you feeling today?")

    try:
        while True:
            try:
                message = read("> ").strip()
            except KeyboardInterrupt:
                write("")
                if save_path is not None:
                    write("Nothing was saved.")
                break
            except EOFError:
                write("")
                if save_path is not None:
                    _finish(save_path, transcript, write)
                break

            if message.lower() == EXIT_COMMAND:
                if save_path is None:
                    break
                # Never hold someone in the conversation after a crisis message.
                if crisis_seen or _enough_words(transcript):
                    _finish(save_path, transcript, write)
                    break
                write("Let's chat a little more so we can help. Tell me more about your problem.")
                continue

            transcript.append(message)
            # Screen here too, before the engine sees the message, so the screen
            # does not depend on how an engine is written.
            reply = (
                Reply(CRISIS_MESSAGE, crisis=True)
                if _needs_crisis_reply(message)
                else engine.respond(message)
            )
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
