import random

import pytest

from moodcheck.conversation.eliza import ElizaEngine
from moodcheck.conversation.engine import Reply
from moodcheck.conversation.session import EXIT_COMMAND, MIN_WORDS, run
from moodcheck.research.classifier import Prediction, verdict_message
from moodcheck.research.config import DEPRESSED_LABEL, NOT_DEPRESSED_LABEL
from moodcheck.safety import CRISIS_MESSAGE, CRISIS_REMINDER, DISCLAIMER, RESEARCH_NOTE
from moodcheck.storage import TRANSCRIPT_HEADER, TranscriptSaveError

LONG = " ".join(["word"] * MIN_WORDS)


def _session(lines, save_path=None):
    out = []
    feed = iter(lines)

    def read(_prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    run(ElizaEngine(random.Random(0)), read=read, write=out.append, save_path=save_path)
    return out


def test_full_crisis_message_once_then_reminder():
    out = _session(["I want to die", "I'm not suicidal, just tired"])
    assert out.count(CRISIS_MESSAGE) == 1
    assert CRISIS_REMINDER in out


class _Recorder(ElizaEngine):
    """Records what reaches the engine, to prove the session really ended."""

    def __init__(self):
        super().__init__(random.Random(0))
        self.seen = []

    def _generate(self, message):
        self.seen.append(message)
        return super()._generate(message)


@pytest.mark.parametrize("save", [True, False])
def test_user_in_crisis_can_leave_without_minimum_words(tmp_path, save):
    path = tmp_path / "out.txt"
    engine = _Recorder()
    unread = iter(["I want to kill myself", EXIT_COMMAND, "still here?"])
    out = []
    run(engine, read=lambda _p: next(unread), write=out.append, save_path=path if save else None)
    assert engine.seen == []
    assert list(unread) == ["still here?"]  # the session never asked for more
    assert not any("chat a little more" in line for line in out)
    assert out[-1] == CRISIS_REMINDER
    assert not path.exists()


def test_session_without_crisis_has_no_reminder():
    out = _session(["I had a long day at work"])
    assert CRISIS_REMINDER not in out
    assert CRISIS_MESSAGE not in out


def test_without_save_goodbye_exits_at_once_and_writes_nothing(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    out = _session(["hi", EXIT_COMMAND, "never read"])
    assert any("Nothing you type is saved" in line for line in out)
    assert not any("saved to" in line for line in out)
    assert list(work.iterdir()) == []


def test_with_save_goodbye_needs_enough_words(tmp_path):
    path = tmp_path / "t.txt"
    out = _session(["hi", EXIT_COMMAND, LONG, EXIT_COMMAND], save_path=path)
    assert any("chat a little more" in line for line in out)
    assert path.read_text(encoding="utf-8") == f"{TRANSCRIPT_HEADER}\nhi\n{LONG}\n"
    assert any("were saved to" in line for line in out)


def test_eof_saves_a_long_enough_conversation(tmp_path):
    path = tmp_path / "t.txt"
    out = _session([LONG], save_path=path)
    assert path.read_text(encoding="utf-8") == f"{TRANSCRIPT_HEADER}\n{LONG}\n"
    assert any("were saved to" in line for line in out)


def test_eof_with_too_few_words_says_nothing_was_saved(tmp_path):
    path = tmp_path / "t.txt"
    out = _session(["hi"], save_path=path)
    assert not path.exists()
    assert any("Nothing was saved" in line for line in out)


def test_ctrl_c_discards_and_says_so(tmp_path):
    path = tmp_path / "t.txt"
    out = []
    feed = iter([LONG])

    def read(_prompt):
        try:
            return next(feed)
        except StopIteration:
            raise KeyboardInterrupt from None

    run(ElizaEngine(), read=read, write=out.append, save_path=path)
    assert not path.exists()
    assert any("Nothing was saved" in line for line in out)


def test_crisis_exit_with_too_few_words_says_nothing_was_saved(tmp_path):
    path = tmp_path / "t.txt"
    out = _session(["I want to kill myself", EXIT_COMMAND], save_path=path)
    assert any("Nothing was saved" in line for line in out)
    assert out[-1] == CRISIS_REMINDER


def test_opening_line_states_when_saving_happens(tmp_path):
    out = _session([], save_path=tmp_path / "t.txt")
    assert any(str(MIN_WORDS) in line and "t.txt" in line for line in out)


def test_session_screens_even_if_an_engine_skips_the_screen():
    engine = ElizaEngine()
    seen = []
    engine.respond = lambda message: seen.append(message) or Reply("bypassed")  # type: ignore
    out = []
    feed = iter(["I want to die", EXIT_COMMAND])
    run(engine, read=lambda _p: next(feed), write=out.append)
    assert CRISIS_MESSAGE in out
    assert seen == []


def test_undecodable_characters_do_not_hide_crisis_language():
    out = _session(["I don\ufffdt want to live"])
    assert CRISIS_MESSAGE in out


def test_save_error_still_shows_crisis_reminder(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("existing")
    out = []
    feed = iter(["I want to die", LONG, EXIT_COMMAND])
    try:
        run(
            ElizaEngine(),
            read=lambda _p: next(feed),
            write=out.append,
            save_path=path,
        )
    except TranscriptSaveError:
        pass
    else:
        raise AssertionError("expected TranscriptSaveError")
    assert out[-1] == CRISIS_REMINDER
    assert path.read_text() == "existing"


def test_saved_transcript_has_one_message_per_line(tmp_path):
    path = tmp_path / "t.txt"
    _session(["my day was fine", "no\u2028punctuation here", LONG], save_path=path)
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[:3] == [TRANSCRIPT_HEADER, "my day was fine", "no punctuation here"]
    assert lines[3] == LONG
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_crisis_transcript_overrides_model_guess():
    text = "Work is fine. I want to kill myself. The weather is nice."
    assert verdict_message(Prediction(NOT_DEPRESSED_LABEL, 3, 3), text) == CRISIS_MESSAGE


def test_crisis_phrase_split_across_a_line_break_is_still_caught():
    text = "Work was fine\nI want to\ndie\n"
    assert verdict_message(Prediction(NOT_DEPRESSED_LABEL, 3, 3), text) == CRISIS_MESSAGE


def test_verdict_includes_disclaimer_and_vote():
    msg = verdict_message(Prediction(DEPRESSED_LABEL, 4, 6), "I had a long day at work.")
    assert msg.startswith(
        "The model's guess: the transcript shows signs of depression (4 of 6 parts)."
    )
    assert DISCLAIMER in msg
    assert RESEARCH_NOTE in msg


def test_verdict_explains_a_tie():
    msg = verdict_message(Prediction(NOT_DEPRESSED_LABEL, 2, 4), "a. b. c. d.")
    assert "(2 of 4 parts; a tie counts as no)" in msg


def test_verdict_for_not_depressed_and_singular():
    msg = verdict_message(Prediction(NOT_DEPRESSED_LABEL, 0, 1), "Fine day.")
    assert msg.startswith(
        "The model's guess: the transcript does not show signs of depression (0 of 1 part)."
    )


def test_verdict_without_votes_omits_the_count():
    msg = verdict_message(Prediction(NOT_DEPRESSED_LABEL, 0, 0), "the of and")
    assert msg.startswith("The model's guess: the transcript does not show signs of depression.")
