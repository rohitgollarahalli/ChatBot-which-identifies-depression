from src.chatbot import EXIT_COMMAND, run
from src.classifier import DEPRESSED_LABEL, NOT_DEPRESSED_LABEL, verdict_message
from src.safety import CRISIS_MESSAGE, CRISIS_REMINDER, DISCLAIMER


def _session(lines, tmp_path):
    out = []
    feed = iter(lines)

    def read(_prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    run(tmp_path / "out.txt", read=read, write=out.append)
    return out


def test_full_crisis_message_once_then_reminder(tmp_path):
    out = _session(["I want to die", "I'm not suicidal, just tired"], tmp_path)
    assert out.count(CRISIS_MESSAGE) == 1
    assert CRISIS_REMINDER in out


def test_user_in_crisis_can_leave_without_minimum_words(tmp_path):
    out = _session(["I want to kill myself", EXIT_COMMAND, "still here?"], tmp_path)
    assert "still here?" not in " ".join(out)
    assert out[-1] == CRISIS_REMINDER
    assert not (tmp_path / "out.txt").exists()


def test_session_without_crisis_has_no_reminder(tmp_path):
    out = _session(["I had a long day at work"], tmp_path)
    assert CRISIS_REMINDER not in out
    assert CRISIS_MESSAGE not in out


def test_crisis_transcript_overrides_model_guess():
    text = "Work is fine. I want to kill myself. The weather is nice."
    assert verdict_message(NOT_DEPRESSED_LABEL, text) == CRISIS_MESSAGE


def test_verdict_includes_disclaimer():
    msg = verdict_message(DEPRESSED_LABEL, "I had a long day at work.")
    assert "The model's guess" in msg
    assert DISCLAIMER in msg
