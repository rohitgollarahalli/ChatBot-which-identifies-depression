import dataclasses

import pytest

from moodcheck.conversation.engine import Engine, Reply
from moodcheck.safety import CRISIS_MESSAGE


class Recorder(Engine):
    def __init__(self):
        self.seen = []

    def _generate(self, message):
        self.seen.append(message)
        return f"echo {message}"


def test_reply_is_frozen():
    reply = Reply("hi", False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reply.text = "other"  # type: ignore[misc]


def test_crisis_short_circuits_generate():
    engine = Recorder()
    reply = engine.respond("I want to kill myself")
    assert reply == Reply(CRISIS_MESSAGE, True)
    assert engine.seen == []


def test_normal_message_goes_to_generate():
    engine = Recorder()
    assert engine.respond("nice weather") == Reply("echo nice weather", False)
    assert engine.seen == ["nice weather"]


def test_subclass_cannot_override_respond():
    with pytest.raises(TypeError, match="respond"):

        class Sneaky(Engine):
            def respond(self, message):
                return Reply("no screen", False)

            def _generate(self, message):
                return ""


def test_engine_requires_generate():
    class Incomplete(Engine):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]
