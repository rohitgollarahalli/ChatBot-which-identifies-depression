import random

from moodcheck.conversation.eliza import ElizaEngine


def test_same_seed_same_replies():
    messages = ["I need a break", "I am tired.", "my mother called", "why is it so?", "ok"]
    first = [ElizaEngine(random.Random(1)).respond(m).text for m in messages]
    second = [ElizaEngine(random.Random(1)).respond(m).text for m in messages]
    assert first == second


def test_reflection_and_trailing_punctuation():
    engine = ElizaEngine(random.Random(0))
    text = engine.respond("I am worried about my exams!").text
    assert "worried about your exams" in text
    assert "!" not in text


def test_pattern_choice():
    reply = ElizaEngine(random.Random(0)).respond("hello there")
    assert reply.text in {
        "Hello... I'm glad you could drop by today.",
        "Hi there... how are you feeling today?",
    }
    assert not reply.crisis


def test_default_rng_works():
    assert ElizaEngine().respond("anything at all").text
