import pytest

from src.chatbot import respond
from src.safety import CRISIS_MESSAGE, is_crisis

CRISIS = [
    "I want to kill myself",
    "i've been thinking about suicide",
    "Sometimes I think I'd be better off dead.",
    "I want to end my life",
    "I don't want to be alive anymore",
    "I want to die",
    "I keep hurting myself",
    "I've been self-harming again",
    "there's no reason to live",
    "I'm suicidal",
    "I want to end it all",
]

SAFE = [
    "I'm tired after work",
    "My phone died yesterday",
    "This exam is killing me",
    "I need a holiday",
    "I dyed my hair blue",
]


@pytest.mark.parametrize("text", CRISIS)
def test_detects_crisis_language(text):
    assert is_crisis(text)


@pytest.mark.parametrize("text", SAFE)
def test_ignores_everyday_language(text):
    assert not is_crisis(text)


@pytest.mark.parametrize("text", CRISIS)
def test_bot_never_reflects_crisis_language(text):
    # ELIZA reflection would turn "I want to die" into "Why do you want to die?".
    assert respond(text) == CRISIS_MESSAGE


def test_crisis_message_points_to_help():
    assert "988" in CRISIS_MESSAGE
    assert "findahelpline.com" in CRISIS_MESSAGE
