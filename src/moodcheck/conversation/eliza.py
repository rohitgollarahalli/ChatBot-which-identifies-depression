"""An ELIZA-style reflective chatbot engine.

Regex pattern matching in the spirit of Weizenbaum's ELIZA: the engine reflects
the user's statements back as open-ended questions. The crisis screen lives in
``Engine.respond``; this class only generates replies.
"""

from __future__ import annotations

import random
import re

from moodcheck.conversation.engine import Engine

# Pronoun swaps, from nltk.chat.util.reflections (Apache-2.0). Copied because
# importing nltk also loads scikit-learn, which the chat does not need.
_REFLECTIONS = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you",
}

# Each pair is (compiled-pattern, list-of-response-templates).
PAIRS: list[tuple[str, list[str]]] = [
    (
        r"i need (.*)",
        [
            "Why do you need {0}?",
            "Would it really help you to get {0}?",
            "Are you sure you need {0}?",
        ],
    ),
    (
        r"why can'?t i ([^?]*)\??",
        [
            "Do you think you should be able to {0}?",
            "If you could {0}, what would you do?",
            "Have you really tried?",
        ],
    ),
    (
        r"(.*) computer(.*)",
        [
            "Does it seem strange to talk to a computer?",
            "How do computers make you feel?",
            "Do you feel threatened by computers?",
        ],
    ),
    (
        r"i can'?t (.*)",
        [
            "How do you know you can't {0}?",
            "Perhaps you could {0} if you tried.",
            "What would it take for you to {0}?",
        ],
    ),
    (
        r"i am (.*)",
        [
            "Did you come to me because you are {0}?",
            "How long have you been {0}?",
            "How do you feel about being {0}?",
        ],
    ),
    (
        r"i'?m (.*)",
        [
            "How does being {0} make you feel?",
            "Why do you think you're {0}?",
        ],
    ),
    (
        r"i feel (.*)",
        [
            "Good, tell me more about these feelings.",
            "Do you often feel {0}?",
            "When you feel {0}, what do you do?",
        ],
    ),
    (
        r"i think (.*)",
        [
            "Do you doubt {0}?",
            "Do you really think so?",
            "But you're not sure {0}?",
        ],
    ),
    (
        r"because (.*)",
        [
            "Is that the real reason?",
            "What other reasons come to mind?",
        ],
    ),
    (
        r"(.*) sorry (.*)",
        [
            "There are many times when no apology is needed.",
            "What feelings do you have when you apologize?",
        ],
    ),
    (
        r"hello(.*)",
        [
            "Hello... I'm glad you could drop by today.",
            "Hi there... how are you feeling today?",
        ],
    ),
    (
        r"(.*) mother(.*)",
        [
            "Tell me more about your mother.",
            "How do you feel about your mother?",
            "Did you try to talk to your mother about the problems?",
        ],
    ),
    (
        r"(.*) father(.*)",
        [
            "Tell me more about your father.",
            "How did your father make you feel?",
            "Do you think you should talk to your father about the problems you face?",
        ],
    ),
    (
        r"(.*) child(.*)",
        [
            "Did you have close friends since childhood?",
            "What is your favorite childhood memory?",
            "How do you think your childhood experiences relate to your feelings today?",
        ],
    ),
    (
        r"(.*) friend(.*)",
        [
            "Tell me more about your friends.",
            "When you think of a friend, who comes to mind?",
            "Do you have close friends you can talk to?",
        ],
    ),
    (
        r"(.*) job(.*)",
        [
            "Tell me more about your job.",
            "What problems are you facing at your job?",
            "Are you satisfied with the work you are doing?",
        ],
    ),
    (
        r"(.*) wife(.*)",
        [
            "Tell me more about your wife.",
            "What problems are you facing with your wife?",
            "Did you try talking to your wife about this?",
        ],
    ),
    (
        r"(.*)\?",
        [
            "Why do you ask that?",
            "Don't you think the answer lies within yourself?",
            "Why don't you tell me the answer to that?",
        ],
    ),
    (
        r"(.*)",
        [
            "Please tell me more.",
            "Let's change focus a bit. Tell me about your family and friends.",
            "Can you elaborate on that?",
            "How does that make you feel?",
        ],
    ),
]

_COMPILED = [(re.compile(pattern, re.IGNORECASE), responses) for pattern, responses in PAIRS]


def _reflect(fragment: str) -> str:
    """Swap first/second person pronouns (e.g. 'i am' -> 'you are')."""
    return " ".join(_REFLECTIONS.get(word, word) for word in fragment.lower().split())


class ElizaEngine(Engine):
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def _generate(self, message: str) -> str:
        for pattern, responses in _COMPILED:
            match = pattern.match(message.rstrip(".!"))
            if match:
                groups = [_reflect(g) for g in match.groups()]
                return self._rng.choice(responses).format(*groups)
        return "Please tell me more."
