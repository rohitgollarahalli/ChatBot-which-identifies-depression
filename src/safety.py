"""Crisis-language detection and help resources.

The chatbot reflects the user's words back as questions (ELIZA-style). That is
harmful when the words are about suicide or self-harm ("Why do you want to
die?"), so every message is checked here first. The check deliberately errs
towards false positives: showing help resources to someone who did not need
them costs little; missing someone who did costs a lot.

This is a keyword screen, not a risk assessment. It will miss indirect or
misspelled expressions of distress.
"""

from __future__ import annotations

import re

# Resources verified 2026-11-22:
# - 988 Suicide & Crisis Lifeline (US): call or text 988, chat at
#   988lifeline.org/chat. https://www.samhsa.gov/mental-health/988/faqs
# - Find A Helpline (ThroughLine, IASP partner): verified helplines in 130+
#   countries. https://findahelpline.com/about
CRISIS_MESSAGE = (
    "It sounds like you might be going through something really painful, and "
    "I'm only a simple program, so I can't give you the support you deserve.\n"
    "Please talk to someone who can help right now:\n"
    "  - In the US: call or text 988 (Suicide & Crisis Lifeline), or chat at "
    "https://988lifeline.org/chat\n"
    "  - Elsewhere: find a free, confidential helpline at https://findahelpline.com\n"
    "  - If you are in immediate danger, call your local emergency number.\n"
    "You can keep talking to me if you'd like, but please reach out to them too."
)

DISCLAIMER = (
    "This is a student NLP project, not a medical tool. It cannot diagnose "
    "depression. If you're struggling, please talk to a doctor or a mental-health "
    "professional. In a crisis, call or text 988 in the US, or find a helpline at "
    "https://findahelpline.com."
)

_PATTERNS = [
    r"\bsuicid(?:e|al)\b",
    r"\bkill(?:ing)?\s+my\s?self\b",
    r"\bend(?:ing)?\s+(?:my|it)\s+(?:life|all)\b",
    r"\bend\s+it\s+all\b",
    r"\b(?:want|wanna|going|ready)\s+(?:to\s+)?die\b",
    r"\bwish\s+i\s+(?:was|were)\s+dead\b",
    r"\bbetter\s+off\s+dead\b",
    r"\b(?:don'?t|do\s+not)\s+want\s+to\s+(?:be\s+alive|live|exist)\b",
    r"\bno\s+(?:reason|point)\s+(?:to|in)\s+living\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\b(?:hurt(?:ing)?|harm(?:ing)?|cut(?:ting)?)\s+my\s?self\b",
    r"\bself[\s-]?harm(?:ing)?\b",
    r"\boverdos(?:e|ing)\b",
]

_CRISIS_RE = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def is_crisis(text: str) -> bool:
    """Return True if the text contains suicide or self-harm language."""
    # Normalise curly apostrophes so "don’t" matches like "don't".
    return bool(_CRISIS_RE.search(text.replace("’", "'")))
