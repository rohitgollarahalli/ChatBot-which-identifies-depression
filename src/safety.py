"""Crisis-language detection and help resources.

The chatbot reflects the user's words back as questions (ELIZA-style). That is
harmful when the words are about suicide or self-harm ("Why do you want to
die?"), so every message is checked here first. The check errs towards false
positives: showing help resources to someone who did not need them costs
little; missing someone who did costs a lot.

This is a keyword screen, not a risk assessment. It cannot be complete:
indirect statements ("nobody would notice if I was gone") need a semantic
model. Known gaps are tracked as strict xfails in ``tests/test_safety.py``.
"""

from __future__ import annotations

import re
import unicodedata

# Resources verified 2026-11-22:
# - 988 Suicide & Crisis Lifeline (US): call or text 988, chat at
#   988lifeline.org/chat. https://www.samhsa.gov/mental-health/988/faqs
# - Find A Helpline (ThroughLine, IASP partner): directory of verified
#   helplines worldwide. https://findahelpline.com/about
CRISIS_MESSAGE = (
    "It sounds like you might be going through something really painful, and "
    "I'm only a simple program, so I can't give you the support you deserve.\n"
    "Please talk to someone who can help right now:\n"
    "  - In the US: call or text 988 (Suicide & Crisis Lifeline, free, 24/7), "
    "or chat at https://988lifeline.org/chat\n"
    "  - Elsewhere: find a free, confidential helpline at https://findahelpline.com\n"
    "  - If you are in immediate danger, call your local emergency number.\n"
    "Please reach out to one of them first. I'm still here if you want to keep typing."
)

CRISIS_REMINDER = (
    "Please remember: in the US you can call or text 988 any time, and "
    "https://findahelpline.com lists free helplines in other countries."
)

DISCLAIMER = (
    "This is a student NLP project, not a medical tool. It cannot diagnose "
    "depression. If you're struggling, please talk to a doctor or a mental-health "
    "professional. In a crisis, call or text 988 in the US, or find a helpline at "
    "https://findahelpline.com."
)

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʼ´`"})
_LEET = str.maketrans({"1": "i", "3": "e", "0": "o", "4": "a", "@": "a", "$": "s", "!": "i"})

# Words allowed between a first-person subject and "die"/"dead":
# "I want to just die", "I'm so depressed I could die".
_GAP = r"(?:\W+\w+){0,4}?\W+"

_PATTERNS = [
    # Suicide, including common misspellings (sucide, suicidle).
    r"\bsu[i]?c[i]?d",
    r"\bkil+(?:ed|ing|s)?\s*my\s*self\b",
    r"\bkms\b",
    r"\bunalive\b",
    r"\boff\s+myself\b",
    r"\bend\s+myself\b",
    r"\b(?:take|end)(?:ing)?\s+my\s+(?:own\s+)?life\b",
    r"\bend(?:ing)?\s+(?:it\s+all|things|my\s+suffering|everything)\b",
    r"\b(?:going\s+to|gonna|want\s+to|wanna|will|ready\s+to|i'm\s+going\s+to)\s+end\s+it\b(?!\s+with)",
    # Methods.
    r"\b(?:hang|shoot|drown|poison|stab|starve|electrocute)\s+myself\b",
    r"\bslit\s+my\s+wrists?\b",
    r"\bjump\s+(?:off|from)\s+(?:a|the|this)\s+(?:bridge|building|roof|cliff|tower)\b",
    r"\b(?:take|swallow|overdose\s+on)\s+(?:all\s+)?(?:of\s+)?my\s+(?:pills|meds|medication|tablets)\b",
    r"\bod'?d\b",
    r"\boverdos(?:e|ed|ing)\b(?!\s+on\s+(?:caffeine|coffee|sugar|chocolate))",
    # Wanting to die or be dead, said by the speaker.
    r"\bi(?:'m|'d|'ll| am| would| will| could| should| just)?" + _GAP + r"(?:die|dead)\b",
    r"\bi\s+(?:want|wanna|wish)\s+to\s+(?:just\s+)?die\b",
    r"\bfeel\s+like\s+dying\b",
    r"\bwant\s+to\s+(?:just\s+)?die\b",
    r"\b(?:want|wanna)\s+to\s+(?:kill|hurt)\s+me\b",
    r"\b(?:want|wanna)\s+to\s+be\s+gone\b",
    r"\bbetter\s+off\s+(?:dead|without\s+me)\b",
    # Not wanting to live.
    r"\b(?:don't|do\s+not|dont)\s+(?:want|wanna)\s+(?:to\s+)?(?:be\s+alive|live|exist|be\s+here)\b"
    r"(?!\s+(?:in|at|with|near|there|on|like|together|alone\s+in))",
    r"\bno\s+(?:reason|point)\s+(?:to\s+|in\s+)?(?:live|living|go\s+on|going\s+on)\b(?!\s+(?:here|in|at|with))",
    r"\blife\s+(?:isn't|is\s+not|ain't)\s+worth\s+living\b",
    r"\b(?:tired|sick)\s+of\s+(?:living|being\s+alive|life)\b",
    r"\bhate\s+being\s+alive\b",
    r"\b(?:done\s+with|give\s+up\s+on|giving\s+up\s+on)\s+life\b",
    r"\bcan't\s+go\s+on\b",
    r"\bcan't\s+do\s+this\s+any\s*more\b",
    r"\bending\s+it\b(?!\s+with)",
    r"\bdone\s+with\s+everything\b",
    r"\bpoint\s+(?:in|of)\s+living\b",
    r"\b(?:don't|do\s+not|dont)\s+(?:want|wanna)\s+to\s+wake\s+up\b",
    r"\bwish\s+i\s+(?:wasn't|weren't|didn't|was\s+not|were\s+not)\s+(?:alive|exist|here)\b",
    r"\bwish\s+i(?:'d|\s+had|\s+was|\s+were)?\s+never\s+(?:been\s+)?born\b",
    r"\bwish\s+i\s+could\s+disappear\b",
    r"\bdisappear\s+forever\b",
    # Sleep.
    r"\bsleep\s+and\s+never\s+wake\s+up\b",
    r"\b(?:hope|want|wish|wanna)\b(?:\W+\w+){0,3}?\W+(?:not|never|don't|won't)\s+wake\s+up\b",
    # Burden.
    r"\b(?:i'm|i\s+am)\s+(?:a|such\s+a|just\s+a)\s+burden\b",
    # Self-harm.
    r"\bself\s*harm",
    r"\b(?:hurt|harm|cut|burn)(?:ing)?\s+my\s*self\b(?!\s+(?:some\s+)?(?:slack|a\s+break|shaving))",
    r"\b(?:been|started|keep|still)\s+cutting\b(?!\s+(?:back|down|my\s+hair|the|costs|class|onions|out|it))",
]

_CRISIS_RE = re.compile("|".join(_PATTERNS))


def _normalise(text: str) -> str:
    # Map apostrophes before NFKC, which would split U+00B4 into a space + accent.
    text = unicodedata.normalize("NFKC", text.translate(_APOSTROPHES)).translate(_ZERO_WIDTH)
    text = text.lower()
    # Join letters split by punctuation: "k.i.l.l" -> "kill".
    text = re.sub(r"(?<=[a-z])[.\-_*](?=[a-z])", "", text)
    return text


def _deleet(text: str) -> str:
    # Only inside words that also contain letters, so "24h" or "988" stay intact.
    return re.sub(r"\S*[a-z]\S*", lambda m: m.group(0).translate(_LEET), text)


def is_crisis(text: str) -> bool:
    """Return True if the text contains suicide, self-harm or hopelessness language."""
    norm = _normalise(text)
    return bool(_CRISIS_RE.search(norm) or _CRISIS_RE.search(_deleet(norm)))
