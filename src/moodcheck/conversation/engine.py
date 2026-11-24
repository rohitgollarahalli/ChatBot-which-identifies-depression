"""The engine interface.

Every engine gets the crisis screen for free: ``Engine.respond`` runs
``is_crisis`` first and never reaches the engine when it fires. ``respond`` is
final and subclasses may not define it, so a new engine cannot skip the screen.
Engines implement ``_generate`` only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, final

from moodcheck.safety import CRISIS_MESSAGE, is_crisis


@dataclass(frozen=True)
class Reply:
    """A reply to one user message; ``crisis`` is True for the crisis message."""

    text: str
    crisis: bool = False


class Engine(ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "respond" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not define respond(): implement _generate() instead, "
                "so the crisis screen cannot be bypassed."
            )

    @final
    def respond(self, message: str) -> Reply:
        # Never reflect suicide or self-harm language back as a question.
        if is_crisis(message):
            return Reply(CRISIS_MESSAGE, crisis=True)
        return Reply(self._generate(message))

    @abstractmethod
    def _generate(self, message: str) -> str:
        """Return the reply to a message that already passed the crisis screen."""
