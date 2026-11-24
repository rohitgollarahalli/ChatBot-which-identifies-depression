"""The engine interface.

Every engine gets the crisis screen for free: ``Engine.respond`` runs
``is_crisis`` first and never reaches the engine when it fires. ``respond`` is
final: a subclass that defines it, or inherits another one from a mixin, is
rejected. The chat loop also screens each message itself before calling the
engine. Engines implement ``_generate`` only.
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
        if cls.respond is not Engine.respond:
            raise TypeError(
                f"{cls.__name__} must not define or inherit another respond(): "
                "implement _generate() instead, so the crisis screen always runs."
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
