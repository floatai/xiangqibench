"""Agent interface: one chat completion per call."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Completion:
    text: str
    reasoning: str | None = None
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None


class Agent(Protocol):
    """Anything that maps a chat history to the next assistant reply.

    Implementations must be safe to call from one thread at a time; the runner
    gives every worker thread its own agent instance.
    """

    name: str

    def complete(self, messages: Sequence[dict]) -> Completion: ...


class ScriptedAgent:
    """Replays a fixed list of replies; for tests and offline replays."""

    def __init__(self, replies: Sequence[str | Completion], name: str = "scripted"):
        self.name = name
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def complete(self, messages: Sequence[dict]) -> Completion:
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise RuntimeError("ScriptedAgent ran out of replies")
        reply = self._replies.pop(0)
        return reply if isinstance(reply, Completion) else Completion(text=reply)
