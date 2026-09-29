"""Deterministic defenders for the side opposite the agent.

The benchmark defender is :class:`PikafishDefender`: Pikafish at a fixed depth,
falling back per move to :class:`RuleDefender` when the engine rejects a
composed position. Each :class:`DefenderChoice` names the backend that produced
the move, and the runner stores it with the trajectory.

:class:`RuleDefender` scores every legal reply with a bounded forced-mate
search from the agent's side and plays, in order of preference: a reply after
which no mate is found; otherwise the reply with the longest mate distance;
then checking replies; then the lexicographically smallest ICCS string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cchess

from xiangqibench.defender.engine import PikafishEngine
from xiangqibench.defender.search import _apply, forced_mate_search


@dataclass
class DefenderChoice:
    move: str | None
    backend: str
    legal_count: int
    challenger_mate_distance: int | None = None
    gave_check: bool = False


class Defender(Protocol):
    def choose(self, fen: str) -> DefenderChoice: ...

    def info(self) -> dict: ...

    def close(self) -> None: ...


def _side_to_move(board) -> str:
    return "red" if board.move_player == cchess.RED else "black"


def _other(side: str) -> str:
    if side not in ("red", "black"):
        raise ValueError(f"side must be 'red' or 'black', got {side!r}")
    return "black" if side == "red" else "red"


class RuleDefender:
    """Mate-aware search defender; a pure function of the FEN."""

    backend = "rule"

    def __init__(self, challenger: str, search_depth: int = 5):
        self.defender = _other(challenger)
        self.challenger = challenger
        self.search_depth = search_depth

    def info(self) -> dict:
        return {"backend": self.backend, "search_depth": self.search_depth}

    def close(self) -> None:
        pass

    def choose(self, fen: str) -> DefenderChoice:
        board = cchess.ChessBoard(fen)
        if _side_to_move(board) != self.defender:
            raise ValueError(f"defender called on {_side_to_move(board)}'s turn")
        candidates = []
        for src, dst in board.create_moves():
            if board.is_checked_move(src, dst):
                continue
            child = _apply(board, src, dst)
            if child is None:
                continue
            candidates.append((
                forced_mate_search(child, self.challenger, self.search_depth),
                board.is_checking_move(src, dst),
                cchess.pos2iccs(src, dst),
            ))
        if not candidates:
            return DefenderChoice(move=None, backend=self.backend, legal_count=0)
        candidates.sort(key=lambda c: (
            0 if c[0] is None else 1,
            -(c[0] if c[0] is not None else 10**9),
            0 if c[1] else 1,
            c[2],
        ))
        mate, check, move = candidates[0]
        return DefenderChoice(move=move, backend=self.backend, legal_count=len(candidates),
                              challenger_mate_distance=mate, gave_check=check)


class PikafishDefender:
    """Pikafish at fixed depth with a per-move rule-based fallback."""

    backend = "pikafish"

    def __init__(self, challenger: str, engine: PikafishEngine, *, fallback_depth: int = 5):
        self.defender = _other(challenger)
        self.engine = engine
        self._fallback = RuleDefender(challenger, search_depth=fallback_depth)

    def info(self) -> dict:
        return {"backend": self.backend, **self.engine.info(),
                "fallback": self._fallback.info()}

    def close(self) -> None:
        self.engine.close()

    def choose(self, fen: str) -> DefenderChoice:
        board = cchess.ChessBoard(fen)
        if _side_to_move(board) != self.defender:
            raise ValueError(f"defender called on {_side_to_move(board)}'s turn")
        legal = [cchess.pos2iccs(s, d) for s, d in board.create_moves()]
        if not legal:
            return DefenderChoice(move=None, backend=self.backend, legal_count=0)
        move = self.engine.best_move(fen)
        if not move or move not in legal:
            return self._fallback.choose(fen)
        return DefenderChoice(move=move, backend=self.backend, legal_count=len(legal))
