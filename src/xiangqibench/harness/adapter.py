"""Xiangqi endgame adapter: the game-facing half of the harness.

Wraps :class:`~xiangqibench.rules.XiangqiEnv` and produces the objective move
descriptions, piece census, and material summary that the agent receives.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from xiangqibench.rules import XiangqiEnv

_RED_PIECES = {
    "K": "King (帅)", "A": "Advisor (仕)", "B": "Elephant (相)",
    "N": "Horse (马)", "R": "Rook (车)", "C": "Cannon (炮)", "P": "Pawn (兵)",
}
_BLACK_PIECES = {
    "k": "King (将)", "a": "Advisor (士)", "b": "Elephant (象)",
    "n": "Horse (马)", "r": "Rook (车)", "c": "Cannon (炮)", "p": "Pawn (卒)",
}
_CENSUS_ORDER = ["King", "Advisor", "Elephant", "Horse", "Rook", "Cannon", "Pawn"]
# Nominal values for objective material accounting, not a positional evaluation.
_PIECE_VALUES = {"R": 9, "C": 4.5, "N": 4, "A": 2, "B": 2, "P": 1, "K": 0}


@dataclass
class MoveResult:
    success: bool
    error: str | None = None
    in_check: bool = False
    game_over: bool = False
    outcome: dict | None = None
    moving_piece: str | None = None
    captured_piece: str | None = None
    description: str | None = None
    material_summary: str | None = None
    repetition_count: int = 0
    max_repetitions: int = 0


def piece_at(fen: str, col: int, row: int) -> str | None:
    """FEN piece letter at ``(col 0-8 = a-i, row 0-9)``, or ``None`` if empty."""
    if not (0 <= row <= 9 and 0 <= col <= 8):
        return None
    rank = fen.split()[0].split("/")[9 - row]
    c = 0
    for ch in rank:
        if ch.isdigit():
            c += int(ch)
        else:
            if c == col:
                return ch
            c += 1
    return None


def piece_name(ch: str | None) -> str | None:
    if not ch:
        return None
    return _RED_PIECES.get(ch) or _BLACK_PIECES.get(ch)


def material_summary(fen: str) -> str:
    red: dict[str, int] = {}
    black: dict[str, int] = {}
    red_val = black_val = 0.0
    for ch in fen.split()[0]:
        if not ch.isalpha():
            continue
        up = ch.upper()
        val = _PIECE_VALUES.get(up, 0)
        if ch.isupper():
            red[up] = red.get(up, 0) + 1
            red_val += val
        else:
            black[up] = black.get(up, 0) + 1
            black_val += val

    def fmt(counts: dict[str, int]) -> str:
        return " ".join(f"{p}{counts.get(p, 0)}" for p in ("R", "N", "B", "A", "C", "P"))

    diff = red_val - black_val
    if diff > 0:
        bal = f"Red +{diff:g} (nominal)"
    elif diff < 0:
        bal = f"Black +{-diff:g} (nominal)"
    else:
        bal = "even (nominal)"
    return f"Red: {fmt(red)} | Black: {fmt(black)} | Balance: {bal}"


class XiangqiAdapter:
    """Endgame position with the agent-facing views of the rules engine."""

    def __init__(self, initial_fen: str):
        self._initial_fen = initial_fen
        self._env = XiangqiEnv()

    def game_type(self) -> str:
        return "xiangqi-endgame"

    def player_names(self) -> tuple[str, str]:
        return ("red", "black")

    def reset(self) -> str:
        self._env.set_state_from_fen(self._initial_fen)
        return self.get_board_ascii()

    def get_fen(self) -> str:
        return self._env.get_board_state("fen")

    def get_board_ascii(self) -> str:
        return self._env.get_board_state("text")

    def get_legal_moves(self) -> list[str]:
        return self._env.get_legal_moves()

    def is_game_over(self) -> bool:
        return self._env.is_game_over()

    def get_outcome(self) -> dict | None:
        return self._env.get_outcome_details()

    def current_player(self) -> str:
        return self._env.get_current_player()

    def is_in_check(self) -> bool:
        return self._env.is_in_check()

    def get_move_count(self) -> int:
        return self._env.move_count

    def get_move_history(self) -> list[str]:
        return list(self._env.move_history)

    def get_piece_census(self) -> str:
        """Coordinate-keyed listing of every piece, grouped by side and type."""
        red: dict[str, list[str]] = {}
        black: dict[str, list[str]] = {}
        for ri, rank in enumerate(self.get_fen().split()[0].split("/")):
            row, col = 9 - ri, 0
            for ch in rank:
                if ch.isdigit():
                    col += int(ch)
                    continue
                bucket = red if ch.isupper() else black
                bucket.setdefault(piece_name(ch) or ch, []).append(f"{chr(ord('a') + col)}{row}")
                col += 1

        def rank_of(name: str) -> int:
            return next((i for i, label in enumerate(_CENSUS_ORDER) if name.startswith(label)),
                        len(_CENSUS_ORDER))

        def fmt(bucket: dict[str, list[str]]) -> str:
            parts = [f"{name} {','.join(sorted(coords))}"
                     for name, coords in sorted(bucket.items(), key=lambda kv: rank_of(kv[0]))]
            return " | ".join(parts) if parts else "(none)"

        return (
            "Piece positions (ground truth — trust this over your memory):\n"
            f"  Red : {fmt(red)}\n"
            f"  Black: {fmt(black)}"
        )

    def make_move(self, move: str) -> MoveResult:
        if move not in self.get_legal_moves():
            return MoveResult(success=False, error=f"Illegal move: '{move}'")

        side = self.current_player()
        fen_before = self.get_fen()
        fc, fr, tc, tr = ord(move[0]) - 97, int(move[1]), ord(move[2]) - 97, int(move[3])
        moving = piece_name(piece_at(fen_before, fc, fr))
        captured = piece_name(piece_at(fen_before, tc, tr))

        if not self._env.step(move):
            return MoveResult(success=False, error=f"Engine rejected move: '{move}'")

        game_over = self._env.is_game_over()
        verb = f"captures {captured}" if captured else "(no capture)"
        return MoveResult(
            success=True,
            in_check=self._env.is_in_check(),
            game_over=game_over,
            outcome=self._env.get_outcome_details() if game_over else None,
            moving_piece=moving,
            captured_piece=captured,
            description=f"{side.capitalize()} {moving or 'piece'} {move[:2]}→{move[2:]} {verb}",
            material_summary=material_summary(self.get_fen()),
            repetition_count=self._env.get_repetition_count(),
            max_repetitions=self._env.MAX_REPETITIONS,
        )

    def clone(self) -> XiangqiAdapter:
        new = object.__new__(XiangqiAdapter)
        new._initial_fen = self._initial_fen
        new._env = copy.deepcopy(self._env)
        return new

    def forfeit(self, loser: str, reason: str = "forfeit") -> None:
        self._env.forfeit(loser, reason)
