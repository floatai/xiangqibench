"""Trajectory record of one trial."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class InteractionEntry:
    """One agent call within a turn."""

    step: int
    messages_sent_count: int
    llm_response: dict = field(default_factory=dict)
    action_result: dict = field(default_factory=dict)
    messages_sent: list | None = None

    def to_dict(self) -> dict:
        d = {
            "step": self.step,
            "messages_sent_count": self.messages_sent_count,
            "llm_response": self.llm_response,
            "action_result": self.action_result,
        }
        if self.messages_sent is not None:
            d["messages_sent"] = self.messages_sent
        return d


@dataclass
class TurnEntry:
    """One ply: the agent's turn (possibly several calls) or one defender move."""

    turn: int
    player: str
    fen_before: str
    legal_moves_before: list[str]
    interactions: list[InteractionEntry] = field(default_factory=list)
    move_played: str | None = None
    fen_after: str | None = None
    in_check: bool = False
    timestamp: str = ""
    duration_ms: int = 0
    defender_backend: str | None = None

    def to_dict(self) -> dict:
        d = {
            "turn": self.turn,
            "player": self.player,
            "fen_before": self.fen_before,
            "legal_moves_before": self.legal_moves_before,
            "interactions": [i.to_dict() for i in self.interactions],
            "move_played": self.move_played,
            "fen_after": self.fen_after,
            "in_check": self.in_check,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }
        if self.defender_backend is not None:
            d["defender_backend"] = self.defender_backend
        return d


class GameRecord:
    """Accumulates the trajectory, moves, and statistics of a trial."""

    def __init__(self, game_type: str, config: dict, players: dict[str, str]):
        self.game_id = str(uuid.uuid4())
        self.game_type = game_type
        self.config = config
        self.players = players
        self.initial_fen = ""
        self.initial_board = ""
        self.result: dict | None = None
        self.moves: list[str] = []
        self.trajectory: list[TurnEntry] = []
        self._current_turn: TurnEntry | None = None
        self.total_api_calls = 0
        self.stats: dict[str, Any] = {
            "total_simulations": 0,
            "total_view_boards": 0,
            "forced_moves": 0,
            "total_tokens": {},
        }

    @property
    def current_turn(self) -> TurnEntry | None:
        return self._current_turn

    def start_turn(self, turn_number: int, player: str, fen: str, legal_moves: list[str]) -> None:
        self._current_turn = TurnEntry(
            turn=turn_number,
            player=player,
            fen_before=fen,
            legal_moves_before=legal_moves,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def record_interaction(
        self,
        step: int,
        messages_count: int,
        thinking: str | None = None,
        tool_call: dict | None = None,
        raw_response: dict | None = None,
        usage: dict | None = None,
        action_result: dict | None = None,
        messages_sent: list | None = None,
    ) -> None:
        self.total_api_calls += 1
        entry = InteractionEntry(
            step=step,
            messages_sent_count=messages_count,
            llm_response={
                "thinking": thinking,
                "tool_call": tool_call,
                "raw_response": raw_response,
                "usage": usage,
            },
            action_result=action_result or {},
            messages_sent=messages_sent,
        )
        if usage:
            player = self._current_turn.player if self._current_turn else "unknown"
            tokens = self.stats["total_tokens"].setdefault(player, {"prompt": 0, "completion": 0})
            tokens["prompt"] += usage.get("prompt_tokens", 0) or 0
            tokens["completion"] += usage.get("completion_tokens", 0) or 0
        if self._current_turn:
            self._current_turn.interactions.append(entry)

    def end_turn(self, move_played: str | None, fen_after: str, in_check: bool, duration_ms: int = 0) -> None:
        if self._current_turn:
            self._current_turn.move_played = move_played
            self._current_turn.fen_after = fen_after
            self._current_turn.in_check = in_check
            self._current_turn.duration_ms = duration_ms
            self.trajectory.append(self._current_turn)
            if move_played:
                self.moves.append(move_played)
            self._current_turn = None

    def set_result(self, result: dict) -> None:
        self.result = result

    def to_dict(self, per_player_messages: dict[str, list] | None = None) -> dict:
        return {
            "game_id": self.game_id,
            "game_type": self.game_type,
            "config": self.config,
            "players": self.players,
            "initial_fen": self.initial_fen,
            "initial_board": self.initial_board,
            "result": self.result,
            "total_turns": len(self.trajectory),
            "total_api_calls": self.total_api_calls,
            "moves": self.moves,
            "trajectory": [t.to_dict() for t in self.trajectory],
            "per_player_messages": per_player_messages,
            "stats": self.stats,
        }
