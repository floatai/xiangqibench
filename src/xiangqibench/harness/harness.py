"""Game harness: the environment side of one endgame trial.

The harness never calls a model. It exposes ``build_messages`` (the chat the
agent sees), ``execute`` (apply one parsed command), and ``get_record``; the
runner drives the loop and plays the defender.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from xiangqibench.harness.actions import ActionExecutor, ActionResult
from xiangqibench.harness.adapter import XiangqiAdapter
from xiangqibench.harness.config import HarnessConfig
from xiangqibench.harness.player_state import PlayerState
from xiangqibench.harness.prompts import build_system_prompt
from xiangqibench.harness.record import GameRecord
from xiangqibench.harness.simulator import Simulator


class GameHarness:

    def __init__(self, game: XiangqiAdapter, config: HarnessConfig | None = None,
                 players: dict[str, str] | None = None):
        self._game = game
        self._config = config or HarnessConfig()
        self._players = players or {}
        self._simulator = Simulator(max_steps=self._config.max_simulate_steps)
        self._executor = ActionExecutor(self._game, self._config, self._simulator)

        system_prompt = build_system_prompt(self._config)
        self._player_states = {
            p: PlayerState(p, system_prompt, self._config) for p in self._game.player_names()
        }
        self._record = GameRecord(game_type=self._game.game_type(),
                                  config=asdict(self._config), players=self._players)
        self._turn_number = 0
        self._turn_action_counts: dict[str, int] = {}
        self._turn_invalid_count = 0
        self._turn_start_time = 0.0

        board_ascii = self._game.reset()
        fen = self._game.get_fen()
        legal = self._game.get_legal_moves()
        self._record.initial_fen = fen
        self._record.initial_board = board_ascii
        for name, state in self._player_states.items():
            state.add_initial_state(board_ascii, fen, legal, name)

    @property
    def config(self) -> HarnessConfig:
        return self._config

    @property
    def game(self) -> XiangqiAdapter:
        return self._game

    @property
    def record(self) -> GameRecord:
        return self._record

    @property
    def turn_number(self) -> int:
        return self._turn_number

    def current_player(self) -> str:
        return self._game.current_player()

    def is_over(self) -> bool:
        return self._game.is_game_over()

    def build_messages(self, player: str) -> list[dict]:
        """Messages to send to ``player``; opens a new turn on its first call."""
        if player not in self._player_states:
            raise ValueError(f"Unknown player: {player}")
        if not self._turn_action_counts:
            self._turn_number += 1
            self._turn_start_time = time.time()
            self._record.start_turn(self._turn_number, player,
                                    self._game.get_fen(), self._game.get_legal_moves())
            self._player_states[player].flush_pending_opponent_info()
        return self._player_states[player].messages

    def execute(self, player: str, action: str, params: dict[str, Any],
                thinking: str | None = None, usage: dict | None = None,
                raw_response: dict | None = None, raw_text: str | None = None) -> ActionResult:
        """Execute one command; ``raw_text`` is the agent's verbatim reply."""
        if player != self._game.current_player():
            raise ValueError(f"It is {self._game.current_player()}'s turn, not {player}'s.")

        state = self._player_states[player]
        prompt_snapshot = state.messages if self._config.record_prompt_snapshots else None
        state.add_assistant_turn(raw_text, thinking, action, params)

        if (sum(self._turn_action_counts.values()) >= self._config.max_actions_per_turn
                and action not in ActionExecutor.TURN_ENDING_ACTIONS):
            result = self._executor.force_forfeit(player, "action_limit")
            self._record_stat("forfeit")
            self._finalize_forced(player, result, thinking, usage, raw_response)
            return result

        result = self._executor.execute(player, action, params, self._turn_action_counts)
        self._turn_action_counts[action] = self._turn_action_counts.get(action, 0) + 1
        if not result.success and (action == "move" or action not in ActionExecutor.KNOWN_ACTIONS):
            self._turn_invalid_count += 1
        if result.success and action in ("simulate", "view_board"):
            self._record_stat(action)

        self._record.record_interaction(
            step=sum(self._turn_action_counts.values()),
            messages_count=len(state.messages),
            thinking=thinking,
            tool_call={"name": action, "arguments": params},
            raw_response=raw_response,
            usage=usage,
            action_result={
                "action": result.action,
                "success": result.success,
                "feedback": result.feedback,
                "turn_complete": result.turn_complete,
                "game_over": result.game_over,
            },
            messages_sent=prompt_snapshot,
        )
        state.add_tool_result(result.feedback)

        if result.turn_complete:
            self._finalize_turn_end(player, result)
            return result
        if self._turn_invalid_count >= self._config.max_invalid_actions_per_turn:
            return self.force_turn_resolution(player, reason="invalid_action_limit")
        return result

    def build_no_command_feedback(self) -> str:
        """Re-prompt sent when a reply contains no usable command block."""
        cfg = self._config
        cmds = ["move <from><to>"]
        if cfg.enable_simulate:
            cmds.append("simulate <m1> <m2> ...")
        if cfg.enable_view_board:
            cmds.append("view_board")
        cmds += ["legal_moves", "history", "resign"]
        return (
            "No command detected. Put exactly ONE command inside a "
            "```bash``` code block, for example:\n"
            "```bash\nmove h2e2\n```\n"
            f"Commands: {', '.join(cmds)}."
        )

    def append_text_exchange(self, player: str, assistant_text: str, user_feedback: str) -> None:
        self._player_states[player].add_text_exchange(user_feedback, assistant_text)

    def force_turn_resolution(self, player: str, reason: str = "no_tool_call") -> ActionResult:
        """Forfeit a turn the agent cannot resolve within its budgets."""
        result = self._executor.force_forfeit(player, reason)
        thinking = f"[environment: {reason}]"
        self._player_states[player].add_assistant_turn(None, thinking, result.action, result.extra)
        self._record_stat("forfeit")
        self._finalize_forced(player, result, thinking, None, None)
        return result

    def get_record(self) -> dict:
        per_player = {name: s.to_serializable() for name, s in self._player_states.items()}
        return self._record.to_dict(per_player_messages=per_player)

    def _finalize_forced(self, player: str, result: ActionResult, thinking: str | None,
                         usage: dict | None, raw_response: dict | None) -> None:
        state = self._player_states[player]
        self._record.record_interaction(
            step=sum(self._turn_action_counts.values()) + 1,
            messages_count=len(state.messages),
            thinking=thinking,
            tool_call={"name": "move", "arguments": result.extra},
            raw_response=raw_response,
            usage=usage,
            action_result={
                "action": "move", "success": True,
                "feedback": result.feedback, "turn_complete": True,
                "game_over": result.game_over, "forced": True,
            },
        )
        state.add_tool_result(result.feedback)
        self._finalize_turn_end(player, result)

    def _finalize_turn_end(self, player: str, result: ActionResult) -> None:
        move_played = None
        if result.action == "move" and result.success:
            move_played = (result.extra or {}).get("move")

        fen_after = self._game.get_fen()
        in_check = self._game.is_in_check()
        self._record.end_turn(move_played, fen_after, in_check,
                              int((time.time() - self._turn_start_time) * 1000))

        if move_played:
            self._player_states[self._opponent(player)].set_pending_opponent_info(
                self._opponent_notice(player, move_played, result, fen_after, in_check))

        if result.game_over:
            self._record.set_result({"winner": result.winner, "reason": result.termination_reason})
        self._turn_action_counts = {}
        self._turn_invalid_count = 0

    def _opponent_notice(self, player: str, move: str, result: ActionResult,
                         fen_after: str, in_check: bool) -> str:
        extra = result.extra or {}
        captured = extra.get("captured_piece")
        played = f"Opponent ({player}) played: {move}"
        if extra.get("description"):
            played += f" — {extra['description']}"

        if not self._config.push_state:
            notice = f"{played}\nCheck: {'Yes — you are in check!' if in_check else 'No'}\n"
            notice += f"You LOST your {captured} on this move.\n" if captured else "No capture.\n"
            return notice + "It is now your turn."

        notice = f"{played}\nCheck: {'Yes' if in_check else 'No'}\n"
        if captured:
            notice += f"You LOST your {captured} on this move.\n"
        notice += f"Move {self._game.get_move_count()} — FEN: {fen_after}\n\n"
        notice += f"Current board:\n{self._game.get_board_ascii()}"
        census = self._game.get_piece_census()
        if census:
            notice += f"\n\n{census}"
        if not result.game_over:
            legal = self._game.get_legal_moves()
            notice += f"\n\nLegal moves ({len(legal)}): {', '.join(legal)}"
        return notice + "\n\nIt is now your turn. Call a tool to proceed."

    def _opponent(self, player: str) -> str:
        p1, p2 = self._game.player_names()
        return p2 if player == p1 else p1

    def _record_stat(self, action: str) -> None:
        key = {"simulate": "total_simulations", "view_board": "total_view_boards",
               "forfeit": "total_forfeits"}[action]
        self._record.stats[key] = self._record.stats.get(key, 0) + 1
