"""Per-player conversation state in the REPL protocol."""

from __future__ import annotations

import json
from typing import Any

from xiangqibench.harness.config import HarnessConfig

_REPL_ACT = "Type one command inside a ```bash``` block to act (e.g. `move h2e2`). "


class PlayerState:
    """Builds and maintains the chat message list shown to one player."""

    def __init__(self, player: str, system_prompt: str, config: HarnessConfig):
        self.player = player
        self._config = config
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._pending_opponent_info: str | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def add_initial_state(self, board_ascii: str, fen: str, legal_moves: list[str], player_side: str) -> None:
        self._messages.append({"role": "user", "content": self._opening(
            board_ascii, fen, legal_moves, player_side)})

    def _opening(self, board_ascii: str, fen: str, legal_moves: list[str], player_side: str) -> str:
        cfg = self._config
        started = f"Game started. You are playing as {player_side.upper()}."
        if cfg.obs_arm:
            return self._ablation_opening(started, board_ascii, fen, legal_moves)
        if cfg.blindfold:
            return (
                f"{started}\n\n"
                f"Initial board:\n{board_ascii}\n\n"
                f"FEN: {fen}\n\n"
                f"BLINDFOLD: this is the ONLY time you are shown the board. From "
                f"now on you get only per-move diffs — no board, FEN, or legal "
                f"moves. Track the position yourself.\n\n"
                f"{_REPL_ACT}You may `think ...` first, then `move` to end "
                f"your turn. There is no board viewer or legal-move list."
            )
        return (
            f"{started}\n\n"
            f"Initial board:\n{board_ascii}\n\n"
            f"FEN: {fen}\n\n"
            f"Legal moves ({len(legal_moves)}): {', '.join(legal_moves)}\n\n"
            f"{_REPL_ACT}You may `view_board`, `simulate ...`, or "
            f"`legal_moves` first, then `move` to end your turn."
        )

    def _ablation_opening(self, started: str, board_ascii: str, fen: str, legal_moves: list[str]) -> str:
        cfg = self._config
        tools = [name for name, on in (("`view_board`", cfg.enable_view_board),
                                       ("`simulate ...`", cfg.enable_simulate),
                                       ("`legal_moves`", cfg.enable_legal_moves)) if on]
        aux = (f"You may {', '.join(tools)}, or `think ...` first, then `move` to end your turn."
               if tools else "You may `think ...` first, then `move` to end your turn.")
        parts = [started, f"Initial board:\n{board_ascii}", f"FEN: {fen}"]
        if cfg.push_state:
            parts.append(f"Legal moves ({len(legal_moves)}): {', '.join(legal_moves)}")
        else:
            parts.append(
                "This is the only time the board is shown automatically. From now on, "
                "each move result reports only what changed (the move, any capture, "
                "check, and material) — no board, FEN, or legal-move list."
                + (" You can request them with the commands listed in the rules."
                   if cfg.any_query_tool else " Track the position yourself."))
        parts.append(_REPL_ACT + aux)
        return "\n\n".join(parts)

    def set_pending_opponent_info(self, info: str) -> None:
        self._pending_opponent_info = info

    def flush_pending_opponent_info(self) -> None:
        """Deliver the buffered opponent-move notice as the turn's first message."""
        if self._pending_opponent_info is None:
            return
        self._messages.append({"role": "user", "content": self._pending_opponent_info})
        self._pending_opponent_info = None

    def add_assistant_turn(self, raw_text: str | None, thinking: str | None = None,
                           action: str = "", params: dict | None = None) -> None:
        """Append the agent's verbatim reply, or a reconstructed command for
        turns the environment resolves on the agent's behalf."""
        content = raw_text if raw_text is not None else self._render_command_text(
            thinking, action, params or {})
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, feedback: str) -> None:
        content = feedback
        if self._pending_opponent_info:
            content += f"\n\n{self._pending_opponent_info}"
            self._pending_opponent_info = None
        self._messages.append({"role": "user", "content": content})

    def add_text_exchange(self, user_text: str, assistant_text: str) -> None:
        """Record a reply that carried no usable command, and the re-prompt."""
        if assistant_text:
            self._messages.append({"role": "assistant", "content": assistant_text})
        self._messages.append({"role": "user", "content": user_text})

    @staticmethod
    def _render_command_text(thinking: str | None, action: str, args: dict) -> str:
        cmd = action
        if action == "move" and args.get("from_to"):
            cmd = f"move {args['from_to']}"
        elif action == "simulate" and args.get("moves"):
            cmd = "simulate " + " ".join(args["moves"])
        elif action == "think" and args.get("reasoning"):
            cmd = f"think {args['reasoning']}"
        prefix = f"{thinking}\n\n" if thinking else ""
        return f"{prefix}```bash\n{cmd}\n```"

    def to_serializable(self) -> list[dict]:
        return json.loads(json.dumps(self._messages, ensure_ascii=False))
