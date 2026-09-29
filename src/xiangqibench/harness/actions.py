"""Execution of agent commands against the game adapter.

Every feedback string here is part of the benchmark interface: it is exactly
what the agent reads, and conformance tests compare it with archived trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xiangqibench.harness.adapter import XiangqiAdapter
    from xiangqibench.harness.config import HarnessConfig
    from xiangqibench.harness.simulator import Simulator


@dataclass
class ActionResult:
    success: bool
    action: str
    feedback: str
    turn_complete: bool = False
    legal_moves: list[str] | None = None
    game_over: bool = False
    winner: str | None = None
    termination_reason: str | None = None
    extra: dict | None = None


class ActionExecutor:
    """Validates and executes one agent command."""

    TURN_ENDING_ACTIONS = frozenset({"move", "resign", "accept_draw"})
    KNOWN_ACTIONS = frozenset({"move", "resign", "view_board", "simulate",
                               "offer_draw", "accept_draw", "get_history", "get_legal_moves",
                               "think"})

    def __init__(self, game: XiangqiAdapter, config: HarnessConfig, simulator: Simulator):
        self._game = game
        self._config = config
        self._simulator = simulator

    def execute(self, player: str, action: str, params: dict[str, Any],
                turn_action_counts: dict[str, int]) -> ActionResult:
        if action not in self.KNOWN_ACTIONS:
            return ActionResult(
                success=False, action=action,
                feedback=f"Unknown action: '{action}'. Available: {', '.join(sorted(self.KNOWN_ACTIONS))}.",
            )
        if self._game.is_game_over():
            outcome = self._game.get_outcome() or {}
            return ActionResult(
                success=False, action=action, feedback="Game is already over.",
                game_over=True, winner=outcome.get("winner"),
                termination_reason=outcome.get("reason"),
            )
        return getattr(self, f"_exec_{action}")(player, params, turn_action_counts)

    def _no_push_recovery_hint(self) -> str:
        if self._config.any_query_tool:
            return ("The board is not shown automatically — use view_board or "
                    "legal_moves if you need to check the position.")
        return "You cannot see the board — recompute the position from the move history."

    def _exec_move(self, player: str, params: dict, counts: dict) -> ActionResult:
        diff_only = not self._config.push_state
        from_to = params.get("from_to", "").strip().lower()
        if not from_to or len(from_to) != 4:
            if diff_only:
                return ActionResult(
                    success=False, action="move",
                    feedback=(
                        f"Invalid action: bad move format '{from_to}'. Must be 4 "
                        f"characters in ICCS format (e.g. 'h2e2'). Your turn is NOT "
                        f"over — move again. 3 invalid attempts in one turn lose the game."
                    ),
                )
            legal = self._game.get_legal_moves()
            return ActionResult(
                success=False, action="move",
                feedback=(
                    f"Invalid action: bad move format '{from_to}'. "
                    f"Must be 4 characters in ICCS format (e.g. 'h2e2').\n"
                    f"Your turn is NOT over — pick a legal move and move again. "
                    f"3 invalid attempts in one turn lose the game.\n"
                    f"Legal moves ({len(legal)}): {', '.join(legal)}"
                ),
                legal_moves=legal,
            )

        result = self._game.make_move(from_to)
        if not result.success:
            if diff_only:
                return ActionResult(
                    success=False, action="move",
                    feedback=(
                        f"Invalid action: illegal move '{from_to}'. Reason: "
                        f"{result.error or 'move not in legal list'}. "
                        f"{self._no_push_recovery_hint()} Your turn is NOT over — move again. "
                        f"3 invalid attempts in one turn lose the game."
                    ),
                )
            legal = self._game.get_legal_moves()
            return ActionResult(
                success=False, action="move",
                feedback=(
                    f"Invalid action: illegal move '{from_to}'. Reason: {result.error}\n"
                    f"Your turn is NOT over — pick one of the legal moves below and "
                    f"move again. 3 invalid attempts in one turn lose the game.\n"
                    f"Legal moves ({len(legal)}): {', '.join(legal)}"
                ),
                legal_moves=legal,
            )

        lines = [f"Move executed: {from_to}"]
        if result.description:
            lines.append(f"Result: {result.description}")
        if result.captured_piece:
            lines.append(f"Capture: you captured {result.captured_piece}.")
        lines.append(f"Check: {'Yes — you are giving check!' if result.in_check else 'No'}")
        if result.material_summary:
            lines.append(f"Material — {result.material_summary}")
        if not diff_only:
            lines.append(f"Board after your move:\n{self._game.get_board_ascii()}")
            lines.append(f"FEN: {self._game.get_fen()}")

        rep, max_rep = result.repetition_count, result.max_repetitions or 0
        if rep >= 2 and max_rep and not result.game_over:
            lines.append(
                f"⚠ Repetition watch: this exact position (board + side to move) "
                f"has now occurred {rep}/{max_rep} times. At {max_rep} the game "
                f"ends — a neutral repetition is a draw, but a side giving "
                f"perpetual check/chase loses (长将/长捉作负). Vary your move unless "
                f"you are deliberately forcing a draw."
            )

        extra = {
            "move": from_to,
            "description": result.description,
            "captured_piece": result.captured_piece,
            "repetition_count": result.repetition_count,
        }
        if result.game_over and result.outcome:
            reason = result.outcome.get("detail", result.outcome.get("reason", ""))
            winner = result.outcome.get("winner", "")
            lines.append(f"Game over: {winner} wins — {reason}")
            return ActionResult(
                success=True, action="move", feedback="\n".join(lines),
                turn_complete=True, game_over=True,
                winner=winner, termination_reason=result.outcome.get("reason"),
                extra=extra,
            )
        if diff_only:
            return ActionResult(success=True, action="move", feedback="\n".join(lines),
                                turn_complete=True, extra=extra)
        legal = self._game.get_legal_moves()
        lines.append(f"Legal moves ({len(legal)}): {', '.join(legal)}")
        return ActionResult(success=True, action="move", feedback="\n".join(lines),
                            turn_complete=True, legal_moves=legal, extra=extra)

    def _exec_resign(self, player: str, params: dict, counts: dict) -> ActionResult:
        winner = self._opponent(player)
        self._game.forfeit(player, "resignation")
        return ActionResult(
            success=True, action="resign",
            feedback=f"You resigned. Game over.\nWinner: {winner}",
            turn_complete=True, game_over=True,
            winner=winner, termination_reason="resignation",
        )

    def _exec_view_board(self, player: str, params: dict, counts: dict) -> ActionResult:
        if not self._config.enable_view_board:
            return ActionResult(success=False, action="view_board",
                                feedback="view_board is disabled in this game configuration.")
        used = counts.get("view_board", 0)
        if used >= self._config.max_view_board_per_turn:
            return ActionResult(
                success=False, action="view_board",
                feedback=f"You have already used view_board {used} time(s) this turn. Please make a move.",
            )
        census = self._game.get_piece_census()
        census_block = f"{census}\n\n" if census else ""
        feedback = (
            f"Current board:\n\n{self._game.get_board_ascii()}\n\n"
            f"{census_block}"
            f"FEN: {self._game.get_fen()}\n"
            f"Uppercase = Red, lowercase = Black\n"
            f"Current turn: {self._game.current_player()}\n"
            f"Move count: {self._game.get_move_count()}"
        )
        return ActionResult(success=True, action="view_board", feedback=feedback)

    def _exec_simulate(self, player: str, params: dict, counts: dict) -> ActionResult:
        if not self._config.enable_simulate:
            return ActionResult(success=False, action="simulate",
                                feedback="simulate is disabled in this game configuration.")
        used = counts.get("simulate", 0)
        if used >= self._config.max_simulate_per_turn:
            return ActionResult(
                success=False, action="simulate",
                feedback=f"You have already used simulate {used} time(s) this turn. Please make a move.",
            )
        moves = params.get("moves", [])
        if not moves or not isinstance(moves, list):
            return ActionResult(success=False, action="simulate",
                                feedback="simulate requires a non-empty 'moves' array.")
        sim = self._simulator.run(self._game, moves[:self._config.max_simulate_steps])
        return ActionResult(success=True, action="simulate",
                            feedback=sim.to_feedback(), extra=sim.to_dict())

    def _exec_offer_draw(self, player: str, params: dict, counts: dict) -> ActionResult:
        return ActionResult(success=False, action="offer_draw", feedback="Draw offers are disabled.")

    def _exec_accept_draw(self, player: str, params: dict, counts: dict) -> ActionResult:
        return ActionResult(success=False, action="accept_draw",
                            feedback="Draws are disabled in this game configuration.")

    def _exec_get_history(self, player: str, params: dict, counts: dict) -> ActionResult:
        history = self._game.get_move_history()
        if not history:
            return ActionResult(success=True, action="get_history", feedback="No moves played yet.")
        p1, p2 = self._game.player_names()
        lines = [f"Move history ({len(history)} moves):"]
        lines += [f"  {i + 1}. {p1 if i % 2 == 0 else p2}: {m}" for i, m in enumerate(history)]
        return ActionResult(success=True, action="get_history", feedback="\n".join(lines))

    def _exec_get_legal_moves(self, player: str, params: dict, counts: dict) -> ActionResult:
        if not self._config.enable_legal_moves:
            where = "blindfold mode" if self._config.blindfold else "this game configuration"
            return ActionResult(success=False, action="get_legal_moves",
                                feedback=f"get_legal_moves is disabled in {where}.")
        legal = self._game.get_legal_moves()
        return ActionResult(success=True, action="get_legal_moves",
                            feedback=f"Legal moves ({len(legal)}): {', '.join(legal)}",
                            legal_moves=legal)

    def _exec_think(self, player: str, params: dict, counts: dict) -> ActionResult:
        if not params.get("reasoning", "").strip():
            return ActionResult(success=False, action="think",
                                feedback="Please provide your reasoning in the 'reasoning' parameter.")
        return ActionResult(
            success=True, action="think",
            feedback="Reasoning recorded. Now call a tool to act (e.g. move, simulate, view_board).",
        )

    def force_forfeit(self, player: str, reason: str = "action_limit") -> ActionResult:
        """End the trial with ``player`` losing after a per-turn budget is exhausted."""
        winner = self._opponent(player)
        self._game.forfeit(player, reason)
        return ActionResult(
            success=True, action="move",
            feedback=(
                f"Turn could not be resolved ({reason}) — you failed to make a "
                f"legal move within the allowed attempts. You FORFEIT.\n"
                f"Winner: {winner}"
            ),
            turn_complete=True, game_over=True,
            winner=winner, termination_reason=f"forfeit_{reason}",
            extra={"forfeit": True, "reason": reason},
        )

    def _opponent(self, player: str) -> str:
        p1, p2 = self._game.player_names()
        return p2 if player == p1 else p1
