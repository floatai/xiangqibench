"""Forward simulation engine.

Clones the game state and replays a provided move sequence, validating each
step without modifying the real game.

Design:
  - Each "step" is one half-move (ply). Steps alternate between the two players.
  - A "round" = 2 steps (one move per side). So max_steps=6 means 3 full rounds.
  - The `moves` list is the caller's OWN line: odd plies (1,3,5...) are the
    caller's moves; even plies (2,4,6...) are the opponent replies the caller
    ASSUMES. There is no engine picking the opponent's move — the caller is
    testing a hypothetical line and supplies both sides.
  - On each valid step the feedback includes: a natural-language description of
    what moved / was captured, check status, FEN after, and the number of legal
    moves for whoever is to move next (so the caller can pick a realistic reply).
  - On an invalid step the feedback includes: the legal moves at that position
    so the caller can retry with a valid move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xiangqibench.harness.adapter import XiangqiAdapter as GameAdapter


@dataclass
class SimStep:
    step: int
    player: str
    move: str
    valid: bool
    in_check: bool = False
    fen_after: str = ""
    legal_moves_count: int = 0
    legal_moves_sample: list[str] = field(default_factory=list)
    game_over: bool = False
    outcome: dict | None = None
    error: str | None = None
    is_own: bool = True               # True = your move; False = assumed opponent move
    next_player: str = ""             # side to move after this ply (whose legal_moves_count is reported)
    description: str = ""             # e.g. "Red Cannon (炮) h2→e2 captures Black Horse (马)"
    captured_piece: str | None = None


@dataclass
class SimulationResult:
    steps_requested: int = 0
    steps_completed: int = 0
    steps: list[SimStep] = field(default_factory=list)
    simulation_game_over: bool = False
    error: str | None = None

    def to_feedback(self) -> str:
        if not self.steps:
            return "Simulation produced no steps."

        valid_count = sum(1 for s in self.steps if s.valid)
        total = len(self.steps)
        rounds = (valid_count + 1) // 2
        header = f"Forward simulation ({valid_count} of {total} steps valid, ~{rounds} round(s)):"

        lines = [
            header,
            "(odd steps = YOUR moves; even steps = the opponent replies YOU assumed)",
        ]
        for s in self.steps:
            role = "your move" if s.is_own else "assumed opponent move"
            desc = s.description or f"{s.player} {s.move}"
            if s.valid:
                check_str = "Yes" if s.in_check else "No"
                # After this ply it is `next_player`'s turn; the legal-move count
                # belongs to whoever is to move next.
                reply_label = f"{s.next_player} legal replies" if s.next_player else "legal replies"
                status = (
                    f"Valid ✓ | {desc} | Check: {check_str} "
                    f"| {reply_label}: {s.legal_moves_count} "
                    f"| FEN: {s.fen_after}"
                )
                if s.game_over and s.outcome:
                    winner = s.outcome.get("winner", "?")
                    reason = s.outcome.get("reason", "?")
                    status += f" — Game over: {winner} wins ({reason})"
            else:
                legal_hint = ""
                if s.legal_moves_sample:
                    sample = s.legal_moves_sample[:15]
                    total_legal = s.legal_moves_count
                    shown = len(sample)
                    legal_hint = (
                        f"\n         Legal moves for {s.next_player or s.player} "
                        f"at this position ({total_legal}): {', '.join(sample)}"
                    )
                    if total_legal > shown:
                        legal_hint += f" ... ({total_legal - shown} more)"
                status = f"Invalid ✗ | Error: {s.error}{legal_hint}"
            lines.append(f"  Step {s.step} ({role}): {s.player} {s.move} — {status}")

        if self.error:
            lines.append(f"Simulation aborted: {self.error}")
        elif self.simulation_game_over:
            last_outcome = self.steps[-1].outcome or {}
            winner = last_outcome.get("winner", "?")
            reason = last_outcome.get("reason", "?")
            lines.append(f"Simulation ended: {winner} wins ({reason}).")
        else:
            lines.append("Simulation complete. Game state: ongoing.")

        lines.append("Note: this simulation does NOT affect the actual game.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "steps_requested": self.steps_requested,
            "steps_completed": self.steps_completed,
            "steps": [
                {
                    "step": s.step,
                    "player": s.player,
                    "move": s.move,
                    "valid": s.valid,
                    "in_check": s.in_check,
                    "is_own": s.is_own,
                    "next_player": s.next_player,
                    "description": s.description,
                    "captured_piece": s.captured_piece,
                    "fen_after": s.fen_after,
                    "legal_moves_count": s.legal_moves_count,
                    "legal_moves_sample": s.legal_moves_sample,
                    "game_over": s.game_over,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "simulation_game_over": self.simulation_game_over,
            "error": self.error,
        }


class Simulator:
    """Run forward simulation on a cloned game adapter.

    max_steps: maximum number of half-moves (plies) to simulate.
               6 steps = 3 full rounds (each side moves 3 times).
    """

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps

    def run(self, game: GameAdapter, moves: list[str]) -> SimulationResult:
        clone = game.clone()
        steps_to_run = min(len(moves), self.max_steps)
        result = SimulationResult(steps_requested=len(moves))

        # The player to move at step 1 is the one running the simulation ("you").
        # All odd plies are yours; even plies are assumed opponent replies.
        initiator = clone.current_player()

        for i in range(steps_to_run):
            move = moves[i]
            player = clone.current_player()

            if clone.is_game_over():
                result.simulation_game_over = True
                break

            move_result = clone.make_move(move)

            step = SimStep(
                step=i + 1,
                player=player,
                move=move,
                valid=move_result.success,
                is_own=(player == initiator),
            )

            if not move_result.success:
                legal = clone.get_legal_moves()
                step.error = move_result.error or f"'{move}' is not a legal move in the simulated position"
                step.legal_moves_count = len(legal)
                step.legal_moves_sample = legal[:20]
                step.fen_after = clone.get_fen()
                step.next_player = player  # still this player's turn (move rejected)
                result.steps.append(step)
                result.steps_completed = i
                result.error = f"Illegal move at step {i + 1}. Subsequent steps not executed."
                return result

            step.in_check = move_result.in_check
            step.game_over = move_result.game_over
            step.outcome = move_result.outcome
            step.fen_after = clone.get_fen()
            step.description = move_result.description or ""
            step.captured_piece = move_result.captured_piece
            step.next_player = clone.current_player()

            if not move_result.game_over:
                legal = clone.get_legal_moves()
                step.legal_moves_count = len(legal)
                step.legal_moves_sample = legal[:10]
            else:
                step.legal_moves_count = 0

            result.steps.append(step)
            result.steps_completed = i + 1

            if move_result.game_over:
                result.simulation_game_over = True
                break

        return result
