"""Environment-side configuration of one endgame trial."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HarnessConfig:
    """Budgets and observation channels seen by the agent.

    The agent always acts through the REPL command protocol, draws are never
    available, and exhausting a per-turn budget forfeits the trial.

    ``blindfold`` selects the paper's Restricted-observation interface: the
    starting position is shown once, feedback is diff-only, and the query and
    simulation commands are disabled. ``obs_arm`` selects an observation-ablation
    arm, whose single shared prompt describes exactly the channels enabled by
    ``push_state`` and the ``enable_*`` flags.
    """

    max_turns: int = 40
    max_simulate_steps: int = 8
    max_actions_per_turn: int = 8
    max_simulate_per_turn: int = 1
    max_view_board_per_turn: int = 1
    max_invalid_actions_per_turn: int = 3
    enable_view_board: bool = True
    enable_simulate: bool = True
    enable_legal_moves: bool = True
    push_state: bool = True
    blindfold: bool = False
    obs_arm: str | None = None
    record_prompt_snapshots: bool = False

    def __post_init__(self) -> None:
        if self.blindfold and self.obs_arm:
            raise ValueError("blindfold and obs_arm are mutually exclusive")
        if self.blindfold:
            self.enable_view_board = False
            self.enable_simulate = False
            self.enable_legal_moves = False
            self.push_state = False
        for name in ("max_turns", "max_simulate_steps", "max_actions_per_turn",
                     "max_invalid_actions_per_turn"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

    @property
    def any_query_tool(self) -> bool:
        return self.enable_view_board or self.enable_simulate or self.enable_legal_moves

    def prompt_values(self) -> dict[str, int]:
        """Values substituted into ``{placeholder}`` fields of the system prompts."""
        return {
            "max_turns": self.max_turns,
            "max_view_board": self.max_view_board_per_turn,
            "max_simulate": self.max_simulate_per_turn,
            "max_simulate_steps": self.max_simulate_steps,
            "max_actions": self.max_actions_per_turn,
            "max_invalid": self.max_invalid_actions_per_turn,
        }
