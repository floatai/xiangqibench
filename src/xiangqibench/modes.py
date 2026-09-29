"""Evaluation modes.

``sighted`` and ``restricted`` are the two observation settings of the paper
and the leaderboard. ``S``, ``S-NT``, ``R-T`` and ``R`` are the observation
ablation arms: they cross state push (board, FEN, and legal moves attached to
every result) with on-demand query tools (``view_board``, ``simulate``,
``legal_moves``), and share one prompt that describes the enabled channels.
"""

from __future__ import annotations

from dataclasses import dataclass

from xiangqibench.harness.config import HarnessConfig


@dataclass(frozen=True)
class Mode:
    name: str
    setting: str
    summary: str
    push_state: bool
    query_tools: bool
    blindfold: bool = False
    obs_arm: str | None = None

    @property
    def is_ablation(self) -> bool:
        return self.obs_arm is not None

    def harness_config(self, **budgets) -> HarnessConfig:
        return HarnessConfig(
            enable_view_board=self.query_tools,
            enable_simulate=self.query_tools,
            enable_legal_moves=self.query_tools,
            push_state=self.push_state,
            blindfold=self.blindfold,
            obs_arm=self.obs_arm,
            **budgets,
        )


MODES: dict[str, Mode] = {m.name: m for m in (
    Mode("sighted", "repl-sighted",
         "Paper setting. Board, FEN, census and legal moves after every ply; "
         "view_board / simulate / legal_moves available.",
         push_state=True, query_tools=True),
    Mode("restricted", "repl-blind",
         "Paper setting. Starting position shown once, then diff-only feedback; "
         "no query or simulation commands.",
         push_state=False, query_tools=False, blindfold=True),
    Mode("S", "repl-abl-S", "Ablation arm: state push on, query tools on.",
         push_state=True, query_tools=True, obs_arm="S"),
    Mode("S-NT", "repl-abl-S-NT", "Ablation arm: state push on, query tools off.",
         push_state=True, query_tools=False, obs_arm="S-NT"),
    Mode("R-T", "repl-abl-R-T", "Ablation arm: state push off, query tools on.",
         push_state=False, query_tools=True, obs_arm="R-T"),
    Mode("R", "repl-abl-R", "Ablation arm: state push off, query tools off.",
         push_state=False, query_tools=False, obs_arm="R"),
)}

PAPER_MODES = ("sighted", "restricted")
ABLATION_MODES = ("S", "S-NT", "R-T", "R")
_ALIASES = {"blind": "restricted", "blindfold": "restricted"}


def get_mode(name: str) -> Mode:
    key = _ALIASES.get(name, name)
    if key not in MODES:
        by_setting = {m.setting: m for m in MODES.values()}
        if key in by_setting:
            return by_setting[key]
        raise KeyError(f"unknown mode {name!r}; choose from {', '.join(MODES)}")
    return MODES[key]
