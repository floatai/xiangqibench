"""Re-verify a trial record by replaying it through the current harness.

The agent's archived replies and the defender's archived moves are fed back
into a fresh game; the replay passes when every message the environment sends
to the agent, and the final verdict, reproduce the record exactly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from xiangqibench.cases import EndgameCase
from xiangqibench.defender import DefenderChoice
from xiangqibench.harness.protocol import parse_action
from xiangqibench.llm import Completion
from xiangqibench.modes import get_mode

REASONING_NOTE = "archived command came from the reasoning channel, which records store without its code block"

# System-prompt wording that differs between this release and the prompts sent
# during paper data collection, as (released, archived) pairs per setting.
ARCHIVED_PROMPT_WORDING: dict[str, tuple[tuple[str, str], ...]] = {
    "repl-blind": (("at any point except the initial state.", "at any point."),),
}

NO_COMMAND_PREFIX = "No command detected."
_BUDGET_KEYS = ("max_turns", "max_simulate_steps", "max_actions_per_turn", "max_no_command_per_turn",
                "max_simulate_per_turn", "max_view_board_per_turn", "max_invalid_actions_per_turn")


class _ReplayAgent:
    def __init__(self, name: str, replies: list[str]):
        self.name = name
        self._replies = list(replies)

    def complete(self, messages) -> Completion:
        if not self._replies:
            raise RuntimeError("replay ran out of archived agent replies")
        return Completion(text=self._replies.pop(0))


class _ReplayDefender:
    backend = "replay"

    def __init__(self, moves: list[str]):
        self._moves = list(moves)

    def info(self) -> dict:
        return {"backend": self.backend}

    def close(self) -> None:
        pass

    def choose(self, fen: str) -> DefenderChoice:
        move = self._moves.pop(0) if self._moves else None
        return DefenderChoice(move=move, backend=self.backend, legal_count=0)


def agent_replies(messages: list[dict]) -> list[str]:
    """Recover the agent's replies, including empty ones that left no message."""
    replies = []
    for i, msg in enumerate(messages):
        after_user = i > 0 and messages[i - 1]["role"] != "assistant"
        if msg["role"] == "assistant":
            if not msg["content"].startswith("[environment:"):
                replies.append(msg["content"])
            elif msg["content"].startswith("[environment: no_command]") and after_user:
                # The empty reply that exhausted the no-command budget is only
                # visible through the forfeit marker written in its place.
                replies.append("")
        elif msg["role"] == "user" and msg["content"].startswith(NO_COMMAND_PREFIX) and after_user:
            replies.append("")
    return replies


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def used_reasoning_channel(record: dict) -> bool:
    """Whether an archived command was parsed from the reasoning channel.

    Records keep the reasoning with its code block stripped, so such a reply
    cannot be replayed from the stored messages alone.
    """
    challenger = record["endgame"]["case"]["challenger"]
    for turn in record.get("trajectory") or []:
        if turn.get("player") != challenger:
            continue
        for step in turn.get("interactions") or []:
            response = _as_dict(step.get("llm_response"))
            if str(response.get("thinking") or "").startswith("[environment:"):
                continue
            content = _as_dict(response.get("raw_response")).get("content") or ""
            if response.get("tool_call") and parse_action(content).action is None:
                return True
    return False


def _inferred_no_command_limit(messages: list[dict]) -> int | None:
    """Recover the no-command budget of records that predate its config key.

    Only a ``no_command`` forfeit reveals it: the feedback messages of the final
    streak, plus the reply that triggered the forfeit.
    """
    if not messages or not messages[-2]["content"].startswith("[environment: no_command]"):
        return None
    streak = 1
    for msg in reversed(messages[:-2]):
        if msg["role"] == "assistant":
            continue
        if not msg["content"].startswith(NO_COMMAND_PREFIX):
            break
        streak += 1
    return streak


def to_archived_wording(setting: str, system_prompt: str) -> str:
    """Rewrite a released system prompt into the wording of the paper archive."""
    for released, archived in ARCHIVED_PROMPT_WORDING.get(setting, ()):
        system_prompt = system_prompt.replace(released, archived)
    return system_prompt


@dataclass
class ReplayResult:
    ok: bool
    messages_compared: int
    first_mismatch: int | None = None
    expected: str | None = None
    actual: str | None = None
    verdict_expected: tuple = ()
    verdict_actual: tuple = ()
    notes: list[str] = field(default_factory=list)


def replay_record(record: dict) -> ReplayResult:
    from xiangqibench.runner import play_trial

    eg = record["endgame"]
    case = EndgameCase.from_dict(eg["case"])
    mode = get_mode(eg.get("setting") or "repl-sighted")
    challenger = case.challenger
    expected = record["per_player_messages"][challenger]
    moves = list(record.get("moves") or [])
    legacy_duplicate = (str(eg.get("termination_reason") or "").startswith("forfeit")
                        and len(moves) >= 2 and moves[-1] == moves[-2])
    if legacy_duplicate:
        moves.pop()
    defender_moves = moves[1::2]
    cfg = record.get("config") or {}
    budgets = {k: cfg[k] for k in _BUDGET_KEYS if k in cfg}
    if "max_no_command_per_turn" not in budgets:
        inferred = _inferred_no_command_limit(expected)
        if inferred is not None:
            budgets["max_no_command_per_turn"] = inferred

    agent = _ReplayAgent(eg["model"], agent_replies(expected))
    reasoning = [REASONING_NOTE] if used_reasoning_channel(record) else []
    try:
        _, new = play_trial(case, mode, agent, _ReplayDefender(defender_moves), budgets=budgets,
                            api_attempts=1, api_backoff_s=0.0)
    except Exception as exc:
        return ReplayResult(ok=False, messages_compared=0, notes=[*reasoning, f"replay error: {exc}"])

    actual = list(new["per_player_messages"][challenger])
    if "xiangqibench_version" not in eg and actual and actual[0]["role"] == "system":
        actual[0] = {**actual[0], "content": to_archived_wording(mode.setting, actual[0]["content"])}
    n = min(len(expected), len(actual))
    for i in range(n):
        if expected[i] != actual[i]:
            return ReplayResult(ok=False, messages_compared=i, first_mismatch=i,
                                expected=expected[i].get("content"), actual=actual[i].get("content"),
                                notes=reasoning)
    v_exp = (eg.get("outcome"), eg.get("termination_reason"), len(moves))
    v_act = (new["endgame"]["outcome"], new["endgame"]["termination_reason"], new["endgame"]["plies"])
    ok = len(expected) == len(actual) and v_exp == v_act
    res = ReplayResult(ok=ok, messages_compared=n, verdict_expected=v_exp, verdict_actual=v_act)
    if legacy_duplicate:
        res.notes.append("archived record repeats its last move after a forfeit (fixed in 0.1.0)")
    if len(expected) != len(actual):
        res.first_mismatch = n
        res.notes.append(f"message count {len(expected)} archived vs {len(actual)} replayed")
    return res
