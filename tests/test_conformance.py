"""The released harness must reproduce archived trials exactly.

Prompt fixtures hold the system prompt and opening message that were sent in
data collection; trial fixtures are complete archived trials whose agent
replies and defender moves are replayed through the current harness.
"""

import json

import pytest

from conftest import GOLDEN, load_trial_fixture, prompt_fixtures, trial_fixtures
from xiangqibench.cases import load_case
from xiangqibench.config import STANDARD_BUDGETS
from xiangqibench.harness import GameHarness, XiangqiAdapter
from xiangqibench.modes import get_mode
from xiangqibench.replay import replay_record, to_archived_wording, used_reasoning_channel
from xiangqibench.rules import XiangqiEnv

BUDGETS = {k: v for k, v in STANDARD_BUDGETS.items() if k != "max_no_command_per_turn"}


def test_fixtures_cover_every_mode():
    settings = {p.stem for p in prompt_fixtures()}
    assert settings == {"repl-sighted", "repl-blind", "repl-abl-S", "repl-abl-S-NT",
                        "repl-abl-R-T", "repl-abl-R"}
    assert {p.name.split("__")[0] for p in trial_fixtures()} == settings


@pytest.mark.parametrize("path", prompt_fixtures(), ids=lambda p: p.stem)
def test_prompts_match_archive(path):
    golden = json.loads(path.read_text("utf-8"))
    case = load_case(golden["case_id"])
    harness = GameHarness(XiangqiAdapter(case.fen), get_mode(path.stem).harness_config(**BUDGETS))
    messages = harness.build_messages(case.challenger)
    assert to_archived_wording(path.stem, messages[0]["content"]) == golden["system"]
    assert messages[1]["content"] == golden["opening"]


def test_restricted_prompt_uses_paper_wording():
    case = load_case("xq_shi_qing_ya_qu_014")
    harness = GameHarness(XiangqiAdapter(case.fen), get_mode("restricted").harness_config(**BUDGETS))
    system = harness.build_messages(case.challenger)[0]["content"]
    assert "You CANNOT see the board, FEN, or piece census at any point except the initial state." in system


def _explain(result) -> str:
    return (f"first mismatch at message {result.first_mismatch}:\n"
            f"--- archived\n{result.expected}\n--- replayed\n{result.actual}\n{result.notes}")


@pytest.mark.parametrize("path", trial_fixtures(), ids=lambda p: p.name.split(".")[0])
def test_archived_trials_replay_exactly(path):
    result = replay_record(load_trial_fixture(path))
    assert result.ok, _explain(result)


@pytest.mark.parametrize("path", sorted((GOLDEN / "legacy").glob("*__check_flag.json.gz")),
                         ids=lambda p: p.name.split("__")[0])
def test_paper_archive_differs_only_by_check_flag(path, monkeypatch):
    record = load_trial_fixture(path)
    assert record["endgame"]["termination_reason"] == "stalemate"

    fixed = replay_record(record)
    assert not fixed.ok
    assert fixed.verdict_actual[1:2] == ("checkmate",) or "Check: Yes" in (fixed.actual or "")

    monkeypatch.setattr(XiangqiEnv, "is_in_check", lambda self: False)
    legacy = replay_record(record)
    assert legacy.ok, _explain(legacy)


def test_released_restricted_records_replay_with_released_wording():
    from xiangqibench.defender import RuleDefender
    from xiangqibench.llm import ScriptedAgent
    from xiangqibench.runner import play_trial

    case = load_case("xq_shi_qing_ya_qu_014")
    agent = ScriptedAgent(["```bash\nmove d8d9\n```"] + ["```bash\nresign\n```"] * 3)
    _, record = play_trial(case, get_mode("restricted"), agent, RuleDefender(case.challenger))
    assert "except the initial state" in record["per_player_messages"]["red"][0]["content"]
    result = replay_record(record)
    assert result.ok, _explain(result)


def test_replay_infers_budget_missing_from_early_records():
    record = load_trial_fixture(GOLDEN / "legacy" / "repl-sighted__early_no_command_budget.json.gz")
    assert "max_no_command_per_turn" not in record["config"]
    result = replay_record(record)
    assert result.ok, _explain(result)


def test_reasoning_channel_detection_ignores_environment_forfeits():
    def record(thinking, content):
        step = {"llm_response": {"thinking": thinking, "tool_call": {"name": "move"},
                                 "raw_response": {"content": content}}}
        return {"endgame": {"case": {"challenger": "red"}},
                "trajectory": [{"player": "red", "interactions": [step]}]}

    assert used_reasoning_channel(record("My move: f6f0.", "prose without a command"))
    assert not used_reasoning_channel(record("[environment: no_command]", ""))
    assert not used_reasoning_channel(record("plan", "```bash\nmove f6f0\n```"))
