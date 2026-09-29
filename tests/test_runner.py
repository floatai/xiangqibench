"""Offline end-to-end runs with scripted agents and the rule-based defender."""

import json
import re

import cchess
import pytest

from xiangqibench.cases import load_cases
from xiangqibench.config import config_from_dict
from xiangqibench.defender import EngineUnavailable, PikafishDefender, RuleDefender
from xiangqibench.defender.search import best_winning_move
from xiangqibench.llm import Completion
from xiangqibench.modes import get_mode
from xiangqibench.runner import ApiFailure, ApiUnusable, play_trial, run_suite
from xiangqibench.scoring import load_trials, score

_FEN_RE = re.compile(r"FEN: (\S+ [wb])")


class MateAgent:
    """Reads the latest FEN from the chat and plays a shortest forced mate."""

    name = "mate-agent"

    def complete(self, messages):
        fen = _FEN_RE.findall("\n".join(m["content"] for m in messages))[-1]
        board = cchess.ChessBoard(fen)
        side = "red" if board.move_player == cchess.RED else "black"
        move = best_winning_move(board, side, 3)
        return Completion(text=f"Mate.\n```bash\nmove {move}\n```")


class Silent:
    name = "silent"

    def complete(self, messages):
        return Completion(text="I am thinking about it.",
                          usage={"prompt_tokens": 10, "completion_tokens": 2})


class HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class Failing:
    name = "failing"

    def __init__(self, status_code):
        self.status_code, self.calls = status_code, 0

    def complete(self, messages):
        self.calls += 1
        raise HttpError(self.status_code)


def short_case():
    return next(c for c in load_cases() if c.win_in_plies == 3)


def test_mate_agent_wins_and_defender_source_is_recorded():
    case = short_case()
    summary, record = play_trial(case, get_mode("sighted"), MateAgent(), RuleDefender(case.challenger))
    assert summary.status == "pass" and summary.outcome == "challenger_win"
    assert summary.plies == 3 and summary.matched_first_move
    house = [t for t in record["trajectory"] if t["player"] != case.challenger]
    assert house and all(t["defender_backend"] == "rule" for t in house)
    assert record["trajectory"][-1]["in_check"]
    assert record["moves"] == [t["move_played"] for t in record["trajectory"]]


class _Engine:
    """Stands in for PikafishEngine; ``best_move`` returns or raises ``result``."""

    def __init__(self, result):
        self.result = result

    def best_move(self, fen):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_pikafish_defender_falls_back_only_for_rejected_positions():
    fen = "3k1pC2/4P4/9/9/7CR/3n4c/2Pn5/8B/5p3/4K4 b"
    rejected = PikafishDefender("red", _Engine(None)).choose(fen)
    assert rejected.backend == "rule" and rejected.move is not None
    with pytest.raises(EngineUnavailable):
        PikafishDefender("red", _Engine(EngineUnavailable("dead"))).choose(fen)


def test_no_command_forfeits_after_five_replies():
    case = short_case()
    summary, record = play_trial(case, get_mode("restricted"), Silent(), RuleDefender(case.challenger))
    assert summary.termination_reason == "forfeit_no_command"
    assert summary.status == "fail" and summary.plies == 0
    assert record["total_api_calls"] == 5
    assert record["stats"]["total_tokens"][case.challenger] == {"prompt": 50, "completion": 10}


@pytest.mark.parametrize("status,error,calls", [
    (401, ApiUnusable, 1), (404, ApiUnusable, 1), (400, ApiFailure, 1), (500, ApiFailure, 3),
])
def test_retries_only_errors_that_can_succeed(status, error, calls):
    case, agent = short_case(), Failing(status)
    with pytest.raises(error):
        play_trial(case, get_mode("sighted"), agent, RuleDefender(case.challenger),
                   api_attempts=3, api_backoff_s=0)
    assert agent.calls == calls


def test_run_suite_writes_scores_and_resumes(tmp_path):
    case = short_case()
    cfg = config_from_dict({
        "mode": "S-NT", "model": {"name": "mate-agent"}, "defender": {"backend": "rule"},
        "cases": {"ids": [case.id]}, "run": {"trials": 2, "workers": 1, "output_dir": str(tmp_path)},
    })
    report = run_suite(cfg, agent_factory=MateAgent)
    assert report.completed_cells == 1 and report.trials_played == 2 and report.passes == 2

    files = sorted(tmp_path.rglob("*.json"))
    assert len(files) == 2
    rec = json.loads(files[0].read_text())
    assert rec["endgame"]["setting"] == "repl-abl-S-NT" and not rec["endgame"]["standard"]

    again = run_suite(cfg, agent_factory=MateAgent)
    assert again.trials_played == 0 and again.completed_cells == 1

    trials, excluded = load_trials([tmp_path])
    (row,) = score(trials, n=2)
    assert excluded == 0 and row.pass_at[1] == 1.0 and row.pass_hat[2] == 1.0
