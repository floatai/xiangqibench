import pytest

from xiangqibench.scoring import Trial, classify_record, pass_at_k, pass_hat_k, score


def rec(outcome, reason, winner, challenger="red"):
    return {"endgame": {"outcome": outcome, "termination_reason": reason, "winner": winner,
                        "challenger": challenger}}


@pytest.mark.parametrize("record,expected", [
    (rec("challenger_win", "checkmate", "red"), "pass"),
    (rec("challenger_win", "stalemate", "black", "black"), "pass"),
    (rec("challenger_loss", "checkmate", "black"), "fail"),
    (rec("challenger_loss", "resignation", "black"), "fail"),
    (rec("challenger_loss", "forfeit_invalid_action_limit", "black"), "fail"),
    (rec("challenger_loss", "forfeit_no_command", "black"), "fail"),
    (rec("challenger_loss", "forfeit_action_limit", "black"), "fail"),
    (rec("draw", "perpetual_check", "draw"), "fail"),
    (rec("turn_cap", None, None), "fail"),
    (rec("challenger_loss", "forfeit_action_budget_exhausted", "black"), None),
    (rec("api_skipped", "api_error", None), None),
    ({}, None),
])
def test_classify(record, expected):
    assert classify_record(record) == expected


@pytest.mark.parametrize("c,k,n,at,hat", [
    (0, 1, 3, 0.0, 0.0), (1, 1, 3, 1 / 3, 1 / 3), (3, 3, 3, 1.0, 1.0),
    (1, 2, 3, 2 / 3, 0.0), (2, 2, 3, 1.0, 1 / 3), (2, 3, 3, 1.0, 0.0),
])
def test_pass_k(c, k, n, at, hat):
    assert pass_at_k(c, k, n) == pytest.approx(at)
    assert pass_hat_k(c, k, n) == pytest.approx(hat)


def _trial(case, status, t, model="m", setting="repl-sighted"):
    return Trial(model=model, setting=setting, case_id=case, status=status, started_at=t,
                 path=f"{case}-{t}", plies=5, reason="checkmate", standard=True)


def test_score_uses_earliest_trials_and_flags_incomplete():
    trials = [
        _trial("a", "pass", 1), _trial("a", "fail", 2), _trial("a", "fail", 3), _trial("a", "pass", 4),
        _trial("b", "pass", 1), _trial("b", "pass", 2), _trial("b", "pass", 3),
        _trial("c", "fail", 1),
    ]
    (row,) = score(trials, case_ids=["a", "b", "c"], n=3)
    assert row.pass_at[1] == pytest.approx((1 / 3 + 1 + 0) / 3)
    assert row.pass_hat[3] == pytest.approx(1 / 3)
    assert row.solved == 2
    assert row.incomplete_cases == ["c"]
    lo, hi = row.pass_at_ci[1]
    assert lo <= row.pass_at[1] <= hi
