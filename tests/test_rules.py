import pytest

from xiangqibench.cases import load_cases
from xiangqibench.harness.adapter import XiangqiAdapter, material_summary, piece_at
from xiangqibench.rules import XiangqiEnv

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def test_opening_has_44_moves():
    env = XiangqiEnv()
    env.reset()
    assert len(env.get_legal_moves()) == 44
    assert env.get_current_player() == "red"


def test_every_case_is_a_legal_position_with_moves():
    for case in load_cases():
        game = XiangqiAdapter(case.fen)
        game.reset()
        assert not game.is_game_over(), case.id
        assert game.current_player() == case.challenger
        assert case.first_winning_move in game.get_legal_moves(), case.id


def test_illegal_move_is_rejected_without_state_change():
    game = XiangqiAdapter(START)
    game.reset()
    fen = game.get_fen()
    result = game.make_move("a0a5")
    assert not result.success
    assert game.get_fen() == fen


def test_move_description_and_capture():
    game = XiangqiAdapter(START)
    game.reset()
    result = game.make_move("h2h9")
    assert result.success
    assert result.description == "Red Cannon (炮) h2→h9 captures Horse (马)"
    assert result.captured_piece == "Horse (马)"


def test_checkmate_ends_game_for_mover():
    # Two-rook mate: a8 covers row 8, b9 checks along row 9.
    game = XiangqiAdapter("4k4/R8/9/9/9/9/9/9/9/1R1K5 w")
    game.reset()
    result = game.make_move("b0b9")
    assert result.success and result.in_check
    assert result.game_over
    assert result.outcome["winner"] == "red"
    assert result.outcome["reason"] == "checkmate"


def test_piece_at_and_material():
    assert piece_at(START, 4, 0) == "K"
    assert piece_at(START, 4, 9) == "k"
    assert piece_at(START, 4, 5) is None
    assert material_summary(START).endswith("Balance: even (nominal)")


def test_census_lists_every_piece():
    game = XiangqiAdapter(START)
    game.reset()
    census = game.get_piece_census()
    assert "King (帅) e0" in census and "King (将) e9" in census
    assert census.count(",") == 18


@pytest.mark.parametrize("forfeiter,winner", [("red", "black"), ("black", "red")])
def test_forfeit(forfeiter, winner):
    game = XiangqiAdapter(START)
    game.reset()
    game.forfeit(forfeiter, "resignation")
    assert game.is_game_over()
    assert game.get_outcome()["winner"] == winner
