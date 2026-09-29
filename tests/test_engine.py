import pytest

from xiangqibench.cases import load_cases
from xiangqibench.defender import PikafishDefender, PikafishEngine

pytestmark = pytest.mark.engine

# xq_jianghu_endgames_084 after 1. h5h9: Pikafish rejects the black pawns on
# rows 8 and 1 as unreachable, so the defender must use the rule backend.
UNREACHABLE = "3k1pC2/4P4/9/9/7CR/3n4c/2Pn5/8B/5p3/4K4 b"


def test_engine_is_deterministic(pikafish_path):
    case = load_cases()[0]
    with PikafishEngine(path=pikafish_path, depth=10) as engine:
        first = engine.best_move(case.fen)
        assert first == engine.best_move(case.fen)
        assert engine.info()["engine_id"]


def test_unsupported_position_falls_back_to_rule(pikafish_path):
    with PikafishEngine(path=pikafish_path, depth=8) as engine:
        defender = PikafishDefender("red", engine, fallback_depth=3)
        choice = defender.choose(UNREACHABLE)
        assert choice.move is not None
        assert choice.backend == "rule"
        assert engine.best_move(load_cases()[0].fen) is not None
