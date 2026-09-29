import pytest

from xiangqibench.harness.protocol import parse_action


@pytest.mark.parametrize("text,action,params", [
    ("I centralize.\n```bash\nmove h2e2\n```", "move", {"from_to": "h2e2"}),
    ("```\n$ chess move b0c2\n```", "move", {"from_to": "b0c2"}),
    ("```bash\nsim h2e2 h9g7\n```", "simulate", {"moves": ["h2e2", "h9g7"]}),
    ("```bash\nboard\n```", "view_board", {}),
    ("```bash\nmoves\n```", "get_legal_moves", {}),
    ("```bash\nhistory\n```", "get_history", {}),
    ("```bash\nthink rook lift then mate\n```", "think", {"reasoning": "rook lift then mate"}),
    ("```bash\nresign\n```", "resign", {}),
    ("```bash\ndraw\n```", "offer_draw", {}),
    ("```bash\n# comment\n\nmove a0a1\n```", "move", {"from_to": "a0a1"}),
    ("```bash\nfly e0e9\n```", "fly", {}),
])
def test_commands(text, action, params):
    parsed = parse_action(text)
    assert parsed.action == action
    assert parsed.params == params


def test_last_block_wins_and_prose_is_thinking():
    parsed = parse_action("plan\n```bash\nmove a0a1\n```\nactually\n```bash\nmove h2e2\n```")
    assert parsed.params == {"from_to": "h2e2"}
    assert parsed.thinking == "plan\n\nactually"


@pytest.mark.parametrize("text", ["move h2e2", "", "```bash\n\n```"])
def test_missing_command(text):
    assert parse_action(text).action is None
