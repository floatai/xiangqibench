import json

from conftest import GOLDEN
from xiangqibench.cli import main
from xiangqibench.replay import to_archived_wording


def test_modes(capsys):
    assert main(["modes"]) == 0
    out = capsys.readouterr().out
    assert "sighted" in out and "repl-abl-R-T" in out


def test_prompt_matches_golden(capsys):
    golden = json.loads((GOLDEN / "prompts" / "repl-blind.json").read_text("utf-8"))
    assert main(["prompt", "--mode", "restricted", "--case", golden["case_id"]]) == 0
    out = to_archived_wording("repl-blind", capsys.readouterr().out)
    assert golden["system"] in out and golden["opening"] in out


def test_cases(capsys):
    assert main(["cases", "--split", "ablation-gemini-3.1-pro"]) == 0
    assert "20 cases" in capsys.readouterr().out


def test_init_and_bad_config(tmp_path, capsys):
    path = tmp_path / "b.yaml"
    assert main(["init", str(path)]) == 0
    assert main(["init", str(path)]) == 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: {name: m, colour: red}\n")
    assert main(["run", "-c", str(bad)]) == 2
    assert "unknown key" in capsys.readouterr().err


def test_score_on_golden_trials(tmp_path, capsys):
    import gzip
    import shutil

    for src in (GOLDEN / "trials").glob("*.json.gz"):
        with gzip.open(src, "rb") as fin, open(tmp_path / src.name[:-3], "wb") as fout:
            shutil.copyfileobj(fin, fout)
    assert main(["score", str(tmp_path), "--split", "", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {r["setting"] for r in data["rows"]} >= {"repl-sighted", "repl-blind"}
