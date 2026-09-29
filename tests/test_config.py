import pytest

from xiangqibench.cases import available_splits, load_cases
from xiangqibench.config import ConfigError, config_from_dict, load_config
from xiangqibench.modes import MODES, get_mode


def test_modes_registry():
    assert list(MODES) == ["sighted", "restricted", "S", "S-NT", "R-T", "R"]
    assert get_mode("blind").name == "restricted"
    assert get_mode("repl-abl-R-T").name == "R-T"
    cfg = get_mode("restricted").harness_config()
    assert cfg.blindfold and not cfg.push_state and not cfg.any_query_tool
    cfg = get_mode("R-T").harness_config()
    assert cfg.obs_arm == "R-T" and not cfg.push_state and cfg.enable_simulate
    with pytest.raises(KeyError):
        get_mode("nope")


def test_env_interpolation(monkeypatch):
    monkeypatch.setenv("BA_MODEL", "my-model")
    cfg = config_from_dict({"mode": "R", "model": {"name": "${BA_MODEL}", "base_url": "${BA_URL:-http://x}"}})
    assert cfg.model.name == "my-model"
    assert cfg.model.base_url == "http://x"
    assert cfg.run.mode == "R"
    with pytest.raises(ConfigError):
        config_from_dict({"model": {"name": "${BA_UNSET_VAR}"}})


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError):
        config_from_dict({"model": {"name": "m", "temprature": 1}})
    with pytest.raises(ConfigError):
        config_from_dict({"modle": {}})


def test_standard_flag():
    cfg = config_from_dict({"model": "m"}).validate()
    assert cfg.is_standard
    cfg = config_from_dict({"model": "m", "budgets": {"max_turns": 60}}).validate()
    assert not cfg.is_standard
    with pytest.raises(ConfigError):
        config_from_dict({"model": "m", "budgets": {"max_moves": 60}}).validate()


def test_example_config_loads(tmp_path):
    from importlib import resources

    path = tmp_path / "c.yaml"
    path.write_text(resources.files("xiangqibench").joinpath("example_config.yaml").read_text("utf-8"))
    cfg = load_config(path).validate()
    assert cfg.model.provider == "openai" and cfg.is_standard


def test_splits():
    assert len(load_cases()) == 119
    assert set(available_splits()) == {"main", "ablation-gemini-3.1-pro", "ablation-gpt-5.5"}
    assert len(load_cases("ablation-gpt-5.5")) == 21
    with pytest.raises(KeyError):
        load_cases(ids=["missing"])
