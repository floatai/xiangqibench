"""External run configuration (YAML), with environment interpolation.

Values of the form ``${VAR}`` or ``${VAR:-default}`` are expanded from the
environment, so API keys never need to be written into the file. Unknown keys
are rejected to catch typos. Command-line flags override file values.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xiangqibench.modes import get_mode

STANDARD_BUDGETS: dict[str, int] = {
    "max_turns": 40,
    "max_actions_per_turn": 8,
    "max_invalid_actions_per_turn": 3,
    "max_simulate_per_turn": 1,
    "max_simulate_steps": 8,
    "max_view_board_per_turn": 1,
    "max_no_command_per_turn": 5,
}

PROVIDERS = ("openai", "azure", "openai-responses", "anthropic")

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    """How to reach the model under evaluation."""

    name: str = ""
    provider: str = "openai"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    api_version: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 600.0
    max_retries: int = 2
    prompt_cache: bool = False

    @property
    def model_id(self) -> str:
        return self.model or self.name

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class DefenderConfig:
    backend: str = "pikafish"
    engine_path: str | None = None
    nnue: str | None = None
    depth: int = 18
    threads: int = 1
    hash_mb: int = 256
    fallback_depth: int = 5


@dataclass
class CasesConfig:
    split: str = "main"
    ids: list[str] | None = None
    path: str | None = None
    limit: int | None = None


@dataclass
class RunConfig:
    mode: str = "sighted"
    trials: int = 3
    max_extra_attempts: int = 3
    workers: int = 4
    output_dir: str = "runs"
    api_attempts: int = 4
    api_backoff_s: float = 2.0
    verbose: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    defender: DefenderConfig = field(default_factory=DefenderConfig)
    cases: CasesConfig = field(default_factory=CasesConfig)
    run: RunConfig = field(default_factory=RunConfig)
    budgets: dict[str, int] = field(default_factory=dict)

    def effective_budgets(self) -> dict[str, int]:
        return {**STANDARD_BUDGETS, **self.budgets}

    @property
    def is_standard(self) -> bool:
        d, ref = self.defender, DefenderConfig()
        search = ("backend", "depth", "threads", "hash_mb", "fallback_depth")
        return (self.effective_budgets() == STANDARD_BUDGETS
                and all(getattr(d, k) == getattr(ref, k) for k in search))

    def validate(self) -> Config:
        get_mode(self.run.mode)
        if not self.model.name:
            raise ConfigError("model.name is required")
        if self.model.provider not in PROVIDERS:
            raise ConfigError(f"model.provider must be one of {PROVIDERS}")
        if self.defender.backend not in ("pikafish", "rule"):
            raise ConfigError("defender.backend must be 'pikafish' or 'rule'")
        unknown = set(self.budgets) - set(STANDARD_BUDGETS)
        if unknown:
            raise ConfigError(f"unknown budgets: {', '.join(sorted(unknown))}")
        if any(int(v) < 1 for v in self.budgets.values()):
            raise ConfigError("budgets must be positive integers")
        if self.run.trials < 1 or self.run.workers < 1:
            raise ConfigError("run.trials and run.workers must be >= 1")
        return self

    def to_dict(self, redact: bool = True) -> dict:
        data = dataclasses.asdict(self)
        if redact and data["model"].get("api_key"):
            data["model"]["api_key"] = "***"
        return data


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            env = os.environ.get(m.group(1))
            if env is not None:
                return env
            if m.group(2) is not None:
                return m.group(2)
            raise ConfigError(f"environment variable {m.group(1)} is not set")
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _build(cls, data: dict | None, where: str):
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a mapping")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(sorted(unknown))}")
    return cls(**data)


def config_from_dict(data: dict) -> Config:
    data = _expand_env(data or {})
    top = {"model", "defender", "cases", "run", "budgets", "mode"}
    unknown = set(data) - top
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {', '.join(sorted(unknown))}")
    model = data.get("model")
    if isinstance(model, str):
        model = {"name": model}
    run = dict(data.get("run") or {})
    if "mode" in data:
        run.setdefault("mode", data["mode"])
    return Config(
        model=_build(ModelConfig, model, "model"),
        defender=_build(DefenderConfig, data.get("defender"), "defender"),
        cases=_build(CasesConfig, data.get("cases"), "cases"),
        run=_build(RunConfig, run, "run"),
        budgets=dict(data.get("budgets") or {}),
    )


def load_config(path: str | os.PathLike | None) -> Config:
    if path is None:
        return Config()
    import yaml

    text = Path(path).read_text("utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return config_from_dict(data)
