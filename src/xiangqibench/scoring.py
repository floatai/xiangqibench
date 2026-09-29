"""Trial classification and leaderboard metrics.

A trial is ``pass`` if the agent delivers mate, ``fail`` if it loses, is
mated, draws, forfeits by invalid moves, by never issuing a command, or by
exhausting a turn's action budget without moving, or reaches the ply cap, and ``None`` (no verdict, excluded) if it ended for an
infrastructure reason. Per (model, setting, case) cell the earliest ``n``
scored trials are used. pass@k is the unbiased estimator of Chen et al.
(2021); pass^k = C(c, k) / C(n, k) is the probability that k draws without
replacement all succeed. Intervals are 95% case-bootstrap percentiles.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np

LOSS_MARKERS = ("invalid_action_limit", "no_command", "no_tool_call", "action_budget")
ABORT_MARKERS = ("api_error", "api_skipped", "aborted", "exception", "timeout")
N_BOOT = 4000
BOOT_SEED = 2027


def _endgame(rec: dict) -> dict:
    return rec.get("endgame") or {}


def classify_record(rec: dict) -> str | None:
    """Return ``'pass'``, ``'fail'``, or ``None`` for a trial without a verdict."""
    if not rec:
        return None
    eg = _endgame(rec)
    result = rec.get("result") or {}
    winner = eg.get("winner") or result.get("winner")
    reason = str(eg.get("termination_reason") or result.get("reason") or "").lower()
    outcome = str(eg.get("outcome") or "").lower()

    aborted = any(mk in reason for mk in ABORT_MARKERS) or any(mk in outcome for mk in ABORT_MARKERS)
    if aborted and not any(mk in reason for mk in LOSS_MARKERS):
        return None
    if outcome == "turn_cap":
        return "fail"
    if not winner or outcome in ("unfinished", "incomplete"):
        return None
    challenger = eg.get("challenger") if eg.get("challenger") in ("red", "black") else "red"
    won = outcome == "challenger_win" or str(winner).lower() == challenger
    return "pass" if won else "fail"


def pass_at_k(successes: int, k: int, n: int) -> float:
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}]")
    failures = n - successes
    return 1.0 if failures < k else 1.0 - comb(failures, k) / comb(n, k)


def pass_hat_k(successes: int, k: int, n: int) -> float:
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}]")
    return 0.0 if successes < k else comb(successes, k) / comb(n, k)


def bootstrap_mean(values: Iterable[float], *, n_boot: int = N_BOOT,
                   seed: int = BOOT_SEED) -> tuple[float, tuple[float | None, float | None]]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), (None, None)
    observed = float(arr.mean())
    if arr.size < 2:
        return observed, (None, None)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return observed, (float(low), float(high))


@dataclass(frozen=True)
class Trial:
    model: str
    setting: str
    case_id: str
    status: str
    started_at: float
    path: str
    plies: int
    reason: str
    standard: bool


def iter_record_files(paths: Iterable[str | os.PathLike]) -> Iterator[Path]:
    for root in paths:
        root = Path(root)
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(p for p in root.rglob("*.json") if p.is_file())


def load_trials(paths: Iterable[str | os.PathLike]) -> tuple[list[Trial], int]:
    """Load scored trials from record files or directories.

    Returns ``(trials, excluded)`` where ``excluded`` counts trial records
    without a verdict. Files that are not trial records are ignored.
    """
    trials, excluded = [], 0
    for path in iter_record_files(paths):
        try:
            rec = json.loads(path.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        eg = _endgame(rec) if isinstance(rec, dict) else {}
        case_id = (eg.get("case") or {}).get("id") or eg.get("case_id")
        if not case_id or not eg.get("model"):
            continue
        status = classify_record(rec)
        if status is None:
            excluded += 1
            continue
        started = eg.get("started_at")
        trials.append(Trial(
            model=str(eg["model"]),
            setting=str(eg.get("setting") or "repl-sighted"),
            case_id=str(case_id),
            status=status,
            started_at=float(started) if isinstance(started, (int, float)) else float("inf"),
            path=str(path),
            plies=int(eg.get("plies") or len(rec.get("moves") or [])),
            reason=str(eg.get("termination_reason") or ""),
            standard=bool(eg.get("standard", True)),
        ))
    return trials, excluded


@dataclass
class Row:
    model: str
    setting: str
    n_cases: int
    trials_per_case: int
    pass_at: dict[int, float] = field(default_factory=dict)
    pass_at_ci: dict[int, tuple] = field(default_factory=dict)
    pass_hat: dict[int, float] = field(default_factory=dict)
    pass_hat_ci: dict[int, tuple] = field(default_factory=dict)
    solved: int = 0
    invalid_rate: float = 0.0
    incomplete_cases: list[str] = field(default_factory=list)
    standard: bool = True

    def to_dict(self) -> dict:
        return {
            "model": self.model, "setting": self.setting, "n_cases": self.n_cases,
            "trials_per_case": self.trials_per_case,
            "pass_at_k": {str(k): v for k, v in self.pass_at.items()},
            "pass_at_k_95ci": {str(k): v for k, v in self.pass_at_ci.items()},
            "pass_hat_k": {str(k): v for k, v in self.pass_hat.items()},
            "pass_hat_k_95ci": {str(k): v for k, v in self.pass_hat_ci.items()},
            "solved": self.solved, "invalid_rate": self.invalid_rate,
            "incomplete_cases": self.incomplete_cases, "standard": self.standard,
        }


def score(trials: Iterable[Trial], *, case_ids: Iterable[str] | None = None,
          n: int = 3) -> list[Row]:
    """Aggregate trials into one row per (model, setting).

    ``case_ids`` fixes the case set (default: every case seen for that row).
    Cells with fewer than ``n`` scored trials are listed in
    ``incomplete_cases`` and scored with the trials they have.
    """
    cells: dict[tuple[str, str], dict[str, list[Trial]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        cells[(t.model, t.setting)][t.case_id].append(t)
    fixed = list(case_ids) if case_ids is not None else None

    rows = []
    for (model, setting), by_case in sorted(cells.items()):
        ids = fixed if fixed is not None else sorted(by_case)
        selected = {cid: sorted(by_case.get(cid, []), key=lambda t: (t.started_at, t.path))[:n]
                    for cid in ids}
        row = Row(model=model, setting=setting, n_cases=len(ids), trials_per_case=n)
        row.incomplete_cases = [cid for cid, ts in selected.items() if len(ts) < n]
        flat = [t for ts in selected.values() for t in ts]
        row.standard = all(t.standard for t in flat)
        row.solved = sum(any(t.status == "pass" for t in ts) for ts in selected.values())
        row.invalid_rate = (sum(any(m in t.reason for m in LOSS_MARKERS) for t in flat) / len(flat)
                            if flat else 0.0)
        for k in range(1, n + 1):
            at, hat = [], []
            for ts in selected.values():
                m = len(ts)
                c = sum(t.status == "pass" for t in ts)
                at.append(pass_at_k(c, k, m) if m >= k else float(c > 0))
                hat.append(pass_hat_k(c, k, m) if m >= k else 0.0)
            row.pass_at[k], row.pass_at_ci[k] = bootstrap_mean(at, seed=BOOT_SEED + 100 * k)
            if k == 1:
                row.pass_hat[k], row.pass_hat_ci[k] = row.pass_at[k], row.pass_at_ci[k]
            else:
                row.pass_hat[k], row.pass_hat_ci[k] = bootstrap_mean(
                    hat, seed=BOOT_SEED + 400 + 100 * k)
        rows.append(row)
    rows.sort(key=lambda r: (r.setting, -r.pass_at.get(1, 0.0), r.model))
    return rows


def _pct(v: float | None) -> str:
    return "–" if v is None or v != v else f"{100 * v:.1f}"


def format_markdown(rows: list[Row]) -> str:
    out = []
    for setting in sorted({r.setting for r in rows}):
        sub = [r for r in rows if r.setting == setting]
        n = sub[0].trials_per_case
        head = ["Model", "pass@1 (95% CI)"] + [f"pass@{k}" for k in range(2, n + 1)] \
            + [f"pass^{k}" for k in range(2, n + 1)] + ["Solved", "Invalid", "Notes"]
        out += [f"### {setting}", "", "| " + " | ".join(head) + " |",
                "|" + "---|" * len(head)]
        for r in sub:
            lo, hi = r.pass_at_ci.get(1, (None, None))
            notes = []
            if r.incomplete_cases:
                notes.append(f"{len(r.incomplete_cases)} incomplete")
            if not r.standard:
                notes.append("non-standard")
            ci = f" ({_pct(lo)}–{_pct(hi)})" if lo is not None and hi is not None else ""
            cells = [r.model, _pct(r.pass_at[1]) + ci]
            cells += [_pct(r.pass_at[k]) for k in range(2, n + 1)]
            cells += [_pct(r.pass_hat[k]) for k in range(2, n + 1)]
            cells += [f"{r.solved}/{r.n_cases}", _pct(r.invalid_rate), ", ".join(notes)]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)
