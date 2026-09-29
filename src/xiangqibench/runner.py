"""Play endgame trials: one agent against the deterministic defender.

:func:`play_trial` runs one game through the harness. :func:`run_suite` runs
every (case, trial) cell of a configuration in parallel, writes one JSON record
per trial, resumes from existing records, and re-runs cells whose trials ended
without a chess verdict until each cell has ``trials`` scored records.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from xiangqibench import __version__
from xiangqibench.cases import EndgameCase, load_cases
from xiangqibench.config import Config
from xiangqibench.defender import Defender, PikafishDefender, PikafishEngine, RuleDefender
from xiangqibench.harness.adapter import XiangqiAdapter
from xiangqibench.harness.harness import GameHarness
from xiangqibench.harness.protocol import parse_action
from xiangqibench.llm import Agent, build_agent
from xiangqibench.modes import Mode, get_mode
from xiangqibench.scoring import classify_record

log = logging.getLogger("xiangqibench")


class ApiFailure(RuntimeError):
    """A model call failed after its retries; the trial has no chess verdict."""


class ApiUnusable(RuntimeError):
    """The endpoint rejects every request (bad key, no access, unknown model)."""


# HTTP statuses that repeat identically on retry: the first set is a property of
# the endpoint and stops the run, the second of the request and skips the trial.
_FATAL_STATUS = frozenset({401, 403, 404})
_NO_RETRY_STATUS = frozenset({400, 413, 422})


@dataclass
class TrialSummary:
    case_id: str
    model: str
    challenger: str
    protocol: str
    defender_kind: str
    outcome: str
    winner: str | None
    termination_reason: str | None
    plies: int
    matched_first_move: bool | None
    first_move: str | None
    final_fen: str
    started_at: float
    finished_at: float
    blindfold: bool
    setting: str
    mode: str
    status: str | None = None
    standard: bool = True
    xiangqibench_version: str = __version__
    defender: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _call_with_retries(agent: Agent, messages: list[dict], attempts: int, backoff_s: float):
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return agent.complete(messages)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in _FATAL_STATUS:
                raise ApiUnusable(f"model endpoint returned HTTP {status}: {exc}") from exc
            last = exc
            log.warning("model call failed (attempt %d/%d): %s", attempt + 1, attempts, exc)
            if status in _NO_RETRY_STATUS:
                break
            if attempt < attempts - 1:
                time.sleep(backoff_s * (attempt + 1))
    raise ApiFailure(str(last))


def _agent_turn(harness: GameHarness, player: str, agent: Agent, *, max_no_command: int,
                api_attempts: int, api_backoff_s: float) -> None:
    """Drive one agent turn to a turn-ending command or a forced resolution."""
    messages = harness.build_messages(player)
    max_calls = harness.config.max_actions_per_turn + max_no_command
    no_command_streak = 0
    for _ in range(max_calls):
        res = _call_with_retries(agent, messages, api_attempts, api_backoff_s)
        harness.record.count_api_call(player, res.usage)
        raw_text = res.text or ""
        parsed = parse_action(raw_text)
        if parsed.action is None and res.reasoning:
            alt = parse_action(res.reasoning)
            if alt.action is not None:
                parsed = alt
                raw_text = raw_text or res.reasoning
        thinking = parsed.thinking or res.reasoning

        if parsed.action is None:
            no_command_streak += 1
            if no_command_streak >= max_no_command:
                harness.force_turn_resolution(player, reason="no_command")
                return
            harness.append_text_exchange(player, raw_text, harness.build_no_command_feedback())
            messages = harness.build_messages(player)
            continue

        no_command_streak = 0
        log.debug("  $ %s", parsed.raw_command)
        result = harness.execute(
            player=player, action=parsed.action, params=parsed.params, thinking=thinking,
            usage=res.usage, raw_text=raw_text,
            raw_response={"content": raw_text, "finish_reason": res.finish_reason},
        )
        if result.game_over or result.turn_complete:
            return
        messages = harness.build_messages(player)

    if not harness.is_over() and harness.current_player() == player:
        harness.force_turn_resolution(player, reason="action_budget_exhausted")


def play_trial(
    case: EndgameCase,
    mode: Mode,
    agent: Agent,
    defender: Defender,
    *,
    budgets: dict[str, int] | None = None,
    api_attempts: int = 4,
    api_backoff_s: float = 2.0,
    standard: bool = True,
) -> tuple[TrialSummary, dict]:
    """Play one trial and return ``(summary, record)``.

    Raises :class:`ApiFailure` if a model call fails after its retries; such a
    trial has no verdict and should not be recorded. Raises :class:`ApiUnusable`
    if the endpoint rejects the credentials or the model outright.
    """
    budgets = dict(budgets or {})
    max_no_command = int(budgets.pop("max_no_command_per_turn", 5))
    config = mode.harness_config(**budgets)
    challenger = case.challenger
    house = "black" if challenger == "red" else "red"
    backend = getattr(defender, "backend", "rule")
    defender_kind = "engine" if backend == "pikafish" else "rule"

    game = XiangqiAdapter(case.fen)
    harness = GameHarness(game, config, players={challenger: agent.name, house: f"house-{defender_kind}"})

    started = time.time()
    plies = 0
    while not harness.is_over() and plies < config.max_turns:
        player = harness.current_player()
        if player == challenger:
            _agent_turn(harness, player, agent, max_no_command=max_no_command,
                        api_attempts=api_attempts, api_backoff_s=api_backoff_s)
        else:
            harness.build_messages(house)
            choice = defender.choose(game.get_fen())
            if choice.move is None:
                break
            turn = harness.record.current_turn
            harness.execute(house, "move", {"from_to": choice.move}, thinking="[house defender]")
            if turn is not None:
                turn.defender_backend = choice.backend
        plies += 1
    finished = time.time()

    record = harness.get_record()
    moves = record.get("moves", [])
    result = record.get("result") or {}
    winner, reason = result.get("winner"), result.get("reason")
    if not harness.is_over():
        outcome = "turn_cap"
    elif winner == challenger:
        outcome = "challenger_win"
    elif winner in (None, "draw"):
        outcome = "draw"
    else:
        outcome = "challenger_loss"

    first_move = moves[0] if moves else None
    summary = TrialSummary(
        case_id=case.id, model=agent.name, challenger=challenger, protocol="repl",
        defender_kind=defender_kind, outcome=outcome, winner=winner,
        termination_reason=reason, plies=len(moves),
        matched_first_move=(first_move == case.first_winning_move
                            if first_move and case.first_winning_move else None),
        first_move=first_move, final_fen=game.get_fen(),
        started_at=started, finished_at=finished,
        blindfold=config.blindfold, setting=mode.setting, mode=mode.name,
        standard=standard, defender=defender.info(),
    )
    record["endgame"] = summary.to_dict()
    record["endgame"]["case"] = case.to_dict()
    summary.status = classify_record(record)
    record["endgame"]["status"] = summary.status
    return summary, record


# ── suite runner ─────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


def cell_dir(output_dir: str | Path, model: str, setting: str, case_id: str) -> Path:
    return Path(output_dir) / _slug(model) / setting / case_id


def write_record(output_dir: str | Path, record: dict) -> Path:
    eg = record["endgame"]
    folder = cell_dir(output_dir, eg["model"], eg["setting"], eg["case_id"])
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(eg["started_at"]))
    path = folder / f"{stamp}-{uuid.uuid4().hex[:8]}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), "utf-8")
    tmp.replace(path)
    return path


def scored_count(output_dir: str | Path, model: str, setting: str, case_id: str) -> int:
    folder = cell_dir(output_dir, model, setting, case_id)
    if not folder.is_dir():
        return 0
    n = 0
    for path in folder.glob("*.json"):
        try:
            if classify_record(json.loads(path.read_text("utf-8"))) is not None:
                n += 1
        except (OSError, json.JSONDecodeError):
            continue
    return n


@dataclass
class SuiteReport:
    completed_cells: int = 0
    incomplete_cells: list[str] = field(default_factory=list)
    trials_played: int = 0
    api_failures: int = 0
    passes: int = 0
    scored: int = 0


def make_engine(cfg: Config) -> PikafishEngine:
    d = cfg.defender
    return PikafishEngine(path=d.engine_path, nnue=d.nnue, depth=d.depth,
                          threads=d.threads, hash_mb=d.hash_mb)


def run_suite(
    cfg: Config,
    *,
    cases: Iterable[EndgameCase] | None = None,
    agent_factory: Callable[[], Agent] | None = None,
    on_trial: Callable[[TrialSummary, Path], None] | None = None,
) -> SuiteReport:
    """Run every case of ``cfg`` until each has ``cfg.run.trials`` scored records."""
    cfg.validate()
    mode = get_mode(cfg.run.mode)
    cases = list(cases) if cases is not None else load_cases(
        cfg.cases.split, path=cfg.cases.path, ids=cfg.cases.ids, limit=cfg.cases.limit)
    agent_factory = agent_factory or (lambda: build_agent(cfg.model))
    budgets = cfg.effective_budgets()
    out = cfg.run.output_dir
    report = SuiteReport()
    lock = threading.Lock()
    local = threading.local()
    engines: list[PikafishEngine] = []

    if cfg.defender.backend == "pikafish":
        make_engine(cfg).close()

    def worker_state():
        if not hasattr(local, "agent"):
            local.agent = agent_factory()
            local.engine = None
            if cfg.defender.backend == "pikafish":
                local.engine = make_engine(cfg)
                with lock:
                    engines.append(local.engine)
        return local.agent, local.engine

    def run_cell(case: EndgameCase) -> tuple[str, bool]:
        agent, engine = worker_state()
        have = scored_count(out, agent.name, mode.setting, case.id)
        budget = max(0, cfg.run.trials - have) + cfg.run.max_extra_attempts
        while have < cfg.run.trials and budget > 0:
            budget -= 1
            defender: Defender = (
                PikafishDefender(case.challenger, engine, fallback_depth=cfg.defender.fallback_depth)
                if engine is not None else
                RuleDefender(case.challenger, search_depth=cfg.defender.fallback_depth))
            try:
                summary, record = play_trial(
                    case, mode, agent, defender, budgets=budgets,
                    api_attempts=cfg.run.api_attempts, api_backoff_s=cfg.run.api_backoff_s,
                    standard=cfg.is_standard)
            except ApiFailure as exc:
                log.error("%s: model API failed, trial skipped (%s)", case.id, exc)
                with lock:
                    report.api_failures += 1
                continue
            path = write_record(out, record)
            with lock:
                report.trials_played += 1
                if summary.status is not None:
                    report.scored += 1
                    report.passes += summary.status == "pass"
            if summary.status is not None:
                have += 1
            log.info("%s  %-8s %-24s plies=%-2d %s", case.id, summary.status or "excluded",
                     summary.termination_reason or summary.outcome, summary.plies, path.name)
            if on_trial:
                on_trial(summary, path)
        return case.id, have >= cfg.run.trials

    try:
        with ThreadPoolExecutor(max_workers=cfg.run.workers) as pool:
            futures = [pool.submit(run_cell, c) for c in cases]
            for fut in as_completed(futures):
                case_id, done = fut.result()
                if done:
                    report.completed_cells += 1
                else:
                    report.incomplete_cells.append(case_id)
    finally:
        for engine in engines:
            engine.close()
    return report
