"""``xiangqibench`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from importlib import resources
from pathlib import Path

from xiangqibench import __version__
from xiangqibench.cases import available_splits, load_case, load_cases
from xiangqibench.config import STANDARD_BUDGETS, ConfigError, load_config
from xiangqibench.modes import MODES, get_mode


def _add_config_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="YAML config file")
    p.add_argument("--mode", help=f"evaluation mode: {', '.join(MODES)}")
    p.add_argument("--model", help="model label (and API model id unless --model-id is set)")
    p.add_argument("--model-id", help="model id sent to the API")
    p.add_argument("--provider", help="openai | azure | openai-responses | anthropic")
    p.add_argument("--base-url", help="API base URL (OpenAI-compatible endpoints, Azure)")
    p.add_argument("--api-key-env", help="environment variable holding the API key")
    p.add_argument("--trials", type=int, help="scored trials per case (default 3)")
    p.add_argument("--workers", type=int, help="parallel games")
    p.add_argument("-o", "--output", help="output directory for trial records")
    p.add_argument("--split", help=f"case split: {', '.join(available_splits())}")
    p.add_argument("--case", action="append", dest="case_ids", help="case id (repeatable)")
    p.add_argument("--limit", type=int, help="run only the first N cases")
    p.add_argument("--defender", choices=("pikafish", "rule"), help="defender backend")


def _load(args):
    cfg = load_config(args.config)
    m, r, c = cfg.model, cfg.run, cfg.cases
    for attr, target, key in (
        ("mode", r, "mode"), ("model", m, "name"), ("model_id", m, "model"),
        ("provider", m, "provider"), ("base_url", m, "base_url"),
        ("api_key_env", m, "api_key_env"), ("trials", r, "trials"),
        ("workers", r, "workers"), ("output", r, "output_dir"), ("split", c, "split"),
        ("case_ids", c, "ids"), ("limit", c, "limit"), ("defender", cfg.defender, "backend"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(target, key, value)
    return cfg


def cmd_run(args) -> int:
    from xiangqibench.runner import run_suite

    cfg = _load(args).validate()
    mode = get_mode(cfg.run.mode)
    if not cfg.is_standard:
        logging.warning("non-standard budgets or defender: results are not leaderboard-comparable")
    logging.info("xiangqibench %s | model=%s mode=%s (%s) trials=%d workers=%d -> %s",
                 __version__, cfg.model.name, mode.name, mode.setting,
                 cfg.run.trials, cfg.run.workers, cfg.run.output_dir)
    report = run_suite(cfg)
    rate = report.passes / report.scored if report.scored else 0.0
    print(json.dumps({
        "completed_cases": report.completed_cells,
        "incomplete_cases": report.incomplete_cells,
        "trials_played": report.trials_played,
        "scored_trials": report.scored,
        "pass_rate": round(rate, 4),
        "api_failures": report.api_failures,
    }, indent=2))
    return 0 if not report.incomplete_cells else 2


def cmd_score(args) -> int:
    from xiangqibench.scoring import format_markdown, load_trials, score

    trials, excluded = load_trials(args.paths)
    if args.setting:
        wanted = {get_mode(s).setting for s in args.setting}
        trials = [t for t in trials if t.setting in wanted]
    case_ids = [c.id for c in load_cases(args.split)] if args.split else None
    if case_ids is not None:
        keep = set(case_ids)
        trials = [t for t in trials if t.case_id in keep]
    rows = score(trials, case_ids=case_ids, n=args.trials)
    if args.format == "json":
        text = json.dumps({"excluded_records": excluded, "rows": [r.to_dict() for r in rows]},
                          indent=2)
    else:
        text = format_markdown(rows) + f"\n{len(trials)} scored trials; {excluded} excluded records.\n"
    if args.out:
        Path(args.out).write_text(text, "utf-8")
    else:
        print(text)
    return 0


def cmd_modes(args) -> int:
    for m in MODES.values():
        print(f"{m.name:<11} {m.setting:<15} {m.summary}")
    print("\nStandard budgets: " + ", ".join(f"{k}={v}" for k, v in STANDARD_BUDGETS.items()))
    return 0


def cmd_prompt(args) -> int:
    from xiangqibench.harness import GameHarness, XiangqiAdapter

    mode = get_mode(args.mode)
    case = load_case(args.case) if args.case else load_cases()[0]
    budgets = {k: v for k, v in STANDARD_BUDGETS.items() if k != "max_no_command_per_turn"}
    harness = GameHarness(XiangqiAdapter(case.fen), mode.harness_config(**budgets))
    messages = harness.build_messages(case.challenger)
    print(f"# mode={mode.name} setting={mode.setting} case={case.id}\n")
    for msg in messages:
        print(f"===== {msg['role']} =====\n{msg['content']}\n")
    return 0


def cmd_cases(args) -> int:
    cases = load_cases(args.split)
    if args.format == "jsonl":
        for c in cases:
            print(json.dumps(c.to_dict(), ensure_ascii=False))
        return 0
    print(f"{'id':<26}{'side':<7}{'plies':<7}{'category':<17}name")
    for c in cases:
        print(f"{c.id:<26}{c.challenger:<7}{c.win_in_plies:<7}{c.category:<17}{c.name}")
    print(f"\n{len(cases)} cases in split {args.split!r}")
    return 0


def cmd_doctor(args) -> int:
    from xiangqibench.defender import EngineUnavailable, PikafishEngine, locate_pikafish

    ok = True
    print(f"xiangqibench {__version__}  python {platform.python_version()}  {platform.platform()}")
    for mod in ("cchess", "numpy", "yaml", "openai", "anthropic"):
        try:
            m = __import__(mod)
            print(f"  [ok]   {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            required = mod in ("cchess", "numpy", "yaml")
            ok &= not required
            print(f"  [{'FAIL' if required else 'opt'}] {mod} not installed")
    path = locate_pikafish(args.engine_path)
    if not path:
        ok = False
        print("  [FAIL] Pikafish not found. Build it from https://github.com/official-pikafish/Pikafish\n"
              "         and set PIKAFISH_PATH (and PIKAFISH_NNUE if pikafish.nnue is not beside it).")
    else:
        try:
            with PikafishEngine(path=path, depth=args.depth) as engine:
                info = engine.info()
                case = load_cases()[0]
                move = engine.best_move(case.fen)
            print(f"  [ok]   Pikafish {info['engine_id']} at {path}")
            print(f"         nnue {info['nnue']} sha256={str(info['nnue_sha256'])[:16]}")
            print(f"         {case.id}: bestmove {move} at depth {args.depth}")
        except EngineUnavailable as exc:
            ok = False
            print(f"  [FAIL] Pikafish at {path} unusable: {exc}")
    print(f"  [ok]   {len(load_cases())} cases; splits: {', '.join(available_splits())}")
    return 0 if ok else 1


def cmd_init(args) -> int:
    target = Path(args.path)
    if target.exists() and not args.force:
        print(f"{target} exists; use --force to overwrite", file=sys.stderr)
        return 1
    target.write_text(resources.files("xiangqibench").joinpath("example_config.yaml").read_text("utf-8"),
                      "utf-8")
    print(f"wrote {target}")
    return 0


class _StdinAgent:
    name = "human"

    def complete(self, messages):
        from xiangqibench.llm import Completion

        print("\n" + "-" * 80 + f"\n{messages[-1]['content']}\n" + "-" * 80)
        line = input("command> ").strip()
        return Completion(text=f"```bash\n{line}\n```")


def cmd_play(args) -> int:
    from xiangqibench.defender import Defender, PikafishDefender, PikafishEngine, RuleDefender
    from xiangqibench.runner import play_trial

    mode = get_mode(args.mode)
    case = load_case(args.case) if args.case else load_cases()[0]
    engine = None
    defender: Defender
    if args.defender == "pikafish":
        engine = PikafishEngine()
        defender = PikafishDefender(case.challenger, engine)
    else:
        defender = RuleDefender(case.challenger)
    try:
        summary, _ = play_trial(case, mode, _StdinAgent(), defender)
    finally:
        if engine:
            engine.close()
    print(f"\nresult: {summary.status} ({summary.termination_reason}) after {summary.plies} plies")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xiangqibench", description=__doc__)
    parser.add_argument("--version", action="version", version=f"xiangqibench {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="evaluate a model")
    _add_config_overrides(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="score trial records into a leaderboard")
    p.add_argument("paths", nargs="+", help="record files or directories")
    p.add_argument("--split", default="main", help="case split fixing the case set ('' = any)")
    p.add_argument("--setting", action="append", help="restrict to mode(s)")
    p.add_argument("--trials", type=int, default=3, help="trials per case (n in pass@k)")
    p.add_argument("--format", choices=("md", "json"), default="md")
    p.add_argument("--out", help="write to file instead of stdout")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("modes", help="list evaluation modes")
    p.set_defaults(func=cmd_modes)

    p = sub.add_parser("prompt", help="print the exact prompt a model sees")
    p.add_argument("--mode", default="sighted")
    p.add_argument("--case")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("cases", help="list benchmark cases")
    p.add_argument("--split", default="main")
    p.add_argument("--format", choices=("table", "jsonl"), default="table")
    p.set_defaults(func=cmd_cases)

    p = sub.add_parser("doctor", help="check dependencies and the Pikafish engine")
    p.add_argument("--engine-path")
    p.add_argument("--depth", type=int, default=12)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("init", help="write an example config file")
    p.add_argument("path", nargs="?", default="xiangqibench.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("play", help="play a case yourself in the terminal")
    p.add_argument("--mode", default="sighted")
    p.add_argument("--case")
    p.add_argument("--defender", choices=("pikafish", "rule"), default="pikafish")
    p.set_defaults(func=cmd_play)
    return parser


def main(argv: list[str] | None = None) -> int:
    from xiangqibench.defender import EngineUnavailable

    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose + (args.command == "run"), 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if getattr(args, "split", None) == "":
        args.split = None
    try:
        return args.func(args)
    except (ConfigError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EngineUnavailable as exc:
        print(f"error: {exc}\nrun `xiangqibench doctor` for setup help", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
