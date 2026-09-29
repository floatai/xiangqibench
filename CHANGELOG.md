# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-09-26

First public release. It packages the XiangqiBench protocol, the 119 positions, the Pikafish
defender, and the scoring used in the paper.

### Added

- The `xiangqibench` package and CLI, with the subcommands `run`, `score`, `modes`, `prompt`,
  `cases`, `doctor`, `init`, and `play`.
- Six evaluation modes, selected by `mode:` in the config file or by `--mode`:
  - the paper settings `sighted` and `restricted`;
  - the observation-ablation arms `S`, `S-NT`, `R-T`, and `R`.
- YAML configuration with `${VAR}` / `${VAR:-default}` expansion from the environment, strict
  key validation, and command-line overrides.
- Model clients for Chat Completions (OpenAI, OpenRouter, vLLM, SGLang, or any compatible
  server), Azure OpenAI, the OpenAI Responses API, and Anthropic. Calls are retried, and a
  trial whose calls all fail writes no record.
- The Pikafish defender, with a per-position fallback to a depth-5 rule search. Each defender
  move records its backend, and the engine id and NNUE sha256 are stored with every trial.
- A resumable, parallel suite runner. Each worker has its own engine, and records are written
  atomically.
- Scoring: pass@k, pass^k, earliest-three selection per cell, and 95% case-bootstrap intervals
  with the paper's seeds.
- `xiangqibench.replay.replay_record`, which re-verifies an archived trial against the current
  code.
- Conformance tests:
  - archived prompts for all six settings, byte for byte;
  - replay of 14 archived trials;
  - rules, protocol, config, scoring, runner, and CLI tests.

### Fixed (relative to the code that produced the paper's archive)

These fixes change what future runs record. Archived results are scored as they were recorded.
The effect of each fix on the paper's analyses is described in the paper.

- **Check flag.** Feedback used to report `Check: No` after every move, including checking and
  mating moves, and mates were labelled `stalemate`. The flag is now computed on the position
  after the move, and mates are labelled `checkmate`. The trajectory's `in_check` is also set on
  the final, game-ending move. Winners were never affected.
- **Engine failures.** An engine that failed during search, for example on a network it cannot
  load, silently handed every defender move to the rule fallback. It now stops the run with an
  error, and only positions the engine rejects use the fallback.
- **Repetition.** A threefold repetition in which one side checked on every one of its moves now
  loses for that side (perpetual check) instead of being a draw.
- **Forfeit move list.** Forfeit turns appended a copy of the previous move to `moves`, so
  `plies` was one too high for forfeits after at least one move. Forfeits now append nothing.
- **Statistics.** The simulation counter in `stats` was never incremented. All counters now use
  their documented keys.
- **Defender timing.** The defender's `duration_ms` now includes the engine search time.
- **Record size.** Per-call prompt snapshots are off by default. The full message history is
  still stored once per trial.

[0.1.0]: https://github.com/floatai/xiangqibench/releases/tag/v0.1.0
