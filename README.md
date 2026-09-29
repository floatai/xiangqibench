# XiangqiBench

[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)][paper]
[![PyPI](https://img.shields.io/pypi/v/xiangqibench.svg)](https://pypi.org/project/xiangqibench/)
[![Python](https://img.shields.io/pypi/pyversions/xiangqibench.svg)](https://pypi.org/project/xiangqibench/)
[![CI](https://github.com/floatai/xiangqibench/actions/workflows/ci.yml/badge.svg)](https://github.com/floatai/xiangqibench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Paper][paper]** | **[Data card](DATA_CARD.md)** | **[Changelog](CHANGELOG.md)** | **[Citation](#citation)**

This repository contains the official implementation of *Finding the Move Is Not Winning the
Game: XiangqiBench for Closed-Loop Evaluation of LLM Agents*.

XiangqiBench asks an LLM agent to convert 119 composed xiangqi (Chinese chess) endgames into
checkmate against a Pikafish defender. The agent acts through a small command protocol under
per-turn budgets and is scored only on whether it actually delivers mate: finding the right
first move is not enough.

## Installation

```bash
pip install xiangqibench                  # OpenAI-compatible and Azure endpoints
pip install "xiangqibench[anthropic]"     # adds the native Anthropic client
```

XiangqiBench requires Python 3.10 or later and a [Pikafish](https://github.com/official-pikafish/Pikafish)
binary for the defender:

```bash
git clone https://github.com/official-pikafish/Pikafish && make -C Pikafish/src -j build
export PIKAFISH_PATH=$PWD/Pikafish/src/pikafish
xiangqibench doctor                       # checks the engine and prints its build and NNUE hash
```

The paper's defender was Pikafish `fd168f68` with the network whose sha256 begins `a2f41d4d`,
searched to depth 18 with one thread and a 256 MB hash. Upstream has since replaced that network,
and older builds cannot load the new one, so build the current Pikafish as shown above. Its
moves can differ from the paper's defender. Every record stores the engine build and the network
hash.

## Usage

```bash
export OPENAI_API_KEY=...
xiangqibench run --model gpt-5.5 --mode sighted --limit 5
xiangqibench score runs/
```

Full runs are configured with a single YAML file; `xiangqibench init` writes a commented
example. API keys are read from the environment, never from the file.

```yaml
mode: restricted                # sighted | restricted | S | S-NT | R-T | R
model:
  name: qwen3-235b
  provider: openai              # openai | azure | openai-responses | anthropic
  base_url: http://localhost:8000/v1
  api_key_env: VLLM_API_KEY
run:
  trials: 3
  workers: 8
```

```bash
xiangqibench run -c my_run.yaml           # resumable: re-running fills in missing trials
```

Command-line flags override the file, and unknown keys are rejected.

### Modes

| Mode | Observation | Tools |
|---|---|---|
| `sighted` | Board, FEN, and legal moves after every ply | `view_board`, `simulate`, `get_legal_moves` |
| `restricted` | Starting position once, then move diffs only | none |
| `S`, `S-NT`, `R-T`, `R` | Observation ablations: state push (S/R) × tool access (T/NT) | as named |

`sighted` and `restricted` are the paper's two settings. `xiangqibench prompt --mode <mode>`
prints the exact system prompt for any mode.

### Scoring

`xiangqibench score` reports pass@k and pass^k over the earliest three scored trials per
(model, mode, case), with 95% case-bootstrap intervals using the paper's seeds. Trials cut short
by infrastructure errors are excluded and re-run automatically. Runs that change the standard
budgets or defender settings are marked `standard: false`.

### Python API

```python
from xiangqibench import load_config
from xiangqibench.runner import run_suite

report = run_suite(load_config("my_run.yaml"))
```

Any object with a `name` attribute and a `complete(messages) -> Completion` method can be
evaluated as an agent; see `xiangqibench.runner.play_trial`.

## Reproducibility

Every trial is stored as one JSON record with the full message history, the move list, the
resolved configuration, and the defender's identity, including which backend chose each
defender move. The test suite replays archived trials from the paper against this code and checks
every environment message and verdict (`xiangqibench.replay`). Known differences from the code
that produced the paper's archive are listed in the [changelog](CHANGELOG.md).

```bash
pip install -e ".[dev]" && pytest
```

## Citation

```bibtex
@article{xiangqibench2026,
  title   = {Finding the Move Is Not Winning the Game: {XiangqiBench} for Closed-Loop
             Evaluation of {LLM} Agents},
  author  = {XiangqiBench authors},
  year    = {2026},
  url     = {https://github.com/floatai/xiangqibench}
}
```

## License

The code is released under the [MIT License](LICENSE). The historical positions are in the
public domain. Pikafish is licensed under GPL-3.0; it is not distributed with this package and
runs as a separate process.

[paper]: https://arxiv.org/abs/XXXX.XXXXX
