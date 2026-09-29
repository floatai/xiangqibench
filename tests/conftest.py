import gzip
import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden"


def load_trial_fixture(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def trial_fixtures() -> list[Path]:
    return sorted((GOLDEN / "trials").glob("*.json.gz"))


def prompt_fixtures() -> list[Path]:
    return sorted((GOLDEN / "prompts").glob("*.json"))


@pytest.fixture(scope="session")
def pikafish_path():
    from xiangqibench.defender import locate_pikafish

    path = locate_pikafish()
    if not path:
        pytest.skip("Pikafish not available (set PIKAFISH_PATH)")
    return path
