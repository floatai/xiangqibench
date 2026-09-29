"""The XiangqiBench endgame suite.

The packaged suite has 119 composed xiangqi endgames (116 from 适情雅趣
*Shi Qing Ya Qu*, 3 from 江湖排局 *Jianghu* collections). In each, the side to
move has a forced mate in 3-11 plies, verified by Pikafish. Named splits
select subsets; ``main`` is the full suite used for the leaderboard.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

MAIN_SPLIT = "main"


@dataclass(frozen=True)
class EndgameCase:
    id: str
    source: str
    raw_index: int
    name: str
    fen: str
    challenger: str
    category: str
    win_in_plies: int
    defender_threat_in_plies: int | None
    first_winning_move: str | None
    legal_move_count: int
    tags: list[str] = field(default_factory=list)
    best_move_hint: str | None = None
    difficulty_score: float = 0.0
    tier: str = ""
    piece_theme: list[str] = field(default_factory=list)
    verified_by: str = "engine"
    engine_mate_plies: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EndgameCase:
        return cls(
            id=data["id"],
            source=data["source"],
            raw_index=data["raw_index"],
            name=data.get("name") or "",
            fen=data["fen"],
            challenger=data["challenger"],
            category=data["category"],
            win_in_plies=data["win_in_plies"],
            defender_threat_in_plies=data.get("defender_threat_in_plies"),
            first_winning_move=data.get("first_winning_move"),
            legal_move_count=data.get("legal_move_count", 0),
            tags=list(data.get("tags") or []),
            best_move_hint=data.get("best_move_hint"),
            difficulty_score=float(data.get("difficulty_score") or 0.0),
            tier=data.get("tier") or "",
            piece_theme=list(data.get("piece_theme") or []),
            verified_by=data.get("verified_by") or "engine",
            engine_mate_plies=data.get("engine_mate_plies"),
        )


def _data_dir():
    return resources.files("xiangqibench").joinpath("data")


def available_splits() -> list[str]:
    splits = [MAIN_SPLIT]
    for entry in _data_dir().joinpath("splits").iterdir():
        if entry.name.endswith(".txt"):
            splits.append(entry.name[:-4])
    return sorted(splits)


def _read_id_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def load_cases(
    split: str = MAIN_SPLIT,
    *,
    path: os.PathLike | str | None = None,
    ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[EndgameCase]:
    """Load cases from the packaged suite (or a JSONL ``path``).

    ``split`` names a packaged subset; ``ids`` restricts to explicit case ids
    (order preserved from the suite). Unknown ids raise ``KeyError``.
    """
    text = (Path(path).read_text("utf-8") if path
            else _data_dir().joinpath("cases.jsonl").read_text("utf-8"))
    cases = [EndgameCase.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]

    wanted: list[str] | None = None
    if split != MAIN_SPLIT:
        entry = _data_dir().joinpath("splits", f"{split}.txt")
        if not entry.is_file():
            raise KeyError(f"unknown split {split!r}; available: {', '.join(available_splits())}")
        wanted = _read_id_list(entry.read_text("utf-8"))
    if ids is not None:
        ids = list(ids)
        wanted = ids if wanted is None else [i for i in wanted if i in set(ids)]

    if wanted is not None:
        by_id = {c.id: c for c in cases}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise KeyError(f"unknown case ids: {', '.join(missing[:5])}")
        keep = set(wanted)
        cases = [c for c in cases if c.id in keep]
    return cases[:limit] if limit else cases


def load_case(case_id: str) -> EndgameCase:
    return load_cases(ids=[case_id])[0]
