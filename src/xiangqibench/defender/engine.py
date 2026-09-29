"""UCI wrapper around the Pikafish xiangqi engine.

Searches use a fixed ``go depth`` with ``Threads 1`` and a fresh
``ucinewgame`` per position, so a pinned engine build and network return the
same move for the same FEN.

Composed endgames (排局) may place pieces on squares unreachable in play, e.g.
a pawn behind its starting rank. Pikafish treats such a position as a fatal
error and exits; :meth:`PikafishEngine.analyse` detects this, restarts the
engine, and raises :class:`UnsupportedPosition` so the caller can fall back.
Any other engine failure raises :class:`EngineUnavailable`, so a broken engine
stops the run instead of silently handing every move to the fallback.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

_MOVE_RE = re.compile(r"^[a-i][0-9][a-i][0-9]$")
_SCORE_MATE_RE = re.compile(r"score mate (-?\d+)")
_SCORE_CP_RE = re.compile(r"score cp (-?\d+)")
_DEPTH_RE = re.compile(r"\bdepth (\d+)")

_CANDIDATE_PATHS = (
    "~/.local/engines/Pikafish/src/pikafish",
    "~/.local/bin/pikafish",
    "/opt/homebrew/bin/pikafish",
    "/usr/local/bin/pikafish",
)


class EngineUnavailable(RuntimeError):
    """No usable Pikafish binary could be located or started."""


class UnsupportedPosition(RuntimeError):
    """The engine rejected a FEN as illegal or unreachable."""


@dataclass
class Analysis:
    bestmove: str | None
    score_mate: int | None
    score_cp: int | None
    pv: list[str] = field(default_factory=list)
    depth: int = 0


def normalize_fen(fen: str) -> str:
    """Pad a short xiangqi FEN to the six fields UCI engines expect."""
    parts = fen.strip().split()
    fillers = {2: "-", 3: "-", 4: "0", 5: "1"}
    while len(parts) < 6:
        parts.append(fillers[len(parts)])
    return " ".join(parts[:6])


def locate_pikafish(path: str | None = None) -> str | None:
    """Resolve the engine binary: explicit path, ``PIKAFISH_PATH``, ``PATH``, common locations."""
    for cand in (path, os.environ.get("PIKAFISH_PATH")):
        if cand:
            cand = os.path.expanduser(cand)
            if os.path.exists(cand):
                return cand
            found = shutil.which(cand)
            if found:
                return found
    found = shutil.which("pikafish")
    if found:
        return found
    for cand in _CANDIDATE_PATHS:
        cand = os.path.expanduser(cand)
        if os.path.exists(cand):
            return cand
    return None


def locate_nnue(engine_path: str, nnue: str | None = None) -> str | None:
    """Resolve the NNUE network: explicit path, ``PIKAFISH_NNUE``, or ``pikafish.nnue`` beside the binary."""
    for cand in (nnue, os.environ.get("PIKAFISH_NNUE")):
        if cand:
            return os.path.expanduser(cand)
    sibling = Path(engine_path).with_name("pikafish.nnue")
    return str(sibling) if sibling.exists() else None


def _engine_error(lines: list[str]) -> str:
    errors = [line.removeprefix("info string ") for line in lines if "ERROR" in line]
    return " ".join(errors) or "no output"


def file_sha256(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PikafishEngine:
    """A single Pikafish subprocess; ``analyse`` is serialised by a lock."""

    def __init__(self, *, path: str | None = None, nnue: str | None = None,
                 depth: int = 18, threads: int = 1, hash_mb: int = 256,
                 start_timeout: float = 20.0):
        resolved = locate_pikafish(path)
        if resolved is None:
            raise EngineUnavailable(
                "Pikafish not found. Install it and set PIKAFISH_PATH "
                "(see `xiangqibench doctor`)."
            )
        self.engine_path = resolved
        self.nnue = locate_nnue(resolved, nnue)
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.start_timeout = start_timeout
        self.engine_id: str | None = None
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._start()

    def info(self) -> dict:
        return {
            "engine_path": self.engine_path,
            "engine_id": self.engine_id,
            "nnue": self.nnue,
            "nnue_sha256": file_sha256(self.nnue),
            "depth": self.depth,
            "threads": self.threads,
            "hash_mb": self.hash_mb,
        }

    def _start(self) -> None:
        try:
            self._process = subprocess.Popen(
                [self.engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except OSError as exc:
            raise EngineUnavailable(f"cannot start {self.engine_path}: {exc}") from exc
        self._send("uci")
        for line in self._read_until("uciok"):
            if line.startswith("id name "):
                self.engine_id = line[len("id name "):].strip()
        if self.nnue:
            self._send(f"setoption name EvalFile value {self.nnue}")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("isready")
        if not any("readyok" in line for line in self._read_until("readyok")):
            self.close()
            raise EngineUnavailable(f"{self.engine_path} did not answer readyok")
        # The network is loaded lazily on the first search, and a network that does
        # not match the binary makes Pikafish exit there, so probe with one search.
        self._send("position startpos")
        self._send("go depth 1")
        lines = self._read_until("bestmove")
        if not any(line.startswith("bestmove") for line in lines):
            self.close()
            raise EngineUnavailable(f"{self.engine_path} failed a test search: {_engine_error(lines)}")

    def _send(self, cmd: str) -> None:
        proc = self._process
        if proc and proc.stdin:
            try:
                proc.stdin.write(cmd + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                self._process = None

    def _read_until(self, target: str, timeout: float | None = None) -> list[str]:
        timeout = self.start_timeout if timeout is None else timeout
        lines: list[str] = []
        proc = self._process
        if not proc or not proc.stdout:
            return lines
        stdout = proc.stdout

        def reader() -> None:
            try:
                while True:
                    line = stdout.readline()
                    if not line:
                        break
                    line = line.strip()
                    lines.append(line)
                    if target in line:
                        break
            except (ValueError, OSError):
                pass

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            self.close()
            raise EngineUnavailable(f"engine timed out waiting for '{target}'")
        return lines

    def close(self) -> None:
        proc, self._process = self._process, None
        if not proc:
            return
        try:
            if proc.poll() is None and proc.stdin and not proc.stdin.closed:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
        except Exception:
            pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream and not stream.closed:
                    stream.close()
            except Exception:
                pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def __enter__(self) -> PikafishEngine:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def analyse(self, fen: str, depth: int | None = None) -> Analysis:
        with self._lock:
            return self._analyse_locked(fen, self.depth if depth is None else depth)

    def _analyse_locked(self, fen: str, depth: int) -> Analysis:
        if self._process is None or self._process.poll() is not None:
            self.close()
            self._start()
        fen_norm = normalize_fen(fen)
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
        self._send(f"position fen {fen_norm}")
        self._send("isready")
        probe = self._read_until("readyok", timeout=15.0)
        if self._process is None or any(
                "Unsupported position" in line or "CRITICAL ERROR" in line for line in probe):
            self.close()
            self._start()
            raise UnsupportedPosition(f"engine rejected position: {fen_norm}")
        self._send(f"go depth {depth}")
        lines = self._read_until("bestmove", timeout=max(60.0, depth * 8.0))
        if not any(line.startswith("bestmove") for line in lines):
            self.close()
            raise EngineUnavailable(f"engine exited during search: {_engine_error(lines)}")

        result = Analysis(bestmove=None, score_mate=None, score_cp=None)
        for line in lines:
            if line.startswith("info") and " pv " in line:
                mate = _SCORE_MATE_RE.search(line)
                if mate:
                    result.score_mate, result.score_cp = int(mate.group(1)), None
                else:
                    cp = _SCORE_CP_RE.search(line)
                    if cp:
                        result.score_cp, result.score_mate = int(cp.group(1)), None
                d = _DEPTH_RE.search(line)
                if d:
                    result.depth = int(d.group(1))
                result.pv = line.split(" pv ", 1)[1].split()
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
                    result.bestmove = parts[1]
        return result

    def best_move(self, fen: str, depth: int | None = None) -> str | None:
        """Engine best move in ICCS, or ``None`` if the position is rejected."""
        try:
            res = self.analyse(fen, depth=depth)
        except UnsupportedPosition:
            return None
        if res.bestmove and _MOVE_RE.match(res.bestmove):
            return res.bestmove
        return None
