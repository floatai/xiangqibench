"""Bounded forced-mate search on :mod:`cchess` boards.

:func:`forced_mate_search` returns the smallest ``N`` such that ``attacker``
forces checkmate within ``N`` plies, or ``None``; stalemate is a loss for the
side to move, as in xiangqi. It powers the rule-based defender used for
positions the engine rejects. The search ignores repetition rules.

Performance comes from three places: a zhash transposition table that caches
both proven mate distances and depth-bounded "no mate" results; a move
ordering that explores checks before quiet moves on the attacker's turn; and
iterative deepening so easy puzzles return early.
"""

from __future__ import annotations

import cchess


def _side_name(board) -> str:
    """Return ``"red"`` / ``"black"`` for the side to move on ``board``."""
    return "red" if board.move_player == cchess.RED else "black"


def _legal_moves(board, prefer_checks: bool, checks_only: bool = False) -> list:
    """Legal moves with optional check-only filter and check-first ordering.

    Filters out pseudo-legal moves that leave the moving side in check. When
    ``checks_only`` is true, only moves that put the opponent in check are
    returned (used to restrict an attacker to check-net searches in
    threatmate-race verification). When ``prefer_checks`` is true, checking
    moves are returned first.
    """
    out_check = []
    out_quiet = []
    for src, dst in board.create_moves():
        if board.is_checked_move(src, dst):
            continue
        gives_check = board.is_checking_move(src, dst)
        if gives_check:
            out_check.append((src, dst))
        elif not checks_only:
            out_quiet.append((src, dst))
    if not prefer_checks:
        return out_check + out_quiet
    return out_check + out_quiet


def _apply(board, src, dst):
    """Return the child board after playing ``(src, dst)``.

    Falls through to ``None`` when ``cchess`` rejects the move because the
    resulting position is malformed (e.g. one side's king has been captured).
    Search routines treat this as "do not explore this move", which is safe
    for mate proofs because it can only narrow the move set.
    """
    try:
        nb = board.copy()
        nb.move(src, dst)
        nb.next_turn()
        return nb
    except Exception:
        return None


class _MateSolver:
    """Bounded mate-in-N search with a zhash transposition table.

    The cache stores either a positive integer (proven shortest mate distance
    in plies for ``attacker``) or a tuple ``("none", depth)`` recording that
    no mate within ``depth`` plies exists. Mixing the two lets us reuse work
    across iterative-deepening calls without ever returning an unsound result.
    """

    def __init__(self, attacker: str, attacker_checks_only: bool = False):
        self.attacker = attacker
        self.attacker_checks_only = attacker_checks_only
        self._cache: dict = {}

    def search(self, board, depth: int) -> int | None:
        if depth < 0:
            return None
        z = board.zhash
        cached = self._cache.get(z)
        if isinstance(cached, int):
            return cached if cached <= depth else None
        if isinstance(cached, tuple) and cached[1] >= depth:
            return None

        side = _side_name(board)
        is_attacker_turn = side == self.attacker
        moves = _legal_moves(
            board,
            prefer_checks=is_attacker_turn,
            checks_only=is_attacker_turn and self.attacker_checks_only,
        )

        if not moves:
            # Attacker without a (checking) move cannot mate; a defender without
            # a legal move loses, since stalemate is a loss in xiangqi.
            result = None if is_attacker_turn else 0
        elif depth == 0:
            result = None
        elif is_attacker_turn:
            best: int | None = None
            for src, dst in moves:
                child = _apply(board, src, dst)
                if child is None:
                    continue
                n = self.search(child, depth - 1)
                if n is None:
                    continue
                cand = n + 1
                if best is None or cand < best:
                    best = cand
                    if best == 1:
                        break
            result = best
        else:
            worst = 0
            broke = False
            for src, dst in moves:
                child = _apply(board, src, dst)
                if child is None:
                    # Defender cannot legally make this reply; skip.
                    continue
                n = self.search(child, depth - 1)
                if n is None:
                    broke = True
                    break
                cand = n + 1
                if cand > worst:
                    worst = cand
            result = None if broke else worst

        if isinstance(result, int):
            prev = self._cache.get(z)
            if not isinstance(prev, int) or result < prev:
                self._cache[z] = result
        else:
            prev = self._cache.get(z)
            if isinstance(prev, int):
                pass  # keep the proven mate
            elif not isinstance(prev, tuple) or prev[1] < depth:
                self._cache[z] = ("none", depth)
        return result


def forced_mate_search(
    board,
    attacker: str,
    max_depth: int,
    attacker_checks_only: bool = False,
) -> int | None:
    """Iteratively deepen mate search up to ``max_depth`` plies.

    When ``attacker_checks_only`` is true, only checking moves are considered
    on the attacker's turn; this makes the search a *check-net* prover, sound
    for threatmate-race style puzzles in which the attacker must keep the
    initiative or lose.
    """
    solver = _MateSolver(attacker, attacker_checks_only=attacker_checks_only)
    last: int | None = None
    for depth in range(1, max_depth + 1):
        last = solver.search(board, depth)
        if last is not None:
            return last
    return None


def best_winning_move(
    board,
    attacker: str,
    max_depth: int,
    attacker_checks_only: bool = False,
) -> str | None:
    """Return one ICCS move that achieves the shortest forced mate, if any."""
    if _side_name(board) != attacker:
        return None
    solver = _MateSolver(attacker, attacker_checks_only=attacker_checks_only)
    for depth in range(1, max_depth + 1):
        if solver.search(board, depth) is None:
            continue
        best_move = None
        best_n: int | None = None
        for src, dst in _legal_moves(
            board,
            prefer_checks=True,
            checks_only=attacker_checks_only,
        ):
            child = _apply(board, src, dst)
            if child is None:
                continue
            n = solver.search(child, depth - 1)
            if n is None:
                continue
            cand = n + 1
            if best_n is None or cand < best_n:
                best_n = cand
                best_move = cchess.pos2iccs(src, dst)
        return best_move
    return None
