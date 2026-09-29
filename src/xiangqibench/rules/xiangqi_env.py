"""Xiangqi rules on top of the ``cchess`` move generator.

Adjudication implemented here:

* checkmate and stalemate: the side to move with no legal move loses
  (stalemate is a loss in Xiangqi, unlike Western chess);
* threefold repetition: a draw, unless exactly one side gave check on every one
  of its moves since the earliest of the last three occurrences of the repeated
  position, in which case that side loses (perpetual check);
* flying general and self-check are excluded by ``cchess``.

Perpetual chase is not adjudicated.
"""

from __future__ import annotations

from collections import Counter

import cchess

WIN_REASONS = {
    "checkmate": {"en": "Checkmate", "zh": "将死"},
    "stalemate": {"en": "Stalemate (no legal moves)", "zh": "困毙（无合法走法）"},
    "perpetual_check": {
        "en": "Perpetual check (checked on every move of a threefold repetition)",
        "zh": "长将作负（三次重复局面中每步均将军）",
    },
    "repetition": {"en": "Threefold repetition", "zh": "三次重复局面"},
    "king_captured": {"en": "King captured", "zh": "将帅被吃"},
    "max_moves": {"en": "Maximum moves reached (draw)", "zh": "达到最大回合数（和棋）"},
    "resignation": {"en": "Resignation", "zh": "认输"},
}


class XiangqiEnv:
    """Xiangqi game state with move validation and terminal adjudication."""

    RED = 1
    BLACK = 2
    MAX_REPETITIONS = 3

    def __init__(self) -> None:
        self.board: cchess.ChessBoard = cchess.ChessBoard(cchess.FULL_INIT_FEN)
        self.max_turns = 150
        self._game_over = False
        self._winner: str | None = None
        self._win_reason: str | None = None
        self._win_reason_detail: str | None = None
        self.move_count = 0
        self.position_history: list[str] = []
        self.check_history: list[dict] = []
        self.move_history: list[str] = []
        self.reset()

    def reset(self) -> None:
        self.set_state_from_fen(cchess.FULL_INIT_FEN)

    def set_state_from_fen(self, fen: str) -> None:
        self.board = cchess.ChessBoard(fen)
        self._game_over = False
        self._winner = None
        self._win_reason = None
        self._win_reason_detail = None
        self.position_history = [self._get_position_key()]
        self.check_history = []
        self.move_history = []
        self.move_count = 0

    @staticmethod
    def _position_key_from_fen(fen: str) -> str:
        parts = (fen or "").split()
        if len(parts) > 1:
            return parts[0] + " " + parts[1]
        return parts[0] if parts else ""

    def _get_position_key(self) -> str:
        return self._position_key_from_fen(self.board.to_fen())

    def get_repetition_count(self) -> int:
        """Occurrences of the current position (board and side to move) so far."""
        if not self.position_history:
            return 0
        return Counter(self.position_history).get(self.position_history[-1], 0)

    def is_in_check(self) -> bool:
        """Whether the side to move is in check.

        ``cchess.ChessBoard.is_checking()`` reports whether the side to move
        attacks the opposing general, so the turn is flipped temporarily.
        """
        player = self.board.move_player
        player.next()
        try:
            return self.board.is_checking()
        finally:
            player.next()

    def step(self, action: str) -> bool:
        """Play an ICCS move (e.g. ``h2e2``); return False if it is illegal."""
        if self._game_over or action not in self.get_legal_moves():
            return False
        moving_side = self.get_current_player()
        self.board.move_iccs(action)
        self.board.next_turn()
        self.move_count += 1
        self.move_history.append(action)
        self.position_history.append(self._get_position_key())
        self.check_history.append(
            {"side": moving_side, "gave_check": self.is_in_check(), "move": action}
        )
        self._check_game_over()
        return True

    def _check_game_over(self) -> None:
        if not self.get_legal_moves():
            self._game_over = True
            self._winner = "black" if self.get_current_player() == "red" else "red"
            if self.is_in_check():
                self._win_reason = "checkmate"
                self._win_reason_detail = (
                    f"{self._winner.upper()} wins by checkmate / {self._winner}方将死对方获胜"
                )
            else:
                self._win_reason = "stalemate"
                self._win_reason_detail = (
                    f"{self._winner.upper()} wins by stalemate / {self._winner}方困毙对方获胜"
                )
            return

        fen = self.board.to_fen()
        if "K" not in fen or "k" not in fen:
            self._game_over = True
            self._winner = "black" if "K" not in fen else "red"
            self._win_reason = "king_captured"
            self._win_reason_detail = f"{self._winner.upper()} wins - king captured"
            return

        repetition = self._check_repetition()
        if repetition:
            self._game_over = True
            self._winner = repetition["winner"]
            self._win_reason = repetition["reason"]
            self._win_reason_detail = repetition["detail"]
            return

        if self.move_count >= self.max_turns:
            self._game_over = True
            self._winner = "draw"
            self._win_reason = "max_moves"
            self._win_reason_detail = (
                f"Draw - Maximum {self.max_turns} moves reached / 和棋 - 达到最大{self.max_turns}回合"
            )

    def _perpetual_checkers(self) -> set[str]:
        """Sides that gave check on every one of their moves in the repetition cycle.

        ``check_history[i]`` is the move that produced ``position_history[i + 1]``.
        """
        current = self.position_history[-1]
        occurrences = [i for i, pos in enumerate(self.position_history) if pos == current]
        if len(occurrences) < self.MAX_REPETITIONS:
            return set()
        cycle = self.check_history[occurrences[-self.MAX_REPETITIONS]:]
        checkers = set()
        for side in ("red", "black"):
            own = [entry["gave_check"] for entry in cycle if entry["side"] == side]
            if own and all(own):
                checkers.add(side)
        return checkers

    def _check_repetition(self) -> dict | None:
        if Counter(self.position_history)[self.position_history[-1]] < self.MAX_REPETITIONS:
            return None
        checkers = self._perpetual_checkers()
        if len(checkers) == 1:
            loser = checkers.pop()
            winner = "black" if loser == "red" else "red"
            return {
                "winner": winner,
                "reason": "perpetual_check",
                "detail": (
                    f"{winner.upper()} wins - {loser} gave perpetual check (长将作负) / "
                    f"{winner}方获胜 - {loser}方长将作负"
                ),
            }
        return {
            "winner": "draw",
            "reason": "repetition",
            "detail": "Draw by threefold repetition / 三次重复局面和棋",
        }

    def get_legal_moves(self) -> list[str]:
        return [
            cchess.pos2iccs(src, dst)
            for src, dst in self.board.create_moves()
            if not self.board.is_checked_move(src, dst)
        ]

    def is_game_over(self) -> bool:
        return self._game_over

    def get_outcome_details(self) -> dict | None:
        if not self._game_over:
            return None
        info = WIN_REASONS.get(
            self._win_reason or "", {"en": self._win_reason or "unknown", "zh": self._win_reason or "未知"}
        )
        return {
            "winner": self._winner,
            "reason": self._win_reason,
            "reason_en": info["en"],
            "reason_zh": info["zh"],
            "detail": self._win_reason_detail or f"{info['en']} / {info['zh']}",
        }

    def forfeit(self, loser: str, reason: str = "forfeit") -> None:
        if self._game_over:
            return
        self._game_over = True
        self._winner = "black" if loser == "red" else "red"
        self._win_reason = "forfeit"
        loser_zh = "红方" if loser == "red" else "黑方"
        winner_zh = "黑方" if loser == "red" else "红方"
        self._win_reason_detail = (
            f"{self._winner.capitalize()} wins by forfeit ({reason}) / {winner_zh}获胜（{loser_zh}{reason}）"
        )

    def get_current_player(self) -> str:
        return "red" if self.board.get_move_color() == self.RED else "black"

    def get_board_state(self, format: str = "text") -> str:
        if format == "fen":
            return self.board.to_fen()
        return self._get_text_board()

    # cchess draws Black's rook and horse with non-standard stone-radical glyphs
    # (砗, 碼); Black conventionally uses the traditional forms 車 and 馬.
    _BLACK_GLYPH_FIXES = str.maketrans({"砗": "車", "碼": "馬"})

    def _get_text_board(self) -> str:
        lines = [""]
        lines.extend(line.translate(self._BLACK_GLYPH_FIXES) for line in self.board.text_view())
        lines.append("")
        current = self.get_current_player()
        lines.append(
            "当前回合 / Current turn: "
            + ("红方 Red (大写字母/Uppercase)" if current == "red" else "黑方 Black (小写字母/Lowercase)")
        )
        if self.is_in_check():
            lines.append("⚠️  将军! / CHECK!")
        lines.append("走法格式 / Move format: 起点终点 (如 h2e2 表示炮从h2走到e2)")
        return "\n".join(lines)
