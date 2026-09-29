"""System prompts shown to the agent.

The two paper interfaces (Sighted and Restricted observation) are stored
verbatim in ``templates/`` exactly as they were sent during data collection;
only the budget values are placeholders. The observation-ablation arms share
one generated prompt that describes exactly the channels the arm enables.
Conformance tests pin every rendered prompt against archived trajectories.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

from xiangqibench.harness.config import HarnessConfig

_BAR = "=" * 80
_SECTION3_MARKER = f"{_BAR}\n3. BOARD STATE"


@cache
def load_template(name: str) -> str:
    return resources.files("xiangqibench.harness").joinpath(f"templates/{name}.txt").read_text("utf-8")


def fill(template: str, cfg: HarnessConfig) -> str:
    """Substitute ``{placeholder}`` budget fields; other braces are left untouched."""
    for key, value in cfg.prompt_values().items():
        template = template.replace("{" + key + "}", str(value))
    return template


def build_system_prompt(cfg: HarnessConfig) -> str:
    if cfg.obs_arm:
        rules = build_observation_rules_prompt(cfg)
        repl = build_repl_prompt(
            enable_view_board=cfg.enable_view_board,
            enable_simulate=cfg.enable_simulate,
            enable_legal_moves=cfg.enable_legal_moves,
            push_state=cfg.push_state,
        )
        return fill(f"{rules}\n\n{repl}", cfg)
    return fill(load_template("restricted" if cfg.blindfold else "sighted"), cfg)


def build_observation_rules_prompt(cfg: HarnessConfig) -> str:
    """Endgame rules prompt for an observation-ablation arm.

    Sections 1-2 (rules, move format) are shared verbatim with the Sighted
    prompt; sections 3-5 describe the arm's channels, and the action-limit rule
    states the forfeit policy the runner applies."""
    sighted = load_template("sighted")
    head = sighted[:sighted.index(_SECTION3_MARKER)]

    if cfg.push_state:
        state = (
            "You get THREE views; trust them in this order:\n"
            "  1. PIECE POSITIONS census — coordinate-keyed ground truth of every piece. This\n"
            "     is authoritative; do NOT rely on memory (a common failure is misreading a\n"
            "     glyph, e.g. confusing 将 with 象).\n"
            "  2. FEN string — compact full-board encoding (ranks listed row 9 → row 0;\n"
            "     N=Horse, B=Elephant).\n"
            "  3. ASCII board — Chinese glyphs; useful for intuition but easy to misread.\n"
            "The board, FEN, and legal-move list are attached automatically to every move\n"
            "result and to every notice of the opponent's move.\n"
            "Reconcile your mental model with the census/FEN before every move."
        )
    else:
        state = (
            "You are shown the full board and FEN ONCE, in the opening message. After that\n"
            "the environment does NOT attach the board, FEN, or legal-move list\n"
            "automatically: each move result and each notice of the opponent's move reports\n"
            "only what moved, any capture, whether it is check, and material.\n"
        )
        if cfg.any_query_tool:
            state += ("You can request the current position at any time with the tools below;\n"
                      "view_board shows the board, FEN, and the coordinate-keyed piece census.")
        else:
            state += "Track the position yourself from the opening board and the move sequence."

    tools = [
        "  think(reasoning)      — **ALWAYS call this first.** Record your analysis of the\n"
        "                          mating idea, candidate forcing moves, and the lines you\n"
        "                          calculated. Does not end your turn.",
        "  move(from_to)         — Play a move. Ends your turn.",
        "  resign()              — Give up (you lose). Only if the position is truly lost.",
    ]
    if cfg.enable_view_board:
        tools.append("  view_board()          — Show the board with FEN (max {max_view_board}/turn).")
    if cfg.enable_simulate:
        tools.append(
            "  simulate(moves)       — Test a hypothetical line WITHOUT affecting the real\n"
            "                          game. YOU supply both sides: [your_move, assumed_reply,\n"
            "                          your_move, ...]. Each entry is one half-move. Returns per\n"
            "                          ply: what moved/captured, check, FEN, and the number of\n"
            "                          legal replies. Max {max_simulate_steps} half-moves,\n"
            "                          {max_simulate}/turn. Use it to verify a forced mate\n"
            "                          before committing.")
    if cfg.enable_legal_moves:
        tools.append("  get_legal_moves()     — List all legal moves in the current position.")
    tools.append("  get_history()         — Show the full move history.")
    flow = "think → (simulate to verify the mate) → move" if cfg.enable_simulate else "think → move"
    after = ("what moved/captured, whether it is check, material balance, the updated FEN,\n"
             "and the legal moves for the next position." if cfg.push_state else
             "what moved/captured, whether it is check, and material balance.")
    rejected = ("rejected with the legal-move list" if cfg.push_state
                else "rejected without a legal-move list")

    return (
        f"{head}{_BAR}\n3. BOARD STATE — WHAT YOU SEE\n{_BAR}\n\n{state}\n\n"
        f"{_BAR}\n4. AVAILABLE TOOLS\n{_BAR}\n\n"
        "Each turn you MUST eventually call `move` (or `resign`) to end your turn. Before\n"
        "that you SHOULD `think`, and you MAY use auxiliary tools:\n\n"
        + "\n".join(tools) +
        f"\n\nRecommended flow: {flow}\n\n"
        "After each `move` the environment returns objective facts only (it never tells\n"
        f"you who is winning — assessing the position is YOUR job): {after}\n\n"
        f"{_BAR}\n5. CONSTRAINTS\n{_BAR}\n\n"
        "  • You can ONLY act via the provided commands/tools; plain text is rejected.\n"
        "  • You MUST call `think` BEFORE `move`.\n"
        "  • A turn ends only on `move` or `resign`. Auxiliary tools do not end the turn.\n"
        "  • If you exceed {max_actions} actions in a turn without moving, you FORFEIT.\n"
        f"  • Illegal moves are {rejected}; {{max_invalid}} invalid move\n"
        "    attempts in one turn FORFEIT the game.\n"
        "  • Keep the attack forcing. Quiet moves usually throw away the win.\n"
    )


def build_repl_prompt(
    *,
    enable_view_board: bool = True,
    enable_simulate: bool = True,
    enable_legal_moves: bool = True,
    push_state: bool = True,
) -> str:
    """INTERACTION PROTOCOL section listing only the commands the arm accepts."""
    lines = ["  move <from><to>            Play a move in ICCS format (e.g. `move h2e2`). ENDS your turn."]
    if enable_simulate:
        lines.append(
            "  simulate <m1> <m2> ...     Forward-test a hypothetical line you supply for BOTH\n"
            "                             sides: [your move, assumed opponent move, ...]\n"
            "                             (e.g. `simulate h2e2 h9g7 b0c2`)."
        )
    if enable_view_board:
        lines.append("  view_board                 Show the board, FEN, and the ground-truth piece list.")
    if enable_legal_moves:
        lines.append("  legal_moves                List all legal moves right now.")
    lines.append("  think <reasoning>          Record private reasoning (does NOT end your turn).")
    lines.append("  history                    Show the move history.")
    lines.append("  resign                     Forfeit the game. ENDS your turn.")
    commands = "\n".join(lines)

    look = [c for c, on in (("`view_board`", enable_view_board),
                            ("`simulate ...`", enable_simulate),
                            ("`legal_moves`", enable_legal_moves)) if on]
    if look[:2] == ["`view_board`", "`simulate ...`"]:
        look_line = ("  • One command per turn-step. To look before you leap, send `view_board` or\n"
                     "    `simulate ...` first, read the output, then send `move ...`.\n")
    elif look:
        look_line = (f"  • One command per turn-step. To look before you leap, send "
                     f"{' or '.join(look)}\n    first, read the output, then send `move ...`.\n")
    else:
        look_line = "  • One command per turn-step; `think ...` does not end your turn.\n"
    rejected = ("  • Illegal or unknown commands are rejected with the list of legal moves; keep\n"
                if push_state else
                "  • Illegal or unknown commands are rejected (no legal-move list is shown); keep\n")
    rules_tail = (
        "  • Put your command in a ```bash``` block. Only the LAST block is executed.\n"
        f"{look_line}"
        "  • If you emit no command block, you will be asked again.\n"
        f"{rejected}"
        "    trying — you do NOT lose your turn for one mistake, but repeated invalid\n"
        "    commands in a single turn will be resolved against you."
    )
    return f"""
{_BAR}
INTERACTION PROTOCOL (read carefully)
{_BAR}

You control the game through a command-line interface, like a terminal.
This is the ONLY way you may act. Ignore any mention elsewhere of JSON tool
calls or `function(args)` syntax — in this game you act by typing commands.

On each of your turns:
  1. Think out loud in plain text (your private reasoning — the opponent never
     sees it).
  2. Then issue EXACTLY ONE command inside a fenced code block:

```bash
move h2e2
```

The environment runs your command and replies with terminal output (prefixed
with `$ <command>`), then it is your turn again until you end it with a move.

Available commands (one per block):
{commands}

Rules:
{rules_tail}
""".strip()
