"""REPL command protocol.

The agent writes prose (its reasoning) plus a single command inside a fenced
code block:

    I'll centralize my cannon.

    ```bash
    move h2e2
    ```

`parse_action` turns that into ``(thinking, action, params)`` for
:meth:`GameHarness.execute`. Validation of the command itself happens in the
action executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Fenced code block: ```lang\n ... \n``` (lang optional). DOTALL for body.
_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)

# Command token (after an optional `$` / `chess` prefix) -> harness action.
# Draw commands are still parsed so they receive the "disabled" reply.
_COMMAND_ALIASES = {
    "move": "move",
    "simulate": "simulate",
    "sim": "simulate",
    "view_board": "view_board",
    "board": "view_board",
    "view": "view_board",
    "legal_moves": "get_legal_moves",
    "get_legal_moves": "get_legal_moves",
    "moves": "get_legal_moves",
    "history": "get_history",
    "get_history": "get_history",
    "resign": "resign",
    "offer_draw": "offer_draw",
    "draw": "offer_draw",
    "accept_draw": "accept_draw",
    "think": "think",
}


@dataclass
class ParsedAction:
    thinking: str = ""
    action: str | None = None
    params: dict = field(default_factory=dict)
    raw_command: str = ""
    error: str | None = None


def _extract_command_block(text: str) -> tuple[str, str]:
    """Return (thinking, command) where command is the first meaningful line of
    the LAST fenced code block, and thinking is the text with blocks removed."""
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        return text.strip(), ""

    last = matches[-1]
    body = last.group(1)
    # First non-empty, non-comment line is the command.
    command = ""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        command = stripped
        break

    thinking = _FENCE_RE.sub("", text).strip()
    return thinking, command


def _normalize_command(command: str) -> str:
    """Strip an optional leading `$ ` shell prompt or `chess` program name."""
    cmd = command.strip()
    if cmd.startswith("$"):
        cmd = cmd[1:].strip()
    tokens = cmd.split()
    if tokens and tokens[0].lower() == "chess":
        tokens = tokens[1:]
    return " ".join(tokens)


def parse_action(text: str) -> ParsedAction:
    """Parse an LLM REPL response into a ParsedAction.

    - thinking: prose outside the code block (records reasoning even when the
      model returns no separate 'content').
    - action/params: derived from the command in the last code block.
    - error: set when no usable command is present (caller should re-prompt).
    """
    thinking, raw_command = _extract_command_block(text)
    if not raw_command:
        return ParsedAction(
            thinking=thinking,
            error="No command found. Put exactly one command inside a "
                  "```bash``` code block, e.g. ```bash\\nmove h2e2\\n```.",
        )

    normalized = _normalize_command(raw_command)
    tokens = normalized.split()
    if not tokens:
        return ParsedAction(thinking=thinking, raw_command=raw_command,
                            error="Empty command.")

    verb = tokens[0].lower()
    args = tokens[1:]
    action = _COMMAND_ALIASES.get(verb, verb)  # unknown -> passthrough token

    params: dict = {}
    if action == "move":
        params = {"from_to": args[0]} if args else {"from_to": ""}
    elif action == "simulate":
        params = {"moves": args}
    elif action == "think":
        # Everything after `think ` is the reasoning text.
        reasoning = normalized[len(tokens[0]):].strip()
        params = {"reasoning": reasoning}
    # view_board / get_legal_moves / get_history / resign / draws take no params.

    return ParsedAction(
        thinking=thinking,
        action=action,
        params=params,
        raw_command=raw_command,
    )


def render_observation(action_result, command: str) -> str:
    """Render a harness ActionResult as terminal stdout, echoing the command."""
    return f"$ {command}\n{action_result.feedback}"

