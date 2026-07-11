# pyright: strict
"""The raw-logging ban (PR7): first-party WARNING+ goes through ``hub_note``.

Ruff's LOG015 bans the module-level ``logging.*`` convenience calls, but no
lint rule can see a method call on an instance attribute — this canary is the
enforcement for ``logger.warning(...)`` / ``self.logger.error(...)`` shapes,
including the live stdlib aliases ``warn`` and ``fatal``. ``debug`` is
deliberately exempt: DEBUG chatter stays raw forever (the bridge files it).
The allowlist names the one sanctioned straggler — ``setup_logger``'s
invalid-level critical in log.py — scoped to its enclosing function and pinned
to exactly one call, so a second raw site even in that function trips.
"""

import ast
from pathlib import Path
from typing import NamedTuple

PACKAGE = Path(__file__).resolve().parent.parent / "pearlarr"

BANNED_METHODS = frozenset({"warning", "warn", "error", "critical", "fatal", "exception", "info", "log"})

# (path relative to the package, enclosing function, method) — the sanctioned straggler.
ALLOWED = frozenset({("modules/log.py", "setup_logger", "critical")})


class RawCall(NamedTuple):
    """One banned-method call on a logger receiver."""

    rel: str
    function: str | None
    method: str
    line: int


def _is_logger_receiver(node: ast.expr) -> bool:
    """True when the call receiver's terminal identifier is a logger binding."""

    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name in {"logger", "_logger"}
        case _:
            return False


def _raw_logging_calls(tree: ast.AST, rel: str) -> list[RawCall]:
    """Every banned raw-logging call in ``tree``, with its enclosing function."""

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def enclosing_function(node: ast.AST) -> str | None:
        cursor = parents.get(node)
        while cursor is not None:
            if isinstance(cursor, ast.FunctionDef | ast.AsyncFunctionDef):
                return cursor.name
            cursor = parents.get(cursor)
        return None

    # ast.walk is breadth-first; sort back into source order.
    return sorted(
        (
            RawCall(rel, enclosing_function(node), node.func.attr, node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in BANNED_METHODS
            and _is_logger_receiver(node.func.value)
        ),
        key=lambda call: call.line,
    )


def test_the_scanner_catches_every_receiver_shape_and_exempts_debug() -> None:
    # The canary is only as good as its own detector: pin the shapes it must
    # catch (bare, self., self._, chained, the warn/fatal aliases), the DEBUG
    # exemption, and function attribution (the allowlist key).
    source = (
        "logger.warning('a')\n"
        "self.logger.error('b')\n"
        "def inner():\n"
        "    self._logger.info('c')\n"
        "    self._mgr.logger.critical('d')\n"
        "logger.warn('alias')\n"
        "self.logger.fatal('alias')\n"
        "logger.debug('exempt')\n"
        "self.console.log('not a logger')\n"
    )
    hits = _raw_logging_calls(ast.parse(source), "synthetic.py")
    assert hits == [
        RawCall("synthetic.py", None, "warning", 1),
        RawCall("synthetic.py", None, "error", 2),
        RawCall("synthetic.py", "inner", "info", 4),
        RawCall("synthetic.py", "inner", "critical", 5),
        RawCall("synthetic.py", None, "warn", 6),
        RawCall("synthetic.py", None, "fatal", 7),
    ]


def test_no_raw_first_party_logging_above_debug() -> None:
    calls: list[RawCall] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        calls.extend(_raw_logging_calls(ast.parse(path.read_text(encoding="utf-8")), rel))

    offenders = [c for c in calls if (c.rel, c.function, c.method) not in ALLOWED]
    assert not offenders, (
        "raw first-party logging is retired; emit through output.runtime.hub_note "
        "(DEBUG stays raw; see this file's docstring):\n"
        + "\n".join(f"pearlarr/{c.rel}:{c.line} .{c.method}(...) in {c.function or '<module>'}" for c in offenders)
    )
    # The straggler stays EXACTLY one call: a second raw site sharing its
    # (file, function, method) key must trip, not slide under the allowlist.
    assert len(calls) - len(offenders) == 1
