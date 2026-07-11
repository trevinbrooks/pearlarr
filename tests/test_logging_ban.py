# pyright: strict
"""The raw-logging ban (PR7): first-party WARNING+ goes through ``hub_note``.

Ruff's LOG015 bans the module-level ``logging.*`` convenience calls, but no
lint rule can see a method call on an instance attribute — this canary is the
enforcement for ``logger.warning(...)`` / ``self.logger.error(...)`` shapes.
``debug`` is deliberately exempt: DEBUG chatter stays raw forever (the bridge
files it). The allowlist names the one sanctioned straggler: ``setup_logger``'s
invalid-level complaint in log.py, which the bridge adopts.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "seadexarr"

BANNED_METHODS = frozenset({"warning", "error", "critical", "exception", "info", "log"})

# (path relative to the package, method) pairs allowed to stay raw.
ALLOWED = frozenset({("modules/log.py", "critical")})


def _is_logger_receiver(node: ast.expr) -> bool:
    """True when the call receiver's terminal identifier is a logger binding."""

    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name in {"logger", "_logger"}
        case _:
            return False


def _offenders(tree: ast.AST, rel: str) -> list[str]:
    """Every banned raw-logging call in ``tree``, as ``path:line method`` strings."""

    found: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in BANNED_METHODS
            and _is_logger_receiver(node.func.value)
            and (rel, node.func.attr) not in ALLOWED
        ):
            found.append(f"seadexarr/{rel}:{node.lineno} .{node.func.attr}(...)")
    return found


def test_the_scanner_catches_every_receiver_shape_and_exempts_debug() -> None:
    # The canary is only as good as its own detector: pin the shapes it must
    # catch (bare, self., self._, chained) and the DEBUG exemption.
    source = (
        "logger.warning('a')\n"
        "self.logger.error('b')\n"
        "self._logger.info('c')\n"
        "self._mgr.logger.critical('d')\n"
        "logger.debug('exempt')\n"
        "self.console.log('not a logger')\n"
    )
    hits = _offenders(ast.parse(source), "synthetic.py")
    assert hits == [
        "seadexarr/synthetic.py:1 .warning(...)",
        "seadexarr/synthetic.py:2 .error(...)",
        "seadexarr/synthetic.py:3 .info(...)",
        "seadexarr/synthetic.py:4 .critical(...)",
    ]


def test_no_raw_first_party_logging_above_debug() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        offenders.extend(_offenders(ast.parse(path.read_text(encoding="utf-8")), rel))
    assert not offenders, (
        "raw first-party logging is retired; emit through output.runtime.hub_note "
        "(DEBUG stays raw; see this file's docstring):\n" + "\n".join(offenders)
    )
