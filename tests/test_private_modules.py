# pyright: strict
"""A private module (`_name`) is imported only from inside its own package.

No lint rule covers it: pyright strict's reportPrivateUsage doesn't flag `from pkg._mod import Name`, and ruff's
PLC2701 skips relative and same-package imports.
"""

import ast
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
ROOTS = ("pearlarr", "tests", "scripts")


class ImportTarget(NamedTuple):
    """One dotted path an import names, and the line it's on."""

    path: tuple[str, ...]
    line: int


def _targets(tree: ast.AST, package: tuple[str, ...]) -> list[ImportTarget]:
    """Every dotted path an import in `tree` names, relative ones resolved against `package`."""

    found: list[ImportTarget] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [ImportTarget(tuple(alias.name.split(".")), node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else ()
            module = (*base, *(node.module.split(".") if node.module else ()))
            found += [ImportTarget((*module, alias.name), node.lineno) for alias in node.names]
    return found


def _private_reach(target: tuple[str, ...], importer: tuple[str, ...]) -> str | None:
    """The private module `target` reaches from outside its package, or None."""

    for depth, part in enumerate(target):
        if not part.startswith("_") or part.startswith("__"):
            continue
        owner = target[:depth]
        on_disk = REPO.joinpath(*owner, part)
        if (on_disk.is_dir() or on_disk.with_suffix(".py").is_file()) and importer[: len(owner)] != owner:
            return ".".join(target[: depth + 1])
    return None


def _reaches(source: str, rel: Path) -> list[str]:
    """Each import in `source`, the file at `rel`, that reaches another package's private module."""

    package = rel.parent.parts
    return [
        f"{rel.as_posix()}:{target.line} imports {hit}"
        for target in _targets(ast.parse(source), package)
        if (hit := _private_reach(target.path, package))
    ]


def test_the_scan_catches_outside_reaches_and_allows_the_package_its_own() -> None:
    outside = Path("pearlarr/window_placement.py")
    assert _reaches("from .placer._state import PlacerState\n", outside) == [
        "pearlarr/window_placement.py:1 imports pearlarr.placer._state",
    ]
    assert _reaches("import pearlarr.placer._sequence_passes\n", Path("tests/x.py")) == [
        "tests/x.py:1 imports pearlarr.placer._sequence_passes",
    ]
    assert _reaches("from pearlarr.placer import _passes\n", Path("scripts/x.py")) == [
        "scripts/x.py:1 imports pearlarr.placer._passes",
    ]
    assert _reaches("from .placer import place_batch\n", outside) == []
    assert _reaches("from ._state import PlacerState\n", Path("pearlarr/placer/_passes.py")) == []


def test_no_module_imports_another_packages_private_module() -> None:
    reaches = [
        reach
        for root in ROOTS
        for path in sorted((REPO / root).rglob("*.py"))
        for reach in _reaches(path.read_text(encoding="utf-8"), path.relative_to(REPO))
    ]
    assert reaches == []
