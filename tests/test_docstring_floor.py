# pyright: strict
"""The docstring floor for the shared test infrastructure.

Ruff gates docstring presence for modules, packages, and classes repo-wide,
but deliberately not for functions and methods (D102/D103 off - leaf presence
gates manufacture ceremony). `fakes.py` and `builders.py` are the exception:
they are the suite's shared vocabulary, so every public callable in them is
documented. `@override` methods are exempt - their contract lives on the
abstract hook they implement.
"""

import ast
from pathlib import Path

import pytest

_SHARED_INFRA = ("fakes.py", "builders.py")


def _has_override(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "override" for d in node.decorator_list)


def _undocumented(tree: ast.Module) -> list[str]:
    """Names of public classes/callables without a docstring, `Class.method`-qualified."""

    missing: list[str] = []

    def walk(node: ast.Module | ast.ClassDef, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if child.name.startswith("_"):
                continue
            qualified = f"{prefix}{child.name}"
            if isinstance(child, ast.ClassDef):
                if ast.get_docstring(child) is None:
                    missing.append(qualified)
                walk(child, prefix=f"{qualified}.")
            elif not _has_override(child) and ast.get_docstring(child) is None:
                missing.append(qualified)

    walk(tree, prefix="")
    return missing


@pytest.mark.parametrize("filename", _SHARED_INFRA)
def test_shared_test_infra_is_fully_documented(filename: str) -> None:
    """Every public class and non-override callable in the file carries a docstring."""

    source = (Path(__file__).parent / filename).read_text(encoding="utf-8")
    missing = _undocumented(ast.parse(source))
    assert not missing, f"{filename} public symbols missing docstrings: {', '.join(missing)}"
