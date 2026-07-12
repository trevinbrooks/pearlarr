# pyright: strict
"""CLI documentation pipeline: help-text completeness and style on the built command tree.

The generator (docs/cli.md) hard-fails on a missing command or option help;
these tests add the style rules it cannot judge: sentence-case starts, no
trailing periods on option help, and no restated config defaults (the
generated configuration reference owns defaults).
"""

import re

import typer.core
import typer.main

from pearlarr.cli import pearlarr_cli

type _Node = typer.core.TyperCommand | typer.core.TyperGroup

# typer's own completion options ship their own help text (trailing period
# included); the style rules govern our strings, not typer's.
_TYPER_BUILTINS = frozenset({"install_completion", "show_completion"})


def _tree() -> list[tuple[str, _Node]]:
    """Every (path, command) pair of the built CLI, root included."""

    root = typer.main.get_command(pearlarr_cli)
    assert isinstance(root, typer.core.TyperGroup)
    nodes: list[tuple[str, _Node]] = [("pearlarr", root)]
    for name, sub in root.commands.items():
        assert isinstance(sub, (typer.core.TyperCommand, typer.core.TyperGroup))
        nodes.append((f"pearlarr {name}", sub))
        if isinstance(sub, typer.core.TyperGroup):
            for leaf_name, leaf in sub.commands.items():
                assert isinstance(leaf, (typer.core.TyperCommand, typer.core.TyperGroup))
                nodes.append((f"pearlarr {name} {leaf_name}", leaf))
    return nodes


def _options() -> list[tuple[str, typer.core.TyperOption]]:
    """Every visible (command path + flag, option) pair, `--help` excluded."""

    pairs: list[tuple[str, typer.core.TyperOption]] = []
    for path, node in _tree():
        pairs.extend(
            (f"{path} {param.opts[0]}", param)
            for param in node.params
            if isinstance(param, typer.core.TyperOption) and not param.hidden and param.name != "help"
        )
    return pairs


def test_every_command_has_help() -> None:
    missing = [path for path, node in _tree() if not (node.help or "").strip()]
    assert missing == []


def test_every_option_has_help() -> None:
    missing = [path for path, param in _options() if not (param.help or "").strip()]
    assert missing == []


def test_option_help_is_sentence_case_without_trailing_period() -> None:
    offenders = [
        path
        for path, param in _options()
        if param.name not in _TYPER_BUILTINS
        and param.help is not None
        and (not param.help[0].isupper() or param.help.rstrip().endswith("."))
    ]
    assert offenders == []


def test_option_help_never_restates_config_defaults() -> None:
    # The generated configuration reference owns defaults; help prose restating
    # them is the drift the single-source pipeline exists to kill.
    banned = re.compile(r"[Dd]efaults? to|[Dd]efault:|\(default")
    offenders = [path for path, param in _options() if param.help and banned.search(param.help)]
    assert offenders == []
