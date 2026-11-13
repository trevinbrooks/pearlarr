"""A rich-console line with its level, and the printer that puts a batch of them on the shared console."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

from .console_text import (
    ConsoleRender,
    KvLine,
    SectionRule,
    StyledLine,
    TitledRule,
    print_titled_rule,
    render_kv,
    render_rule,
)
from ..log import print_literal


@dataclass(frozen=True, slots=True)
class LegacyLine:
    """One rich-console line: level, plain message, console payload."""

    level: int
    message: str
    payload: ConsoleRender | None = None


def render_legacy_lines(console: Console, lines: Iterable[LegacyLine], level: int) -> None:
    """Render legacy lines on the shared console: a line prints iff its level clears `level`."""

    for line in lines:
        if line.level < level:
            continue
        match line.payload:
            case TitledRule(title=title, style=style, heavy=heavy):
                print_titled_rule(console, title, style, heavy=heavy)
            case SectionRule(char=char):
                console.print(render_rule(char))
            case KvLine() as kv:
                print_literal(console, render_kv(kv))
            case (StyledLine() | None) as payload:
                style = payload.style if payload is not None else ""
                print_literal(console, Text(line.message, style=style))
