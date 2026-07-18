"""Re-render a captured JSON event stream as the human text grammar.

`pearlarr replay` reads a capture of the JSON envelope stream - what a run with
`advanced.log_format: json` or a subcommand's `--json` wrote to stdout - and
prints each envelope back as one `ts LEVEL [bracket] message k=v` line, for
reading a docker-captured or archived log after the fact. `-` reads stdin.

The grammar lives once, in `output.textline.render_envelope_line`. This module
owns only IO and policy. The stream is a lossy RENDERING of typed events (not a
serialization), so the formatter is generic and additive-proof: an unknown
newer event name still renders. The capture streams line by line (months of a
container's log never sit in memory whole), and messiness is never fatal:
docker interleaves stderr text with the JSON, so a non-object / malformed line
is skipped and counted, and a non-UTF-8 byte decodes to U+FFFD instead of
aborting the read.

The rendered lines are the command's PRODUCT, not events: they go straight to
stdout with `typer.echo` (the same class as `CliTextRenderer`'s echoes), while
the skip count and the schema-mismatch heads-up ride the hub as warnings and
the read-failure / no-events arms as errors.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from typing import TYPE_CHECKING, NamedTuple

import typer

from .output import JsonValue, hub_error, hub_warn
from .output.textline import JSON_SCHEMA_VERSION, render_envelope_line

if TYPE_CHECKING:
    from collections.abc import Iterable

_STDIN_ARG = "-"


class _RenderCounts(NamedTuple):
    rendered: int
    skipped: int


def replay(source: str) -> bool:
    """Render the JSON envelope capture at `source` (`-` = stdin) as text lines.

    Returns True when at least one event rendered. False (with the reason already
    reported through the hub) when the capture can't be read or held no events.
    """

    try:
        if source == _STDIN_ARG:
            stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
            try:
                counts = _render_stream(stream)
            finally:
                stream.detach()  # the wrapper must never close the process's stdin
        else:
            with open(source, encoding="utf-8", errors="replace") as handle:
                counts = _render_stream(handle)
    except BrokenPipeError:
        # Downstream closed (e.g. `| head`): stop quietly like any line tool. The
        # dup2 keeps the interpreter's exit flush from whining about the dead pipe.
        with contextlib.suppress(OSError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return True
    except OSError as e:
        hub_error(f"Cannot read {source} ({e})")
        return False

    if counts.skipped:
        hub_warn(f"Skipped {counts.skipped} non-event line{'s' if counts.skipped != 1 else ''}")
    if counts.rendered == 0:
        label = "stdin" if source == _STDIN_ARG else source
        hub_error(
            f"No events found in {label} - expected a capture of a run's advanced.log_format: json "
            "output, or a subcommand's --json output",
        )
        return False
    return True


def _render_stream(lines: Iterable[str]) -> _RenderCounts:
    """Echo every renderable envelope in `lines`, counting what rendered and what didn't."""

    rendered = 0
    skipped = 0
    schema_checked = False
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        payload = _parse_object(stripped)
        if payload is None:
            skipped += 1
            continue
        text = render_envelope_line(payload)
        if text is None:
            skipped += 1
            continue
        if not schema_checked:
            # One heads-up per stream, on the first line we recognize as an envelope.
            schema_checked = True
            _warn_on_foreign_schema(payload)
        typer.echo(text)
        rendered += 1
    return _RenderCounts(rendered, skipped)


def _parse_object(line: str) -> dict[str, JsonValue] | None:
    """The line's JSON object, or None when it isn't valid JSON or isn't an object."""

    try:
        parsed: JsonValue = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _warn_on_foreign_schema(payload: dict[str, JsonValue]) -> None:
    """Warn once when the stream's schema version isn't the one this Pearlarr reads."""

    version = payload.get("schema_version")
    if version == JSON_SCHEMA_VERSION:
        return
    descriptor = "no schema_version" if version is None else f"schema_version {version}"
    hub_warn(f"Stream states {descriptor}; this Pearlarr reads {JSON_SCHEMA_VERSION} - rendering best-effort")
