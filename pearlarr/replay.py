"""Re-render a captured JSON event stream as the human text grammar.

`pearlarr replay` reads a capture of the JSON envelope stream - what a run with
`advanced.log_format: json` or a subcommand's `--json` wrote to stdout - and
prints each envelope back as one `ts LEVEL [bracket] message k=v` line, for
reading a docker-captured or archived log after the fact. `-` reads stdin.

The grammar lives once, in `output.textline.render_envelope_line`; this module
owns only IO and policy. The stream is a lossy RENDERING of typed events (not a
serialization), so the formatter is generic and additive-proof: an unknown
newer event name still renders. Docker captures interleave stderr text with the
JSON, so a non-object / malformed line is skipped and counted, not fatal.

The rendered lines are the command's PRODUCT, not events: they go straight to
stdout with `typer.echo` (the same class as `CliTextRenderer`'s echoes), while
the skip count and the schema-mismatch heads-up ride the hub as warnings and
the read-failure / no-events arms as errors.
"""

from __future__ import annotations

import json
import sys

import typer

from .output import JsonValue, hub_error, hub_warn
from .output.textline import JSON_SCHEMA_VERSION, render_envelope_line

_STDIN_ARG = "-"


def replay(source: str) -> bool:
    """Render the JSON envelope capture at `source` (`-` = stdin) as text lines.

    Returns True when at least one event rendered; False (with the reason already
    reported through the hub) when the capture can't be read or held no events.
    """

    lines = _read_lines(source)
    if lines is None:
        return False

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

    if skipped:
        hub_warn(f"Skipped {skipped} non-event line{'s' if skipped != 1 else ''}")
    if rendered == 0:
        label = "stdin" if source == _STDIN_ARG else source
        hub_error(
            f"No events found in {label} - expected a capture of a run's advanced.log_format: json "
            "output, or a subcommand's --json output",
        )
        return False
    return True


def _read_lines(source: str) -> list[str] | None:
    """Every line of the capture, or None (after reporting) when it can't be read."""

    try:
        if source == _STDIN_ARG:
            return sys.stdin.read().splitlines()
        with open(source, encoding="utf-8") as handle:
            return handle.read().splitlines()
    except OSError as e:
        hub_error(f"Cannot read {source} ({e})")
        return None


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
