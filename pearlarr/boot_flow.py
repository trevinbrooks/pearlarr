"""The boot flow's producer facade: mint scopes, emit events, never render.

Startup does real work before any series is scanned — read+validate config,
download/parse the id-mapping sources, open the cache, log into qBittorrent,
fetch the library, prefetch AniList + SeaDex metadata. The composition root
(`bootstrap.py`) drives that work through `BootFlow`: the banner facts
ride a `RunStarted` event, each IO step runs inside a
`StepScope` (timed + graduated by the renderers), and the
section mark keeps diagnostics fired BETWEEN steps placed at the boot-ledger
indent. Rendering lives entirely on the hub's surfaces (the RichRenderer's boot
region, the file/plain/json text sinks) — this module emits facts only.

A run scans each configured arr in turn; each scan's per-item logging must start
with no live region above it, so `end_section` caps a section (emitting
the `ready in Xs` capstone when it earned one) and the next `step`
reopens a fresh one. Every method is total: emission rides the hub (which contains
renderer bugs), so presentation can never abort the startup work it wraps.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import final

from .output import (
    BootReady,
    CountsMark,
    RunStarted,
    ScopeFactory,
    ScopeKind,
    ScopeMark,
    StepScope,
    current_hub,
    emit_to_hub,
)


@dataclass(frozen=True, slots=True)
class _CapstoneWindow:
    """The capstone gate's window: opened at a section's first step, closed by `end_section`.

    Opening at the first step keeps the banner→step gap — which holds only
    import work — out of the capstone's timing.
    """

    started_at: float
    counts_mark: CountsMark


@final
class BootFlow:
    """The small facade the composition root drives while starting a run.

    The root emits the `banner`, runs each IO step inside `step`,
    calls `end_section` right before a per-arr scan starts logging (so the
    renderers tear their live region down first), and `close` once at the
    end — the safety net when a section was left open.
    """

    def __init__(self, data_dir: str = "", *, clock: Callable[[], float] = time.monotonic) -> None:
        self._data_dir = data_dir
        self._clock = clock
        self._steps = ScopeFactory(emit_to_hub, clock=clock)
        self._section = ScopeMark(ScopeKind.BOOT_SECTION, "boot")
        self._window: _CapstoneWindow | None = None
        self._section_failed = False

    def banner(self) -> None:
        """State the banner facts (version, data dir), then open the boot section.

        Order matters: `RunStarted` is the run boundary (the fold closes any
        stale nodes on it), so the section mark opens after it.
        """

        emit_to_hub(RunStarted(version=_app_version(), data_dir=self._data_dir))
        self._section.open()

    @contextlib.contextmanager
    def step(self, label: str) -> Generator[StepScope]:
        """Run one IO step as a StepScope: started/progress/slow/finished events out.

        The caller's exception still propagates (the step just graduates FAILED
        first); a failed step suppresses the section's capstone.
        """

        self._section.open()
        if self._window is None:
            self._window = _CapstoneWindow(self._clock(), current_hub().counts.bound_mark())
        try:
            with self._steps.step(label) as scope:
                yield scope
        except BaseException:
            # StepScope's __exit__ graduated the step FAILED; the section-level
            # verdict is ours (the scope knows nothing of the section).
            self._section_failed = True
            raise

    def end_section(self) -> None:
        """Cap the section: emit the capstone (when earned), close the scope mark.

        Calling this inside an open `step` body is unsupported.
        """

        self._emit_capstone()
        self._section.close()
        self._window = None
        self._section_failed = False

    def close(self) -> None:
        """Final teardown (idempotent) — safety net if a section was left open."""

        self.end_section()

    def _emit_capstone(self) -> None:
        # No "ready" line on an empty or failed section, or one that recorded an
        # ERROR+ without raising (a refused arm mid-section): the error already
        # carries the meaning, and claiming "ready" after it would lie.
        window = self._window
        if window is None or self._section_failed:
            return
        # The mark carries the counter it was stamped on (a hub swap can't skew the
        # diff); counts exclude file_only forensics, so only an ERROR a visible
        # surface could show suppresses the ready line.
        if window.counts_mark.since().errors > 0:
            return
        emit_to_hub(BootReady(elapsed_s=self._clock() - window.started_at))


def _app_version() -> str:
    """The installed package version as `"vX.Y.Z"` (empty if undeterminable)."""

    try:
        return f"v{version('pearlarr')}"
    except PackageNotFoundError:  # pragma: no cover - only when run from a non-install
        return ""
