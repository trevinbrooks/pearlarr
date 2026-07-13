"""The one exception-carrier type for events: frames captured at construction, locals never."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import ClassVar, final

from rich.traceback import Trace, Traceback


@final
@dataclass(frozen=True, slots=True)
class CapturedTrace:
    """A traceback extracted once from a live exception; holding it never pins frames.

    `show_locals=False` at extraction means frame locals (API keys, webhook URLs)
    are never copied out of the frames — the secrets guarantee is structural.
    """

    rich_trace: Trace
    plain: str
    """The full plain-text traceback for the file sink (stdlib format, no locals)."""

    MAX_FRAMES: ClassVar[int] = 10
    """Frames kept per stack at capture (outermost + innermost halves, rich-style elide)."""

    @classmethod
    def from_exception(cls, exc: BaseException) -> CapturedTrace:
        """Capture `exc` now: a bounded rich trace + the full stdlib plain text."""

        trace = Traceback.extract(type(exc), exc, exc.__traceback__, show_locals=False)
        for stack in trace.stacks:
            if len(stack.frames) > cls.MAX_FRAMES:
                head = cls.MAX_FRAMES // 2
                tail = cls.MAX_FRAMES - head
                stack.frames[:] = [*stack.frames[:head], *stack.frames[-tail:]]
        plain = "".join(traceback.format_exception(exc))
        return cls(rich_trace=trace, plain=plain)
