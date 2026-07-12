# pyright: strict
"""Tests for `output.trace` — the one exception carrier.

Pin the secrets contract (frame locals are NEVER captured: an api_key local must
not appear anywhere in the value), the bounded rich frame count, and the full
plain-text rendering the file sink stores.
"""

from pearlarr.output import CapturedTrace

_SECRET = "hunter2-super-secret-key"


def _leak() -> None:
    # The secret must live in the raising frame's locals without riding the message.
    api_key = _SECRET
    raise ValueError(f"request failed ({len(api_key)} chars)")


def _capture() -> CapturedTrace:
    try:
        _leak()
    except ValueError as exc:
        return CapturedTrace.from_exception(exc)
    raise AssertionError("unreachable")


def test_plain_text_carries_the_full_stdlib_traceback() -> None:
    trace = _capture()

    plain = trace.plain
    assert "ValueError: request failed" in plain
    assert "_leak" in plain
    assert plain == trace.plain


def test_frame_locals_never_ride_the_trace() -> None:
    trace = _capture()

    assert _SECRET not in trace.plain
    assert _SECRET not in repr(trace.rich_trace)
    assert _SECRET not in repr(trace)


def test_frames_are_bounded_at_construction() -> None:
    def deep(n: int) -> None:
        if n == 0:
            raise ValueError("bottom")
        deep(n - 1)

    try:
        deep(50)
    except ValueError as exc:
        trace = CapturedTrace.from_exception(exc)
    else:  # pragma: no cover - deep() always raises
        raise AssertionError("unreachable")

    for stack in trace.rich_trace.stacks:
        assert len(stack.frames) <= CapturedTrace.MAX_FRAMES
    # The bounded stack keeps both ends: the raise site and the outermost caller.
    frames = trace.rich_trace.stacks[0].frames
    assert frames[0].name == "test_frames_are_bounded_at_construction"
    assert frames[-1].name == "deep"


def test_short_stacks_are_untouched() -> None:
    trace = _capture()

    assert 0 < len(trace.rich_trace.stacks[0].frames) <= CapturedTrace.MAX_FRAMES
