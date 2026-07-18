# pyright: strict
"""Troubleshooting anchors: every quoted message fragment still exists in the source.

docs/troubleshooting.md opens each section with the message it is about,
quoted in a fenced `text` block with `...` standing for interpolated parts.
These tests grep every literal fragment against the package source, so
rewording a message without updating its troubleshooting quote fails the
suite (the docs-standard anchor-grep rule).
"""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC = _REPO_ROOT / "docs" / "troubleshooting.md"

# Shorter fragments ("at ...", "is not") are connectors, not anchors. Only
# distinctive spans are grepped.
_MIN_FRAGMENT = 8


def _anchor_fragments() -> list[str]:
    """Every literal fragment of every fenced `text` anchor block in the doc."""

    document = _DOC.read_text(encoding="utf-8")
    fragments: list[str] = []
    for block in re.findall(r"```text\n(.*?)```", document, re.DOTALL):
        for line in block.splitlines():
            fragments.extend(part.strip() for part in line.split("...") if len(part.strip()) >= _MIN_FRAGMENT)
    return fragments


def test_the_doc_still_carries_anchor_blocks() -> None:
    # The mechanism only guards what is quoted. A doc rewritten into pure prose
    # would pass the grep vacuously, so a floor keeps the convention in use.
    assert len(_anchor_fragments()) >= 15


def test_every_anchor_fragment_greps_in_the_source() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted((_REPO_ROOT / "pearlarr").rglob("*.py")))
    missing = [fragment for fragment in _anchor_fragments() if fragment not in source]
    assert missing == [], f"troubleshooting.md quotes messages that no longer exist: {missing}"
