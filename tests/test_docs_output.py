# pyright: strict
"""Output-contract documentation: docs/output.md stays true to the wire.

The generated event catalog is byte-gated by gen_docs `--check`; these tests
pin the hand-authored claims: the wait-summary webhook example's shape against
a real captured POST, and the stated `schema_version` against the constant.
"""

import json
import re
from pathlib import Path
from typing import cast

import httpx
import respx

from pearlarr.config import Arr
from pearlarr.manual_import import Outcome
from pearlarr.notify import Notifier
from pearlarr.output.textline import JSON_SCHEMA_VERSION
from pearlarr.wait_view import WaitOutcomeRow, WaitResult

_OUTPUT_MD = Path(__file__).resolve().parents[1] / "docs" / "output.md"


def _documented_webhook_example() -> dict[str, object]:
    """The fenced JSON example under the wait-summary webhook heading."""

    document = _OUTPUT_MD.read_text(encoding="utf-8")
    section = document.partition("### The wait-summary webhook")[2]
    fence = re.search(r"```json\n(.*?)```", section, re.DOTALL)
    assert fence is not None, "docs/output.md lost its wait-summary webhook example"
    return cast("dict[str, object]", json.loads(fence.group(1)))


@respx.mock
def test_wait_webhook_example_matches_the_wire() -> None:
    # The example is hand-authored (the payload is built inline at the POST
    # site), so key presence AND order are pinned against a real captured POST.
    route = respx.post("https://hook.example").respond(json={})
    notifier = Notifier(discord_url=None, webhook_url="https://hook.example", web=httpx.Client())
    result = WaitResult((WaitOutcomeRow("Frieren", Outcome.IMPORTED),), elapsed_s=431.7)

    assert notifier.push_wait_summary(arr=Arr.SONARR, result=result) is True
    wire = cast("dict[str, object]", json.loads(route.calls.last.request.content))
    example = _documented_webhook_example()
    assert list(example) == list(wire)
    wire_rows = cast("list[dict[str, object]]", wire["rows"])
    example_rows = cast("list[dict[str, object]]", example["rows"])
    assert [list(row) for row in example_rows] == [list(row) for row in wire_rows]


def test_documented_schema_version_matches_the_constant() -> None:
    document = _OUTPUT_MD.read_text(encoding="utf-8")
    assert f"`schema_version` is currently `{JSON_SCHEMA_VERSION}`" in document
