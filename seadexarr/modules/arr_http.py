"""Shared raw-endpoint HTTP for the arr clients.

The clients wrap most of their API surface per endpoint; helpers here hold the
request/parse/fail-open boilerplate one endpoint shares across both arrs
(history today - a migration target for the older per-client endpoints).
"""

import logging
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlencode

import requests

from .seadex_types import ARR_REQUEST_TIMEOUT_S, HistoryRecord


def fetch_history_since(
    session: requests.Session,
    base_url: str,
    api_key: str,
    logger: logging.Logger,
    date: str,
    *,
    arr_label: str,
    include_flags: Mapping[str, str],
    item_key: str,
) -> list[HistoryRecord] | None:
    """History records since ``date`` (``/api/v3/history/since``, ascending), or None.

    One unfiltered call (``eventType`` is single-valued server-side; the activity
    scan filters client-side). Fails open with a warning - None on a request
    error, a non-200, a non-JSON body (e.g. a proxy login page), or a non-array
    payload - so a broken endpoint can never abort the run.

    Args:
        session (requests.Session): The client's shared session.
        base_url (str): The arr's base url (no trailing slash).
        api_key (str): The arr's API key.
        logger (logging.Logger): Warn sink for the fail-open paths.
        date (str): ISO8601 lower bound (arr-clock, inclusive).
        arr_label (str): "Sonarr"/"Radarr", for the warning text.
        include_flags (Mapping[str, str]): The arr's include-* query params.
        item_key (str): The record's item-id field (``seriesId``/``movieId``).
    """

    params = urlencode({"date": date, **include_flags, "apikey": api_key})
    try:
        response = session.get(f"{base_url}/api/v3/history/since?{params}", timeout=ARR_REQUEST_TIMEOUT_S)
    except requests.RequestException:
        logger.warning(f"Could not fetch {arr_label} history (request failed)")
        return None
    if response.status_code != 200:
        logger.warning(f"Could not fetch {arr_label} history (status code {response.status_code})")
        return None
    try:
        payload: object = response.json()
    except ValueError:
        logger.warning(f"Could not fetch {arr_label} history (non-JSON body)")
        return None
    if not isinstance(payload, list):
        logger.warning(f"Could not fetch {arr_label} history (unexpected payload)")
        return None

    # Element dicts are unvalidated JSON: cast at the parse boundary, skip strays.
    records = cast("list[object]", payload)
    return [
        HistoryRecord.from_api(cast("dict[str, Any]", record), item_key=item_key)
        for record in records
        if isinstance(record, dict)
    ]
