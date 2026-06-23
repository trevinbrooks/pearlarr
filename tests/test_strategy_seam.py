"""Seam tests for the Phase 6b composition split (see ``REFACTOR_PLAN.md``).

These pin the contract between the run engine and the Arr strategies: each
``ArrSync`` hook reaches the shared pipeline only through the ``RunServices``
view handed to it per call - never a stored back-reference. The strategies are
built bare (``object.__new__``) so no live Sonarr/Radarr client is constructed.
"""

from unittest import mock

from seadexarr.modules.seadex_radarr import RadarrSync
from seadexarr.modules.seadex_sonarr import SonarrSync

from .builders import make_logger


def _bare(cls, **attrs):
    """A strategy instance with __init__ bypassed and only the read attrs set."""

    obj = object.__new__(cls)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


class _Item:
    """A stand-in Arr item exposing whatever id attributes a test sets."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestItemAnilistIdsDelegates:
    """item_anilist_ids resolves through run.get_anilist_ids, with arr-specific ids."""

    def test_radarr_uses_tmdb_and_imdb(self) -> None:
        strat = _bare(RadarrSync)
        run = mock.MagicMock()
        run.get_anilist_ids.return_value = {7: {}}

        result = strat.item_anilist_ids(run, _Item(tmdbId=42, imdbId="tt7"), log_ignored=False)

        assert result == {7: {}}
        run.get_anilist_ids.assert_called_once_with(
            tmdb_id=42, imdb_id="tt7", tmdb_type="movie", log_ignored=False,
        )

    def test_sonarr_uses_tvdb_and_imdb(self) -> None:
        strat = _bare(SonarrSync)
        run = mock.MagicMock()

        strat.item_anilist_ids(run, _Item(tvdbId=99, imdbId="tt9"))

        run.get_anilist_ids.assert_called_once_with(
            tvdb_id=99, imdb_id="tt9", log_ignored=True,
        )


class TestFilterToSingle:
    """filter_to_single narrows by the arr's external id (no engine needed)."""

    def test_radarr_matches_tmdb_id(self) -> None:
        strat = _bare(RadarrSync, logger=make_logger())
        items = [_Item(tmdbId=1), _Item(tmdbId=2)]

        assert strat.filter_to_single(items, 2) == [items[1]]
        assert strat.filter_to_single(items, 7) == []

    def test_sonarr_matches_tvdb_id(self) -> None:
        strat = _bare(SonarrSync, logger=make_logger())
        items = [_Item(tvdbId=10), _Item(tvdbId=20)]

        assert strat.filter_to_single(items, 10) == [items[0]]


class TestRunStartHook:
    """get_items doubles as the run-start hook: it resets the per-run scratch."""

    def test_sonarr_get_items_clears_ep_list_cache(self) -> None:
        strat = _bare(SonarrSync, _ep_list_cache={5: ["stale"]})
        strat.get_all_sonarr_series = mock.MagicMock(return_value=["series"])

        result = strat.get_items()

        assert result == ["series"]
        assert strat._ep_list_cache == {}


class TestProcessAlIdThreadsRun:
    """The per-id head runs through `run`; a missing SeaDex entry stops this id."""

    def test_radarr_no_seadex_entry_returns_false(self) -> None:
        strat = _bare(RadarrSync)
        run = mock.MagicMock()
        run.al_id_prologue.return_value = None

        assert strat.process_al_id(run, "radarr", _Item(id=1), "Title", 5, {}) is False
        run.al_id_prologue.assert_called_once_with(5)

    def test_sonarr_no_seadex_entry_returns_false(self) -> None:
        strat = _bare(SonarrSync)
        run = mock.MagicMock()
        run.al_id_prologue.return_value = None

        assert strat.process_al_id(run, "sonarr", _Item(id=1), "Title", 5, {}) is False
        run.al_id_prologue.assert_called_once_with(5)
