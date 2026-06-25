"""Minimal local type stubs for the (un-stubbed) ``arrapi`` package.

Covers ONLY the surface ``sonarr_client.py`` / ``radarr_client.py`` use:
constructing ``SonarrAPI`` / ``RadarrAPI`` and calling ``all_series`` /
``all_movies``. arrapi ships no ``py.typed`` (and no inline annotations pyright
honors under strict), so without this stub every member read is ``Unknown``,
tripping ``reportMissingTypeStubs`` on the import and ``reportUnknownMemberType``
downstream.

The returned ``_SeriesItem`` / ``_MovieItem`` shapes expose the attribute
surface the clients narrow to via ``cast("list[SonarrItem]", ...)`` /
``cast("list[RadarrItem]", ...)`` (``id`` / ``title`` / ``imdbId`` /
``monitored`` plus the per-arr key ``tvdbId`` / ``tmdbId``), so those casts stay
valid. Extra attributes the project doesn't read are intentionally omitted.
"""

from requests import Session

class _SeriesItem:
    id: int
    title: str
    imdbId: str | None
    monitored: bool
    tvdbId: int

class _MovieItem:
    id: int
    title: str
    imdbId: str | None
    monitored: bool
    tmdbId: int

class SonarrAPI:
    def __init__(
        self, url: str, apikey: str, session: Session | None = ...,
    ) -> None: ...
    def all_series(self) -> list[_SeriesItem]: ...

class RadarrAPI:
    def __init__(
        self, url: str, apikey: str, session: Session | None = ...,
    ) -> None: ...
    def all_movies(self) -> list[_MovieItem]: ...
