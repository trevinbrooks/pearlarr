# pyright: strict
"""Truth table for `radarr_movies_matching` (the ignore_movies_in_radarr check).

The season-0 gate + id-match loop extracted from `SonarrSync.process_al_id`:
only a specials (season-0) mapping can match a Radarr movie, an unset mapping id
never matches (the `is not None` guards make None == None a non-match), and a
movie matching on both ids appears once. `process_al_id` keeps the outer
feature guard and the skip logging; this pins the matching itself.
"""

from pearlarr.modules.mappings import MappingEntry
from pearlarr.modules.seadex_sonarr import radarr_movies_matching
from pearlarr.modules.seadex_types import RadarrMovie


def test_non_zero_seasons_never_match() -> None:
    """Season -1 (the field default) and a regular season match nothing.

    The movie is id-equal on both axes, so this pins that the season gate alone
    (movies ride along only as season-0 specials) blocks the match.
    """

    movie = RadarrMovie(title="Movie", tmdbId=42, imdbId="tt0000042")
    for season in (-1, 1):
        mapping = MappingEntry(anilist_id=1, tvdb_season=season, tmdb_movie_id=42, imdb_id="tt0000042")
        assert radarr_movies_matching(mapping, [movie]) == []


def test_season_zero_matches_by_tmdb_id() -> None:
    """A season-0 mapping matches a movie on the TMDB id alone."""

    mapping = MappingEntry(anilist_id=1, tvdb_season=0, tmdb_movie_id=42)
    movie = RadarrMovie(title="Movie", tmdbId=42)
    assert radarr_movies_matching(mapping, [movie]) == [movie]


def test_season_zero_matches_by_imdb_id() -> None:
    """A season-0 mapping matches a movie on the IMDb id alone."""

    mapping = MappingEntry(anilist_id=1, tvdb_season=0, imdb_id="tt0000042")
    movie = RadarrMovie(title="Movie", imdbId="tt0000042")
    assert radarr_movies_matching(mapping, [movie]) == [movie]


def test_a_movie_matching_both_ids_appears_once() -> None:
    """Both ids matching is one append, not two (the either/or is per movie)."""

    mapping = MappingEntry(anilist_id=1, tvdb_season=0, tmdb_movie_id=42, imdb_id="tt0000042")
    movie = RadarrMovie(title="Movie", tmdbId=42, imdbId="tt0000042")
    assert radarr_movies_matching(mapping, [movie]) == [movie]


def test_unset_mapping_ids_never_match() -> None:
    """Mapping ids of None match nothing - None == None is NOT a match.

    Pins the `is not None` guards: a movie whose `imdbId` is also None (and
    one carrying the `tmdbId=0` model default) must not pair with an id-less
    mapping, or every id-less special would "already be in Radarr".
    """

    mapping = MappingEntry(anilist_id=1, tvdb_season=0)
    movies = [
        RadarrMovie(title="No imdb", imdbId=None),
        RadarrMovie(title="Default tmdb", tmdbId=0),
    ]
    assert radarr_movies_matching(mapping, movies) == []


def test_multiple_movies_keep_matches_in_order() -> None:
    """Non-matching movies drop out; the matching ones keep library order."""

    mapping = MappingEntry(anilist_id=1, tvdb_season=0, tmdb_movie_id=42, imdb_id="tt0000042")
    first = RadarrMovie(title="First", tmdbId=42)
    other = RadarrMovie(title="Other", tmdbId=7)
    last = RadarrMovie(title="Last", imdbId="tt0000042")
    result = radarr_movies_matching(mapping, [first, other, last])
    assert result == [first, last]
