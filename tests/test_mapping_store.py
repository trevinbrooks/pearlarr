# pyright: strict
"""Unit tests for ``MappingStore``: the freshness gate, atomic populate, schema
rebuild on version change, corruption fail-open, and cross-open durability.

These pin the guarantees the parse-once-per-download cache relies on - in
particular that a torn populate can never leave a digest marked fresh against
empty tables, and that an unchanged source persists across process restarts.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from seadexarr.modules.mapping_store import (
    SCHEMA_VERSION,
    SOURCE_ANIDB,
    SOURCE_ANIME_IDS,
    AniBridgeRows,
    AniBridgeXrefRow,
    AnimeIdRow,
    MappingStore,
)

# (anilist_id, tvdb_id, tvdb_season, tvdb_epoffset, tmdb_movie_id, imdb_id, anidb_id)
ROW = AnimeIdRow(100, 200, 2, 3, None, "tt100", 50)
ROW2 = AnimeIdRow(101, 201, -1, 0, None, None, None)


class TestFreshnessGate:
    def test_is_fresh_only_for_matching_digest(self) -> None:
        store = MappingStore.open(":memory:")
        assert not store.is_fresh(SOURCE_ANIME_IDS, "d1")
        store.replace_anime_ids("d1", [ROW])
        assert store.is_fresh(SOURCE_ANIME_IDS, "d1")
        assert not store.is_fresh(SOURCE_ANIME_IDS, "d2")
        store.close()

    def test_replace_swaps_digest_and_rows(self) -> None:
        store = MappingStore.open(":memory:")
        store.replace_anime_ids("d1", [ROW])
        store.replace_anime_ids("d2", [])  # new content happens to be empty
        assert store.is_fresh(SOURCE_ANIME_IDS, "d2")
        assert not store.is_fresh(SOURCE_ANIME_IDS, "d1")
        assert store.anime_ids_lookup("tvdb_id", 200) == []
        store.close()

    def test_replace_marks_source_fresh_for_its_digest(self) -> None:
        # An empty populate still stamps the digest, so the source reads fresh for
        # that digest (and not for any other) - the gate a re-parse decision hinges on.
        store = MappingStore.open(":memory:")
        assert not store.is_fresh(SOURCE_ANIDB, "d")
        store.replace_anidb("d", [], [])
        assert store.is_fresh(SOURCE_ANIDB, "d")
        store.close()


class TestAtomicReplace:
    """A populate that errors mid-write must roll back to the prior state, never
    leaving the digest fresh against empty/partial tables."""

    def test_failed_populate_keeps_prior_state(self) -> None:
        store = MappingStore.open(":memory:")
        store.replace_anime_ids("d1", [ROW])

        def exploding_rows() -> Iterator[AnimeIdRow]:
            yield ROW2
            raise RuntimeError("boom mid-populate")

        with pytest.raises(RuntimeError):
            store.replace_anime_ids("d2", exploding_rows())

        # The DELETE + partial insert rolled back: still the committed d1 state.
        assert store.is_fresh(SOURCE_ANIME_IDS, "d1")
        assert not store.is_fresh(SOURCE_ANIME_IDS, "d2")
        assert store.anime_ids_lookup("tvdb_id", 200) == [ROW]
        store.close()


class TestSchemaVersion:
    def test_version_mismatch_rebuilds_tables(self, tmp_path: Path) -> None:
        path = str(tmp_path / "mappings.db")
        store = MappingStore.open(path)
        store.replace_anime_ids("d", [ROW])
        store.close()

        # Simulate a different on-disk schema format by changing user_version.
        conn = sqlite3.connect(path)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        conn.commit()
        conn.close()

        store = MappingStore.open(path)
        # Rebuilt from scratch: the old data and meta are gone (safe - re-derived).
        assert not store.is_fresh(SOURCE_ANIME_IDS, "d")
        assert store.anime_ids_lookup("tvdb_id", 200) == []
        store.close()


class TestCorruptionFailOpen:
    def test_garbage_file_is_quarantined_and_store_still_works(self, tmp_path: Path) -> None:
        path = tmp_path / "mappings.db"
        path.write_bytes(b"this is definitely not a sqlite database" * 16)

        store = MappingStore.open(str(path))
        # Fails open onto an in-memory db that is fully usable for this run.
        store.replace_anime_ids("d", [ROW])
        assert store.is_fresh(SOURCE_ANIME_IDS, "d")
        store.close()

        # The unreadable file was moved aside for inspection.
        assert list(tmp_path.glob("mappings.db.corrupt-*"))


class TestDistinctTyping:
    """The per-column/per-axis overloads' runtime halves: anime_ids values come
    from third-party JSON un-coerced (junk is skipped), while anibridge ext ids
    were coerced on write (a mismatch is corruption and raises)."""

    def test_anime_ids_distinct_skips_junk_typed_values(self, tmp_path: Path) -> None:
        path = str(tmp_path / "mappings.db")
        store = MappingStore.open(path)
        store.replace_anime_ids("d", [ROW, ROW2])
        store.close()

        # Plant junk-typed ids behind the store's back (BLOB for the TEXT-affinity
        # imdb column, TEXT for the INTEGER-affinity tvdb column - both survive
        # SQLite's affinity coercion as-is).
        conn = sqlite3.connect(path)
        conn.execute("INSERT INTO anime_ids (anilist_id, tvdb_id, imdb_id) VALUES (102, 'junk', x'ab')")
        conn.commit()
        conn.close()

        store = MappingStore.open(path)
        assert store.anime_ids_distinct("tvdb_id") == {200, 201}
        assert store.anime_ids_distinct("imdb_id") == {"tt100"}
        store.close()

    def test_anibridge_distinct_raises_on_mismatched_stored_type(self, tmp_path: Path) -> None:
        path = str(tmp_path / "mappings.db")
        store = MappingStore.open(path)
        xref = AniBridgeXrefRow("tvdb", 74796, 100)
        store.replace_anibridge("d", AniBridgeRows(entries=[], xrefs=[xref], ranges=[]))
        store.close()

        conn = sqlite3.connect(path)
        conn.execute("INSERT INTO anibridge_xref (axis, ext_id, anilist_id) VALUES ('tvdb', 'junk', 101)")
        conn.commit()
        conn.close()

        store = MappingStore.open(path)
        with pytest.raises(TypeError, match="tvdb"):
            store.anibridge_distinct("tvdb")
        assert store.anibridge_distinct("imdb") == set()
        store.close()


class TestDurability:
    def test_missing_file_created_and_persists_across_opens(self, tmp_path: Path) -> None:
        path = str(tmp_path / "mappings.db")

        store = MappingStore.open(path)
        assert not store.is_fresh(SOURCE_ANIME_IDS, "digest-A")
        store.replace_anime_ids("digest-A", [ROW])
        store.close()

        # A fresh open (a new "process") sees the committed rows + digest.
        store = MappingStore.open(path)
        assert store.is_fresh(SOURCE_ANIME_IDS, "digest-A")
        assert store.anime_ids_lookup("tvdb_id", 200) == [ROW]
        store.close()
