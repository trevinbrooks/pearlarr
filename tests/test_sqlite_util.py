# pyright: strict
"""The connection settings both stores share."""

import sqlite3

import pytest

from pearlarr.sqlite_util import connect


def test_a_declared_foreign_key_is_enforced() -> None:
    conn = connect(":memory:")
    try:
        conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE child (parent_id INTEGER NOT NULL REFERENCES parent (id))")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO child (parent_id) VALUES (1)")
    finally:
        conn.close()
