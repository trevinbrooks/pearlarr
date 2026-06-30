# pyright: strict
"""Tests for the single-instance data-dir run lock."""

from pathlib import Path

import pytest

from seadexarr.modules.runlock import single_instance_lock


class TestSingleInstanceLock:
    def test_second_acquire_in_same_dir_is_blocked(self, tmp_path: Path) -> None:
        with single_instance_lock(str(tmp_path)) as first:
            assert first is True
            # A second run pointed at the same data dir must be refused.
            with single_instance_lock(str(tmp_path)) as second:
                assert second is False
        # Released on exit -> can acquire again.
        with single_instance_lock(str(tmp_path)) as again:
            assert again is True

    def test_different_dirs_do_not_contend(self, tmp_path: Path) -> None:
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        # Intentional parallel instances each get their own data dir / lock.
        with single_instance_lock(str(d1)) as a, single_instance_lock(str(d2)) as b:
            assert a is True
            assert b is True

    def test_nonexistent_data_dir_degrades_to_noop(self, tmp_path: Path) -> None:
        # A missing data dir makes os.open fail; the guard must degrade to a
        # best-effort True (not raise) so the run reaches config validation,
        # which surfaces the real, clean error.
        missing = tmp_path / "does-not-exist"
        with single_instance_lock(str(missing)) as acquired:
            assert acquired is True

    def test_no_fcntl_branch_degrades_to_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Where fcntl is unavailable (e.g. Windows) the guard is a no-op.
        import seadexarr.modules.runlock as rl

        monkeypatch.setattr(rl, "fcntl", None)
        with single_instance_lock("/nonexistent/path/should/never/be/touched") as acquired:
            assert acquired is True
