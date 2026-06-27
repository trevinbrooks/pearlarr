"""Tests for the single-instance data-dir run lock."""

from seadexarr.modules.runlock import single_instance_lock


class TestSingleInstanceLock:
    def test_second_acquire_in_same_dir_is_blocked(self, tmp_path) -> None:
        with single_instance_lock(str(tmp_path)) as first:
            assert first is True
            # A second run pointed at the same data dir must be refused.
            with single_instance_lock(str(tmp_path)) as second:
                assert second is False
        # Released on exit -> can acquire again.
        with single_instance_lock(str(tmp_path)) as again:
            assert again is True

    def test_different_dirs_do_not_contend(self, tmp_path) -> None:
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        # Intentional parallel instances each get their own data dir / lock.
        with single_instance_lock(str(d1)) as a, single_instance_lock(str(d2)) as b:
            assert a is True
            assert b is True
