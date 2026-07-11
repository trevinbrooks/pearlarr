"""Single-instance run lock: guard against two runs sharing one data directory.

A run reads and writes `cache.db` (and the WAL) in its data directory; two runs
pointed at the *same* directory would duplicate work and could race on imports.
SQLite's own locking keeps the file from corrupting, but it won't stop the wasted,
overlapping work - so we take a coarse advisory lock on the data dir and skip a run
that finds another already active there.

Scope is deliberately one host / one local filesystem, via `filelock`
(`fcntl.flock` on POSIX, `msvcrt` on Windows - a real lock everywhere, not the
former no-op Windows fallback). Running multiple instances *intentionally* means
giving each its own data directory (its own lock file), which is allowed. There is
no cross-host story here by design - that would need a real network lock, which
this project doesn't want.
"""

import contextlib
import logging
import os
from collections.abc import Generator

from filelock import FileLock, Timeout

from .output import hub_warn

LOCK_FILENAME = ".pearlarr.lock"


@contextlib.contextmanager
def single_instance_lock(
    data_dir: str,
    *,
    logger: logging.Logger | None = None,
) -> Generator[bool]:
    """Hold an advisory lock on `data_dir` for the duration of the `with` block.

    Yields `True` if this process acquired the lock (no other run is active in
    that directory), or `False` if another run already holds it - the caller
    should skip and retry next cycle. Degrades to a best-effort `True` (no real
    lock) when the lock file can't be created (missing/unwritable `data_dir`)
    or the filesystem can't honor the lock (e.g. ENOLCK on some NFS/FUSE mounts) -
    so a guard failure never crashes or silently skips the run.

    Args:
        data_dir: The run's data directory (where `cache.db` lives).
        logger: For a debug note on the lock path.
    """

    lock_path = os.path.join(data_dir, LOCK_FILENAME)
    lock = FileLock(lock_path)
    try:
        lock.acquire(blocking=False)
    except Timeout:
        # Another process holds the lock.
        yield False
        return
    except OSError as e:
        # Degrade to a no-op lock so the run proceeds to config validation,
        # which surfaces the real, clean error - best-effort by design.
        hub_warn(f"Could not take the run lock at {lock_path} ({e}); proceeding without it")
        yield True
        return
    if logger is not None:
        logger.debug(f"Acquired run lock {lock_path}")
    try:
        yield True
    finally:
        lock.release()
