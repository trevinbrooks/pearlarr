"""Single-instance run lock: guard against two runs sharing one data directory.

A run reads and writes ``cache.db`` (and the WAL) in its data directory; two runs
pointed at the *same* directory would duplicate work and could race on imports.
SQLite's own locking keeps the file from corrupting, but it won't stop the wasted,
overlapping work - so we take a coarse advisory lock on the data dir and skip a run
that finds another already active there.

Scope is deliberately one host / one local filesystem: an ``fcntl`` advisory lock
(POSIX). Running multiple instances *intentionally* means giving each its own data
directory (its own lock file), which is allowed. There is no cross-host story here
by design - that would need a real network lock, which this project doesn't want.

On a platform without ``fcntl`` (e.g. Windows) the guard degrades to a no-op
(always "acquired") rather than blocking the run - best-effort, documented.
"""

import contextlib
import errno
import logging
import os
from collections.abc import Generator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

LOCK_FILENAME = ".seadexarr.lock"


@contextlib.contextmanager
def single_instance_lock(
    data_dir: str,
    *,
    logger: logging.Logger | None = None,
) -> Generator[bool]:
    """Hold an advisory lock on ``data_dir`` for the duration of the ``with`` block.

    Yields ``True`` if this process acquired the lock (no other run is active in
    that directory), or ``False`` if another run already holds it - the caller
    should skip and retry next cycle. Degrades to a best-effort ``True`` (no real
    lock) where ``fcntl`` is unavailable, the lock file can't be created
    (missing/unwritable ``data_dir``), or the filesystem can't honor ``flock``
    (e.g. ENOLCK) - so a guard failure never crashes or silently skips the run.

    Args:
        data_dir (str): The run's data directory (where ``cache.db`` lives).
        logger (logging.Logger | None): For a debug note on the lock path.
    """

    if fcntl is None:
        yield True
        return

    lock_path = os.path.join(data_dir, LOCK_FILENAME)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as e:
        # The lock file can't be created (missing/unwritable ``data_dir``).
        # Degrade to a no-op lock so the run proceeds to config validation,
        # which surfaces the real, clean error - best-effort, like the
        # ``fcntl``-unavailable fallback above.
        if logger is not None:
            logger.warning(f"Could not create run lock {lock_path}: {e}; proceeding without it")
        yield True
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                # Another process holds the lock (LOCK_NB -> immediate failure).
                yield False
                return
            # The filesystem can't honor flock (e.g. ENOLCK on some NFS/FUSE
            # mounts). Degrade to a no-op lock rather than misreport contention.
            if logger is not None:
                logger.warning(f"Could not lock {lock_path}: {e}; proceeding without it")
            yield True
            return
        if logger is not None:
            logger.debug(f"Acquired run lock {lock_path}")
        try:
            yield True
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
