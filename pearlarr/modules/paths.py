"""The data-directory resolver: config, cache, and logs live under one platformdirs root."""

from __future__ import annotations

import os
from dataclasses import dataclass

from platformdirs import user_data_dir

from .env_registry import DATA_DIR_ENV

# The env var stays the canonical override (Docker sets it to /config); the
# CLI's --data-dir flag folds into it (see cli.main).
APP_NAME = "pearlarr"

# The single in-code source for the project's repository URL (the CLI epilog and
# Discord embeds read it). pyproject.toml's [project.urls] can't import it, so
# change the two together. NOTE for any future APP_NAME rename: APP_NAME is also
# the platformdirs directory, so a rename must ship a data-dir migration.
PROJECT_URL = "https://github.com/trevinbrooks/pearlarr"


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Every file the app reads/writes, all under one data directory.

    Unified, *arr-style layout: config, caches and logs share one dir so a single
    volume mount (or backup) covers the lot.
    """

    data_dir: str
    config: str
    cache: str
    cache_backup: str
    mappings_db: str
    log_dir: str


def resolve_paths(data_dir: str | None = None) -> AppPaths:
    r"""Resolve every path under the data directory.

    Precedence: explicit `data_dir` arg > `PEARLARR_DATA_DIR` env >
    `platformdirs.user_data_dir` (`~/Library/Application Support/pearlarr` on
    macOS, `~/.local/share/pearlarr` on Linux, `%LOCALAPPDATA%\pearlarr` on
    Windows). `appauthor=False` drops the Windows author subfolder.
    """

    base = data_dir or os.getenv(DATA_DIR_ENV) or user_data_dir(APP_NAME, appauthor=False)
    base = os.path.abspath(base)
    return AppPaths(
        data_dir=base,
        config=os.path.join(base, "config.yml"),
        cache=os.path.join(base, "cache.db"),
        cache_backup=os.path.join(base, "cache.backup.db"),
        mappings_db=os.path.join(base, "mappings.db"),
        log_dir=os.path.join(base, "logs"),
    )


def ensure_data_dir(paths: AppPaths) -> None:
    """Create the data directory if missing (config-template copy + lock need it)."""

    os.makedirs(paths.data_dir, exist_ok=True)
