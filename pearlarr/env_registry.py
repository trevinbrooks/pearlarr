"""Registry of every environment variable Pearlarr or its Docker entrypoint reads.

The one authored home for the inventory: the docs generator renders it into the
configuration reference, a parity test greps the tree against it, and `paths`
reads the data-dir variable name from it. New variables use the `PEARLARR_`
prefix with `__` as the nesting delimiter, so names stay unambiguous under a
future pydantic-settings split (`PEARLARR_SONARR__URL` -> `sonarr.url`).
"""

from dataclasses import dataclass
from typing import Literal

DATA_DIR_ENV = "PEARLARR_DATA_DIR"
"""The data-directory override variable (also read by the Docker entrypoint)."""


@dataclass(frozen=True, slots=True)
class EnvVar:
    """One environment variable and where it is honored."""

    name: str

    scope: Literal["app", "docker"]
    """`app` is read by the application itself; `docker` only by the container entrypoint."""

    description: str
    """What the variable controls, in the compiled docs dialect (plain text + single backticks)."""


ENV_VARS: tuple[EnvVar, ...] = (
    EnvVar(DATA_DIR_ENV, "app", "Override the data directory; the global `--data-dir` flag wins over it."),
    EnvVar("PEARLARR_CRON", "docker", "Cron schedule for the container's recurring runs."),
    EnvVar(
        "PEARLARR_RUN_ON_START",
        "docker",
        "Whether the container starts with a catch-up run, before the cron cadence takes over.",
    ),
)
