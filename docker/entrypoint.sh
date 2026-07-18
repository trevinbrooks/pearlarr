#!/bin/sh
# Container entrypoint: arg passthrough, first-boot bootstrap, optional
# run-on-start, then supercronic as PID 1.
set -eu

# Any args = a one-off CLI invocation (e.g. `docker compose run --rm pearlarr run single --sonarr`).
if [ "$#" -gt 0 ]; then
    exec pearlarr "$@"
fi

# A root-owned bind mount is the classic first-boot trap. Fail with the fix, not a traceback.
if [ ! -w "${PEARLARR_DATA_DIR}" ]; then
    echo "ERROR: ${PEARLARR_DATA_DIR} is not writable by uid $(id -u) (gid $(id -g))." >&2
    echo "Chown the mounted host directory to that uid, or point the compose user:/PUID/PGID at the directory's owner." >&2
    exit 1
fi

# First boot: write the starter template, then stop so you can fill it in.
if [ ! -f "${PEARLARR_DATA_DIR}/config.yml" ]; then
    pearlarr config init
    echo "Fill in config.yml (on the host side of the ${PEARLARR_DATA_DIR} mount), then start the container again." >&2
    exit 1
fi

# Catch-up pass on boot. A failure must not flap the container.
if [ "${PEARLARR_RUN_ON_START:-true}" = "true" ]; then
    code=0
    pearlarr run single || code=$?
    echo "run-on-start exited with ${code}"
fi

# Cron owns the container cadence (config's schedule.interval_hours only drives
# the bare-metal `run scheduled`). Passthrough because pearlarr already timestamps.
echo "${PEARLARR_CRON:-0 */6 * * *} pearlarr run single" > /tmp/crontab
exec supercronic -passthrough-logs /tmp/crontab
