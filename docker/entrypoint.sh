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

# docker stop signals PID 1 only, and a PID 1 shell drops SIGTERM unless it traps it.
# A stop is passed on to the catch-up run as SIGTERM, which stops it cleanly, and the
# schedule never starts after one.
stopping=false
signals=0
run_pid=
on_stop() {
    stopping=true
    signals=$((signals + 1))
    if [ -n "${run_pid}" ]; then
        kill -TERM "${run_pid}" 2>/dev/null || true
    fi
}
trap on_stop TERM INT

# Catch-up pass on boot. A failure must not flap the container.
if [ "${PEARLARR_RUN_ON_START:-true}" = "true" ]; then
    # In the background so the trap runs during the run. A stop that came before
    # run_pid was set is passed on here.
    pearlarr run single &
    run_pid=$!
    if [ "${stopping}" = "true" ]; then
        kill -TERM "${run_pid}" 2>/dev/null || true
    fi
    while :; do
        seen=${signals}
        code=0
        wait "${run_pid}" || code=$?
        # A trapped signal cuts wait short with a status above 128, so wait again for the run's own.
        if [ "${code}" -le 128 ] || [ "${signals}" -eq "${seen}" ]; then
            break
        fi
    done
    run_pid=
    echo "run-on-start exited with ${code}"
fi

# Cron owns the container cadence (config's schedule.interval_hours only drives
# the bare-metal `run scheduled`). Passthrough because pearlarr already timestamps.
crontab="${TMPDIR:-/tmp}/crontab"
echo "${PEARLARR_CRON:-0 */6 * * *} pearlarr run single" > "${crontab}"

# Checked right before exec, so a stop at any earlier point keeps the schedule from starting.
if [ "${stopping}" = "true" ]; then
    echo "Stopped before the schedule started - exiting"
    exit 0
fi
exec supercronic -passthrough-logs "${crontab}"
