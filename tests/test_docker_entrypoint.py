# pyright: strict
"""The container entrypoint script, run under `/bin/sh` with stub `pearlarr` and `supercronic` commands."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parent.parent / "docker" / "entrypoint.sh"

# Runs about 10s unless a SIGTERM stops it first, which it records.
_STOPPABLE_RUN_STUB = """#!/bin/sh
trap 'touch "$STUB/got_term"; exit 0' TERM
touch "$STUB/started"
i=0
while [ "$i" -lt 200 ]; do sleep 0.05; i=$((i + 1)); done
exit 3
"""

_FAILING_RUN_STUB = """#!/bin/sh
exit 2
"""

_SUPERCRONIC_STUB = """#!/bin/sh
printf '%s\\n' "$@" > "$STUB/supercronic"
"""

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the entrypoint is a POSIX shell script")


def _write_stub(bin_dir: Path, name: str, body: str) -> None:
    stub = bin_dir / name
    stub.write_text(body, encoding="utf-8")
    stub.chmod(0o755)


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 30.0
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.02)


def _start(tmp_path: Path, run_stub: str) -> subprocess.Popen[str]:
    """Start the entrypoint with `run_stub` as `pearlarr`, a recording `supercronic`, and a ready data dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub(bin_dir, "pearlarr", run_stub)
    _write_stub(bin_dir, "supercronic", _SUPERCRONIC_STUB)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yml").write_text("", encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "STUB": str(tmp_path),
        "PEARLARR_DATA_DIR": str(data_dir),
        "TMPDIR": str(tmp_path),
    }
    return subprocess.Popen(["/bin/sh", str(ENTRYPOINT)], env=env, stdout=subprocess.PIPE, encoding="utf-8")


def test_a_stop_during_run_on_start_is_passed_on_and_skips_the_schedule(tmp_path: Path) -> None:
    with _start(tmp_path, _STOPPABLE_RUN_STUB) as shell:
        try:
            _wait_for(tmp_path / "started")
            shell.send_signal(signal.SIGTERM)
            stdout, _ = shell.communicate(timeout=30.0)
        finally:
            shell.kill()

    assert shell.returncode == 0
    assert (tmp_path / "got_term").exists()
    assert "run-on-start exited with 0" in stdout
    assert "Stopped before the schedule started" in stdout
    assert not (tmp_path / "supercronic").exists()


def test_without_a_stop_the_schedule_starts_even_after_a_failed_run(tmp_path: Path) -> None:
    with _start(tmp_path, _FAILING_RUN_STUB) as shell:
        try:
            stdout, _ = shell.communicate(timeout=30.0)
        finally:
            shell.kill()

    assert shell.returncode == 0
    assert "run-on-start exited with 2" in stdout
    crontab = tmp_path / "crontab"
    assert (tmp_path / "supercronic").read_text(encoding="utf-8").splitlines() == ["-passthrough-logs", str(crontab)]
    assert crontab.read_text(encoding="utf-8") == "0 */6 * * * pearlarr run single\n"
