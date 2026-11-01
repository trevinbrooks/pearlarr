# pyright: strict
"""The container entrypoint script, run under `/bin/sh` with stub `pearlarr` and `supercronic` commands."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parent.parent / "docker" / "entrypoint.sh"

# Holds `run single` open until the test creates `release`, so the SIGTERM lands mid-run.
_PEARLARR_STUB = """#!/bin/sh
touch "$STUB/started"
while [ ! -f "$STUB/release" ]; do sleep 0.05; done
touch "$STUB/finished"
"""

_SUPERCRONIC_STUB = """#!/bin/sh
touch "$STUB/supercronic"
"""


def _write_stub(bin_dir: Path, name: str, body: str) -> None:
    stub = bin_dir / name
    stub.write_text(body, encoding="utf-8")
    stub.chmod(0o755)


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 30.0
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.02)


@pytest.mark.skipif(os.name != "posix", reason="the entrypoint is a POSIX shell script")
def test_a_stop_during_run_on_start_lets_it_finish_and_skips_the_schedule(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub(bin_dir, "pearlarr", _PEARLARR_STUB)
    _write_stub(bin_dir, "supercronic", _SUPERCRONIC_STUB)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yml").write_text("", encoding="utf-8")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "STUB": str(tmp_path), "PEARLARR_DATA_DIR": str(data_dir)}

    with subprocess.Popen(["/bin/sh", str(ENTRYPOINT)], env=env, stdout=subprocess.PIPE, encoding="utf-8") as shell:
        try:
            _wait_for(tmp_path / "started")
            shell.send_signal(signal.SIGTERM)
            (tmp_path / "release").touch()
            stdout, _ = shell.communicate(timeout=30.0)
        finally:
            shell.kill()

    assert shell.returncode == 0
    assert (tmp_path / "finished").exists()
    assert not (tmp_path / "supercronic").exists()
    assert "Stopped during run-on-start" in stdout
