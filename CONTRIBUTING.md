# Contributing

## Setup

SeaDexArr is developed with [uv](https://docs.astral.sh/uv/). Python 3.13 is the
floor; development and the Docker image run 3.14, and CI
(`.github/workflows/build.yaml`) tests both.

```
uv sync --group dev
```

This creates `.venv` with the runtime dependencies plus every tool below, pinned
by `uv.lock`. The lockfile is tracked: commit lockfile churn together with the
change that caused it.

## Quality gate

Every change must pass the full chain, in order, with zero findings:

```
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pyright
uv run pyrefly check
uv run pytest -q
```

CI (`.github/workflows/build.yaml`) runs the same gates on Python 3.13 and 3.14,
plus a total-coverage floor and a changed-line coverage gate (diff-cover, 90%).
Dependabot watches the GitHub Actions and pip ecosystems weekly.

### Suppressions

Production code carries zero suppressions: no `# type: ignore`, no `# noqa`, no
inline pyright directives — a PR adding one will be asked to remove it and fix
the underlying type instead. Tests are held to the same bar: every test module
opens with a `# pyright: strict` header. The one sanctioned escape hatch is a
module-level `# pyright: reportPrivateUsage=false` in a white-box test that
legitimately reads a collaborator's private members, and it must carry a
justification comment.

## Tests

- Tests are strict-typed and use recording fakes (`tests/fakes.py`) rather than
  `MagicMock`: a fake records the calls it received and returns typed values, so
  a drifted signature is a type error, not a silently green test.
- Shared object builders live in `tests/builders.py`; prefer extending an
  existing fake or builder over an ad-hoc double.
- Run one file with `uv run pytest tests/test_planner.py -q`.

## Running locally

```
uv run seadexarr config init          # write the starter config.yml
uv run seadexarr paths                # show where config/caches/logs resolve
uv run seadexarr run single --sonarr  # one Sonarr pass
```

Everything lives in one data directory (`seadexarr paths` prints it). Fill in
`config.yml` with your instance URLs and API keys — and never paste real keys or
webhook URLs into issues, docs, or code; placeholders only.

## Environment variables

Environment variables use the `SEADEXARR_` prefix with `__` as the nesting
delimiter: a future pydantic-settings layer would read `SEADEXARR_SONARR__URL`
into `config.sonarr.url`, so names must stay unambiguous under that split. The
current inventory:

- `SEADEXARR_DATA_DIR` — override the data directory (the global `--data-dir`
  flag wins over it)
- `SEADEXARR_CRON`, `SEADEXARR_RUN_ON_START` — Docker-entrypoint-only: the
  container's cron cadence and the boot-time catch-up run

New variables must follow the scheme.
