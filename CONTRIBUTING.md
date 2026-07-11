# Contributing

## Setup

Pearlarr is developed with [uv](https://docs.astral.sh/uv/). Python 3.13 is the
floor; development and the Docker image run 3.14, and CI
(`.github/workflows/build.yaml`) tests both.

```console
uv sync --group dev
uv run pre-commit install
```

This creates `.venv` with the runtime dependencies plus every tool below, pinned
by `uv.lock`. The lockfile is tracked: commit lockfile churn together with the
change that caused it.

The acceptance test for this document is the ten-minute rule: a fresh clone
following it verbatim reaches a green quality gate within ten minutes.

## Quality gate

Every change must pass the full chain, in order, with zero findings:

```console
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pyright
uv run pyrefly check
uv run pytest -q
```

CI (`.github/workflows/build.yaml`) runs the same gates on Python 3.13 and 3.14,
plus a total-coverage floor and a changed-line coverage gate (diff-cover, 90%).
Dependabot watches the GitHub Actions and pip ecosystems weekly. Run one test
file with `uv run pytest tests/test_planner.py -q`.

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

### Verifying against live services

The recording fakes are the supported development loop. When you need a live
check, point a config at real Sonarr/Radarr instances and leave the
`qbittorrent` credentials blank: that is preview mode — every decision is made
and reported, nothing is grabbed, nothing is cached — and it is the safe live
harness. Never paste real API keys, hostnames, or webhook URLs into issues,
docs, code, or screenshots; placeholders only.

## Running locally

```console
uv run pearlarr config init          # write the starter config.yml
uv run pearlarr paths                # show where config/caches/logs resolve
uv run pearlarr run single --sonarr  # one Sonarr pass
```

Everything lives in one data directory (`pearlarr paths` prints it).

## Documentation

One authored home per fact: enumerable facts (config keys, defaults, allowed
values, env vars) are authored once, in code, and generated into every other
surface. Never edit a generated file or island — edit the source and run:

```console
uv run python scripts/gen_docs.py
```

Generated artifacts (each carries a `GENERATED` banner; CI and pre-commit fail
on drift):

| Artifact | Source |
| --- | --- |
| `pearlarr/modules/config_sample.yml` | attribute docstrings in `pearlarr/modules/config.py` |
| `schemas/config.schema.json` | the same config models |
| `docs/configuration.md` (islands between `gen:` markers) | config models + `pearlarr/modules/env_registry.py` |
| `CONTRIBUTING.md` (the env-var island below) | `pearlarr/modules/env_registry.py` |

### Drift map

Change X, also do Y — most rows are enforced mechanically, listed anyway so you
can predict CI:

| You changed | Also do |
| --- | --- |
| A config field, its docstring, default, or constraint | `uv run python scripts/gen_docs.py` |
| An enum a config field references | Docstring every member, then regenerate |
| An environment variable | Register it in `pearlarr/modules/env_registry.py`, then regenerate |
| A config key or cache schema (breaking) | An "Upgrade notes" entry in `CHANGELOG.md` |
| A user-visible behavior | A `CHANGELOG.md` entry |
| A user-facing message | Update any docs that quote it |
| Install or quickstart steps in the README | Re-run them verbatim, end to end |

### Writing rules

- Docstrings are Google style. Types never appear in docstrings (the checkers
  own types); no Sphinx roles; identifiers in single backticks.
- Modules, packages, and classes always carry a docstring. Functions and
  methods are documented unless the full contract is legible from name +
  signature; when in doubt, one line.
- Config-model attribute docstrings are a compiled dialect — plain text plus
  single backticks only — because they render into YAML comments, markdown
  tables, and JSON Schema descriptions. State meaning, interactions, and
  blank/`None` semantics; never defaults, types, or allowed-value lists (the
  generator injects those).
- Comments state a constraint the code cannot show, in 1–2 lines. Change
  provenance (PR numbers, phase tags, dates) never appears in code — that is
  git's job. A workaround for an external defect names the defect and the
  version observed against, so it can be retested and removed.
- `# Invariant:` is a reserved comment prefix for load-bearing invariants at
  their enforcement sites.
- WARNING/ERROR messages read `what - cause - next action`, name the offending
  value and the exact config key/flag/command, sentence case, ASCII `" - "`
  connector, no markdown in terminal strings.
- Docs ride the same PR as the change they describe, never a follow-up.

## Task playbooks

### Add a config key, end to end

1. Add the field to its group model in `pearlarr/modules/config.py`, with an
   attribute docstring in the compiled dialect. Constrain it in the type
   (`Literal`, an enum, `ge=`) so a bad value fails at load; new enum members
   get their own docstrings.
2. `uv run python scripts/gen_docs.py` — the sample, schema, and reference
   tables update themselves.
3. Read the key where the behavior lives; add tests.
4. If the key interacts with others, extend the group's prose island in
   `docs/configuration.md` (outside the `gen:` markers).
5. Add a `CHANGELOG.md` entry (plus "Upgrade notes" if breaking).

### Add a tracker

1. Add the display name to the right `*_TRACKER_NAMES` tuple in
   `pearlarr/modules/config.py` (a parity test pins the set against the seadex
   library's `Tracker` enum).
2. If its releases carry a parseable download page, add a parser to the table
   in `pearlarr/modules/torrents.py` (`PARSEABLE_TRACKERS` derives from it) —
   with tests; trackers without a parser are skipped with a warning.
3. `uv run python scripts/gen_docs.py`.

## Commits

`type(scope): imperative summary` at ≤72 characters (`feat(planner): …`,
`fix(cache): …`); the body says why. Summaries must be meaningful without
private context — no session or chat references.

## Release checklist

1. Roll `Unreleased` in `CHANGELOG.md` into the new version heading with the
   date; verify "Upgrade notes" covers every config/cache change.
2. Bump `version` in `pyproject.toml`; run the full gate.
3. Tag `vX.Y.Z` and push the tag; verify the PyPI package and `ghcr.io` image
   land.
4. Smoke the Docker quick start from the README on a clean host, verbatim.

## Environment variables

<!-- gen:env-vars - GENERATED by scripts/gen_docs.py from pearlarr/modules/env_registry.py; do not edit between the markers; regenerate: uv run python scripts/gen_docs.py -->
| Variable | Read by | Meaning |
| --- | --- | --- |
| `PEARLARR_DATA_DIR` | application | Override the data directory; the global `--data-dir` flag wins over it. |
| `PEARLARR_CRON` | Docker entrypoint | Cron schedule for the container's recurring runs. |
| `PEARLARR_RUN_ON_START` | Docker entrypoint | Whether the container starts with a catch-up run, before the cron cadence takes over. |
<!-- /gen:env-vars -->

Names use the `PEARLARR_` prefix with `__` as the nesting delimiter (a future
settings layer would read `PEARLARR_SONARR__URL` into `sonarr.url`), so new
names must stay unambiguous under that split. Register new variables in
`pearlarr/modules/env_registry.py`.

## License

Contributions are accepted under the repository license (GPL-3.0-or-later),
which covers the documentation as well as the code.
