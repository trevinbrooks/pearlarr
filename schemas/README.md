# Vendored arr OpenAPI captures

`sonarr.schema` and `radarr.schema` are unmodified captures of the official
Sonarr and Radarr v3 OpenAPI specs (JSON documents, despite the extension —
code comments in `pearlarr/modules/seadex_types.py` cite them by these names).

They are vendored so that:

- `tests/test_schema_drift.py` can check every boundary model's wire names
  against the captured contract offline, on every test run;
- reviewers can grep the wire contract the boundary models consume without
  chasing the upstream repositories.

## Regenerating

```
curl -o schemas/sonarr.schema https://raw.githubusercontent.com/Sonarr/Sonarr/develop/src/Sonarr.Api.V3/openapi.json
curl -o schemas/radarr.schema https://raw.githubusercontent.com/Radarr/Radarr/develop/src/Radarr.Api.V3/openapi.json
```

A regen is a deliberate change with its own review, not routine upkeep: after
refreshing, run `uv run pytest tests/test_schema_drift.py` and re-check the
nullability comments in `pearlarr/modules/seadex_types.py` — the models mirror
the captured schemas' `| null` markers exactly, so a nullability change upstream
must be reflected there, not just absorbed.

Sonarr and Radarr publish these specs as part of their GPL-3.0-licensed source.
