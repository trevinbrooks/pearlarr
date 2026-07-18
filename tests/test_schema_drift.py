# pyright: strict
"""Drift net: our boundary models vs the vendored Sonarr/Radarr OpenAPI captures.

Every `seadex_types` model that mirrors an arr resource
is pinned to its component in `schemas/sonarr.schema` / `schemas/radarr.schema`:
each declared field's wire name must exist in the component's `properties`, and
the `QualitySource` vocabulary must match the schema enum. A regen of the
captures (see `schemas/README.md`) that renames or drops a consumed property
fails here, naming the model.field and component.

Scope: declared fields only. The `_WireModel` write bodies get the same check.
Their `extra="allow"` passthrough of undeclared keys is out of scope by design
(passthrough carries whatever the wire had, so there is nothing to drift).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel
from pydantic.fields import FieldInfo

from pearlarr.json_narrow import is_json_list, is_json_obj
from pearlarr.seadex_types import (
    CommandResource,
    HistoryRecord,
    ImportRejection,
    Json,
    Language,
    ManualImportCandidate,
    ManualImportFile,
    MovieFile,
    ParsedFileInfo,
    Quality,
    QualityDefinition,
    QualityModel,
    QualitySource,
    QueueRecord,
    RadarrMovie,
    Revision,
    SonarrEpisode,
    SonarrEpisodeFile,
    SonarrSeries,
)

_SCHEMA_DIR: Final = Path(__file__).resolve().parent.parent / "schemas"


@dataclass(frozen=True)
class _Schema:
    """One vendored OpenAPI capture, reduced to its `components.schemas` map."""

    name: str
    components: dict[str, Json]

    def component(self, name: str) -> dict[str, Json]:
        node = self.components.get(name)
        assert is_json_obj(node), f"{self.name}.schema has no component {name}"
        return node

    def properties(self, component: str) -> dict[str, Json]:
        props = self.component(component).get("properties")
        assert is_json_obj(props), f"{self.name}:{component} has no properties"
        return props

    def _deref(self, node: Json) -> dict[str, Json]:
        """Resolve one `$ref` hop. A plain object node passes through."""

        assert is_json_obj(node), f"{self.name}: expected an object node, got {type(node).__name__}"
        ref = node.get("$ref")
        if isinstance(ref, str):
            return self.component(ref.removeprefix("#/components/schemas/"))
        return node

    def _alias_path_problem(self, component: str, path: AliasPath) -> str | None:
        """Walk an `AliasPath` through the component. The failing step, or None."""

        node = self.component(component)
        for step in path.path:
            if isinstance(step, int):
                # An int step indexes an array: descend into items, no property lookup.
                items = node.get("items")
                if not is_json_obj(items):
                    return f"step {step} indexes a non-array node"
                node = self._deref(items)
                continue
            props = node.get("properties")
            if not is_json_obj(props) or step not in props:
                return f"no property {step!r} on the walked node"
            node = self._deref(props[step])
        return None

    def field_problem(self, component: str, field_name: str, info: FieldInfo) -> str | None:
        """None when the field's wire name resolves in the component. Else the reason."""

        alias = info.validation_alias
        if alias is None or isinstance(alias, str):
            wire = alias or info.alias or field_name
            return None if wire in self.properties(component) else f"no property {wire!r}"
        if isinstance(alias, AliasChoices):
            # Each choice is a candidate wire name. >=1 must exist per schema.
            names = [choice for choice in alias.choices if isinstance(choice, str)]
            assert len(names) == len(alias.choices), f"AliasChoices with a non-str choice: {alias.choices!r}"
            if any(name in self.properties(component) for name in names):
                return None
            return f"none of the alias choices {names!r} are properties"
        return self._alias_path_problem(component, alias)


def _load_schema(name: str) -> _Schema:
    raw: object = json.loads((_SCHEMA_DIR / f"{name}.schema").read_text(encoding="utf-8"))
    assert is_json_obj(raw), f"{name}.schema is not a JSON object"
    components = raw.get("components")
    assert is_json_obj(components), f"{name}.schema has no components"
    schemas = components.get("schemas")
    assert is_json_obj(schemas), f"{name}.schema has no components.schemas"
    return _Schema(name=name, components=schemas)


_SCHEMAS: Final = {name: _load_schema(name) for name in ("sonarr", "radarr")}


@dataclass(frozen=True)
class Spec:
    """One boundary model pinned to its OpenAPI component."""

    model: type[BaseModel]
    component: str
    schemas: tuple[str, ...]
    # Declared fields with no schema property behind them (each justified below).
    exempt: frozenset[str] = frozenset()


_SONARR: Final = ("sonarr",)
_RADARR: Final = ("radarr",)
_BOTH: Final = ("sonarr", "radarr")

# Every seadex_types model that mirrors an arr resource. Deliberately absent:
# the AniList models and seadex_dict dataclasses (not arr wire shapes), and
# CommandBody/CommandFile - POST /command bodies are polymorphic (the
# ManualImportCommand payload) and the spec's Command component models only the
# generic envelope, so there is no schema contract to check them against.
ROSTER: Final = (
    Spec(SonarrSeries, "SeriesResource", _SONARR),
    Spec(SonarrEpisode, "EpisodeResource", _SONARR),
    Spec(SonarrEpisodeFile, "EpisodeFileResource", _SONARR),
    # `offline` is set only by the local SxxExx regex fallback (never from
    # the wire). It marks a parse the positional leg must treat as unknown.
    Spec(ParsedFileInfo, "ParseResource", _SONARR, exempt=frozenset({"offline"})),
    Spec(ManualImportCandidate, "ManualImportResource", _SONARR),
    Spec(ManualImportFile, "ManualImportReprocessResource", _SONARR),
    Spec(QualityDefinition, "QualityDefinitionResource", _SONARR),
    Spec(QueueRecord, "QueueResource", _SONARR),
    # `files` reads AliasPath(body -> files): a runtime echo of the polymorphic
    # command body, which the generic Command component doesn't model.
    Spec(CommandResource, "CommandResource", _SONARR, exempt=frozenset({"files"})),
    Spec(RadarrMovie, "MovieResource", _RADARR),
    Spec(MovieFile, "MovieFileResource", _RADARR),
    # `reason` is synthesized by a before-validator from the `data` map
    # (no such property exists). test_history_data_backs_the_reason_lift pins `data`.
    Spec(HistoryRecord, "HistoryResource", _BOTH, exempt=frozenset({"reason"})),
    Spec(Quality, "Quality", _BOTH),
    Spec(Revision, "Revision", _BOTH),
    Spec(QualityModel, "QualityModel", _BOTH),
    Spec(Language, "Language", _BOTH),
    Spec(ImportRejection, "ImportRejectionResource", _BOTH),
)


def _spec_id(spec: Spec) -> str:
    return spec.model.__name__


@pytest.mark.parametrize("spec", ROSTER, ids=_spec_id)
def test_declared_fields_exist_in_schema(spec: Spec) -> None:
    """Every declared field's wire name resolves in the mapped component(s)."""

    for schema_name in spec.schemas:
        schema = _SCHEMAS[schema_name]
        for field_name, info in spec.model.model_fields.items():
            if field_name in spec.exempt:
                continue
            problem = schema.field_problem(spec.component, field_name, info)
            assert problem is None, f"{spec.model.__name__}.{field_name} -> {schema.name}:{spec.component}: {problem}"


def test_history_data_backs_the_reason_lift() -> None:
    """The `data` map feeding HistoryRecord's synthesized `reason` stays in both schemas."""

    for schema in _SCHEMAS.values():
        assert "data" in schema.properties("HistoryResource"), f"{schema.name}:HistoryResource lost 'data'"


def test_quality_source_enum_matches_sonarr() -> None:
    """QualitySource is set-equal to sonarr.schema's QualitySource enum.

    Sonarr ONLY: Radarr's QualitySource is a disjoint vocabulary (cam, telesync,
    tv, ...) our enum never consumes - quality-source matching lives on the
    Sonarr manual-import path.
    """

    values = _SCHEMAS["sonarr"].component("QualitySource").get("enum")
    assert is_json_list(values)
    schema_values = {value for value in values if isinstance(value, str)}
    assert len(schema_values) == len(values), "non-string entries in the schema enum"
    assert {member.value for member in QualitySource} == schema_values
