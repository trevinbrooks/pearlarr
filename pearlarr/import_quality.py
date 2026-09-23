"""Pure layered quality and language resolution for the manual-import payload (the quality key is never omitted)."""

import re
from dataclasses import dataclass

from .seadex_types import Language, Quality, QualityDefinition, QualityModel, QualitySource, Revision

# Filename source tokens -> QualitySource, ordered most-specific first so a
# "BluRay Remux" name resolves to BLURAY_RAW (not BLURAY), "BD" counts as BluRay,
# and "WEB-DL" wins over a bare "WEB". A token that matches nothing leaves the
# source axis undetermined (None) - it is NEVER defaulted to WEB here. The
# configured default fills it.
_SOURCE_PATTERNS: list[tuple[re.Pattern[str], QualitySource]] = [
    (re.compile(r"remux", re.IGNORECASE), QualitySource.BLURAY_RAW),
    (re.compile(r"blu-?ray|\bbd\b", re.IGNORECASE), QualitySource.BLURAY),
    (re.compile(r"web-?dl", re.IGNORECASE), QualitySource.WEB),
    (re.compile(r"webrip", re.IGNORECASE), QualitySource.WEBRIP),
    (re.compile(r"hdtv", re.IGNORECASE), QualitySource.TELEVISION),
    (re.compile(r"\bdvd\b", re.IGNORECASE), QualitySource.DVD),
    (re.compile(r"\bweb\b", re.IGNORECASE), QualitySource.WEB),
]

_RESOLUTION_PATTERN: re.Pattern[str] = re.compile(r"(2160|1080|720|480)p", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedQuality:
    """Quality as two independent axes: `source` and `resolution`.

    Either axis is `None` when it could not be authoritatively determined, which
    is what lets the quality decision layer the axes across Sonarr's parse, our
    filename parse, and the configured default (each fills only the axes the
    higher-precedence layers left `None`). The resulting `(source, resolution)`
    pair is matched against Sonarr's quality definitions to pick the real quality.
    """

    source: QualitySource | None = None
    resolution: int | None = None


def parse_quality_from_filename(filename: str) -> ParsedQuality:
    """Best-effort `(source, resolution)` parse of a SeaDex filename (or path: only the text is matched).

    Detects a resolution (`2160`/`1080`/`720`/`480`) and a source
    (Remux, BluRay, WEB-DL, WEBRip, WEB, HDTV, DVD), case-insensitively and
    independently. Either axis is `None` when not found - notably an
    unrecognized source is left `None` (NOT defaulted to WEB), so the configured
    default can fill it rather than the file being silently mislabeled.
    """

    res_match = _RESOLUTION_PATTERN.search(filename)
    resolution = int(res_match.group(1)) if res_match is not None else None

    source: QualitySource | None = None
    for pattern, candidate in _SOURCE_PATTERNS:
        if pattern.search(filename):
            source = candidate
            break
    return ParsedQuality(source=source, resolution=resolution)


def quality_axes_from_model(model: QualityModel | None) -> ParsedQuality:
    """The `(source, resolution)` axes of a Sonarr `QualityModel`.

    Reads the canonical schema path `model.quality.source` /
    `model.quality.resolution` (every field defaults to None on the partial
    models the helpers build). An `"unknown"` source or a `0`/absent
    resolution maps to `None` (undetermined), so an unparsed candidate
    cleanly yields `ParsedQuality()` and falls through to the next
    precedence layer.
    """

    if model is None:
        return ParsedQuality()
    quality = model.quality  # an empty/null wire quality already folded to None
    if quality is None:
        return ParsedQuality()
    resolution = quality.resolution
    if resolution is None or resolution <= 0:
        resolution = None
    return ParsedQuality(source=QualitySource.parse(quality.source), resolution=resolution)


def quality_axes_from_name(
    name: str | None,
    quality_defs: list[QualityDefinition],
) -> ParsedQuality:
    """The `(source, resolution)` axes of a configured default quality NAME.

    Resolves the configured `imports.default_quality` (a Sonarr quality name like
    `"Bluray-2160p"`) to its structured axes by matching it, case-insensitively,
    against the `/api/v3/qualitydefinition` list - so the default contributes a
    real `(source, resolution)` the decision fills gaps from. An unset name, or
    one that matches no definition, yields `ParsedQuality()` (no default).
    """

    if not name:
        return ParsedQuality()
    target = name.casefold()
    for definition in quality_defs:
        quality = definition.quality
        if quality is None:
            continue
        def_name = quality.name
        if def_name is not None and def_name.casefold() == target:
            return quality_axes_from_model(QualityModel(quality=quality))
    return ParsedQuality()


def _find_definition(
    source: QualitySource,
    resolution: int,
    quality_defs: list[QualityDefinition],
) -> Quality | None:
    """The nested `Quality` whose `(source, resolution)` matches, or None.

    Scans the `/api/v3/qualitydefinition` list for the definition whose nested
    quality has the given structured source and resolution. `(source, resolution)`
    is unique across Sonarr's standard definitions (the only near-collision, Raw-HD
    vs HDTV-1080p, differs by source), so the pair identifies the quality without
    ever matching on its display name.
    """

    for definition in quality_defs:
        quality = definition.quality
        if quality is None:
            continue
        if quality.resolution == resolution and QualitySource.parse(quality.source) is source:
            return quality
    return None


# A RAW source degrades to its base when no remux/raw definition exists at that
# resolution (try the raw definition first, then this base).
_RAW_DOWNGRADE: dict[QualitySource, QualitySource] = {
    QualitySource.BLURAY_RAW: QualitySource.BLURAY,
    QualitySource.TELEVISION_RAW: QualitySource.TELEVISION,
}


def _candidate_revision(candidate_model: QualityModel | None) -> Revision:
    """The candidate's revision (proper/repack), or a fresh `version 1` default."""

    if candidate_model is not None and candidate_model.revision is not None:
        return candidate_model.revision
    return Revision(version=1, real=0, isRepack=False)


def resolve_quality(
    sonarr: ParsedQuality,
    ours: ParsedQuality,
    default: ParsedQuality,
    quality_defs: list[QualityDefinition],
    candidate_model: QualityModel | None,
) -> QualityModel:
    """Resolve the final manual-import `QualityModel` - never omitted.

    The source and resolution axes are decided independently, each taking the
    first authoritative value in precedence order: Sonarr's parse, then our
    filename parse, then the configured default. When both axes are determined the
    quality definition matching the `(source, resolution)` pair is emitted, so
    the payload always carries a quality Sonarr actually defines (a valid id+name).
    A determined `BLURAY_RAW`/`TELEVISION_RAW` with no matching remux/raw
    definition at that resolution gracefully downgrades to `BLURAY`/`TELEVISION`
    rather than failing.

    Crucially this never returns `None` and the caller never omits the quality:
    omitting it is exactly what made Sonarr crash in
    `FileNameBuilder.AddQualityTokens`. When nothing resolves, Sonarr's own
    candidate model (valid by construction) is re-emitted verbatim. Only if the
    candidate carries no quality at all is an explicit `Unknown` synthesized.
    """

    # Invariant: the import payload always carries a quality key - omitting it
    # crashes Sonarr in FileNameBuilder.AddQualityTokens (observed on Sonarr 4.x).
    source = sonarr.source or ours.source or default.source
    resolution = sonarr.resolution or ours.resolution or default.resolution
    revision = _candidate_revision(candidate_model)

    if source is not None and resolution is not None:
        quality = _find_definition(source, resolution, quality_defs)
        base = _RAW_DOWNGRADE.get(source)
        if quality is None and base is not None:
            quality = _find_definition(base, resolution, quality_defs)
        if quality is not None:
            return QualityModel(quality=quality, revision=revision)

    # No confident match: re-emit Sonarr's own candidate (valid by construction)
    # rather than omit the quality, else synthesize an explicit Unknown. An
    # EMPTY candidate quality already folded to None at the parse boundary, so
    # this None test guards the already-folded empty quality.
    if candidate_model is not None and candidate_model.quality is not None:
        return candidate_model
    unknown = Quality(id=0, name="Unknown", source="unknown", resolution=0)
    return QualityModel(quality=unknown, revision=revision)


def derive_languages(
    is_dual_audio: bool,
    dual: list[str],
    single: list[str],
) -> list[str]:
    """Pick the import language list: `dual` when dual-audio, else `single`."""

    return dual if is_dual_audio else single


def resolve_language_objects(
    names: list[str],
    lang_defs: list[Language],
) -> list[Language]:
    """Resolve configured language names to Sonarr `{id, name}` objects.

    Case-insensitive match against the `/api/v3/language` list, in request
    order. A name with no match is dropped rather than failing the import.
    """

    by_name: dict[str, Language] = {
        name.casefold(): definition for definition in lang_defs if (name := definition.name) is not None
    }
    resolved: list[Language] = []
    for name in names:
        definition = by_name.get(name.casefold())
        if definition is not None:
            # Re-built fresh with BOTH fields set, so the exclude_unset write
            # dump always carries them (a null id included).
            resolved.append(Language(id=definition.id, name=definition.name))
    return resolved
