# pyright: strict
"""The filename quality parse and the layered quality/language selection."""

from pearlarr.import_quality import (
    ParsedQuality,
    derive_languages,
    parse_quality_from_filename,
    quality_axes_from_model,
    quality_axes_from_name,
    resolve_quality,
)
from pearlarr.seadex_types import Quality, QualityDefinition, QualityModel, QualitySource, Revision

# A realistic subset of Sonarr's /api/v3/qualitydefinition, matched on the
# structured (source, resolution) pair (never on the display name). Validated
# from the raw wire shape, as the client boundary does.
_RAW_DEFS: list[dict[str, object]] = [
    {"quality": {"id": 4, "name": "HDTV-720p", "source": "television", "resolution": 720}},
    {"quality": {"id": 6, "name": "Bluray-720p", "source": "bluray", "resolution": 720}},
    {"quality": {"id": 9, "name": "HDTV-1080p", "source": "television", "resolution": 1080}},
    {"quality": {"id": 3, "name": "WEBDL-1080p", "source": "web", "resolution": 1080}},
    {"quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
    {"quality": {"id": 20, "name": "Bluray-1080p Remux", "source": "blurayRaw", "resolution": 1080}},
    {"quality": {"id": 19, "name": "Bluray-2160p", "source": "bluray", "resolution": 2160}},
    {"quality": {"id": 21, "name": "Bluray-2160p Remux", "source": "blurayRaw", "resolution": 2160}},
]
_DEFS: list[QualityDefinition] = [QualityDefinition.model_validate(d) for d in _RAW_DEFS]


def _resolved_name(model: QualityModel) -> str | None:
    """The emitted quality's display name, for asserting which definition won."""

    return model.quality.name if model.quality is not None else None


class TestParseQualityFromFilename:
    """`parse_quality_from_filename` extracts the `(source, resolution)` pair, including the blurayRaw remux case."""

    def test_2160p_webdl(self) -> None:
        assert parse_quality_from_filename(
            "Show.S01E01.2160p.WEB-DL.x265.mkv",
        ) == ParsedQuality(source=QualitySource.WEB, resolution=2160)

    def test_1080p_bluray(self) -> None:
        assert parse_quality_from_filename(
            "[Group] Show - 01 [BluRay 1080p HEVC].mkv",
        ) == ParsedQuality(source=QualitySource.BLURAY, resolution=1080)

    def test_bd_remux_1080p_is_blurayraw(self) -> None:
        # The bug case: a BD remux must parse to (blurayRaw, 1080) - never the
        # bogus "Remux-1080p" name the old joiner produced.
        assert parse_quality_from_filename(
            "The.Seven.Deadly.Sins.1080p.Dual.Audio.BD.Remux.DTS-HD.MA-TTGA.mkv",
        ) == ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=1080)

    def test_no_resolution_still_keeps_source(self) -> None:
        assert parse_quality_from_filename(
            "Show.S01E01.WEB-DL.mkv",
        ) == ParsedQuality(source=QualitySource.WEB, resolution=None)

    def test_no_source_leaves_source_none(self) -> None:
        # No recognized source token -> source stays None (NOT defaulted to WEB),
        # so the configured default fills the axis.
        assert parse_quality_from_filename(
            "Show - 01 [1080p].mkv",
        ) == ParsedQuality(source=None, resolution=1080)


class TestQualityAxesFromModel:
    """`quality_axes_from_model` reads the `(source, resolution)` pair off a `QualityModel`.

    An unknown source or zero resolution is treated as undetermined, as is a None model.
    """

    def test_reads_structured_source_and_resolution(self) -> None:
        model = QualityModel.model_validate(
            {"quality": {"name": "Bluray-1080p", "source": "bluray", "resolution": 1080}},
        )
        assert quality_axes_from_model(model) == ParsedQuality(
            source=QualitySource.BLURAY,
            resolution=1080,
        )

    def test_unknown_source_and_zero_resolution_are_undetermined(self) -> None:
        model = QualityModel.model_validate(
            {"quality": {"name": "Unknown", "source": "unknown", "resolution": 0}},
        )
        assert quality_axes_from_model(model) == ParsedQuality(source=None, resolution=None)

    def test_none_model_is_empty(self) -> None:
        assert quality_axes_from_model(None) == ParsedQuality()


class TestQualityAxesFromName:
    """`quality_axes_from_name` resolves a display name to its `(source, resolution)` pair via the definitions."""

    def test_resolves_default_name_to_axes(self) -> None:
        assert quality_axes_from_name("Bluray-2160p", _DEFS) == ParsedQuality(
            source=QualitySource.BLURAY,
            resolution=2160,
        )

    def test_unset_or_unmatched_is_empty(self) -> None:
        assert quality_axes_from_name(None, _DEFS) == ParsedQuality()
        assert quality_axes_from_name("Not-A-Quality", _DEFS) == ParsedQuality()


class TestResolveQuality:
    """`resolve_quality` picks the winning value per axis - Sonarr's own parse, then ours, then the default.

    It resolves the pair to a definition, falling back to the candidate verbatim or an explicit Unknown
    when nothing resolves.
    """

    def test_sonarr_wins_over_ours_and_default(self) -> None:
        # Sonarr parsed (web, 1080). Our filename parse and the default disagree.
        sonarr = ParsedQuality(source=QualitySource.WEB, resolution=1080)
        ours = ParsedQuality(source=QualitySource.BLURAY, resolution=2160)
        default = ParsedQuality(source=QualitySource.TELEVISION, resolution=720)
        model = resolve_quality(sonarr, ours, default, _DEFS, candidate_model=None)
        assert _resolved_name(model) == "WEBDL-1080p"

    def test_bd_remux_resolves_to_remux_definition(self) -> None:
        # The headline fix: (blurayRaw, 1080) -> "Bluray-1080p Remux" (valid id+name).
        sonarr = ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=1080)
        model = resolve_quality(
            sonarr,
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate_model=None,
        )
        assert _resolved_name(model) == "Bluray-1080p Remux"
        assert model.quality is not None
        assert model.quality.id == 20

    def test_per_axis_fill_from_default(self) -> None:
        # User's example: we parsed (None, 1080). Default is Bluray-2160p ->
        # import as Bluray-1080p, NOT WEBDL-1080p and NOT Unknown.
        ours = ParsedQuality(source=None, resolution=1080)
        default = ParsedQuality(source=QualitySource.BLURAY, resolution=2160)
        model = resolve_quality(
            ParsedQuality(),
            ours,
            default,
            _DEFS,
            candidate_model=None,
        )
        assert _resolved_name(model) == "Bluray-1080p"

    def test_blurayraw_downgrades_to_bluray_when_no_remux_def(self) -> None:
        # 720p has no remux definition. A (blurayRaw, 720) gracefully downgrades
        # to Bluray-720p rather than failing.
        sonarr = ParsedQuality(source=QualitySource.BLURAY_RAW, resolution=720)
        model = resolve_quality(
            sonarr,
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate_model=None,
        )
        assert _resolved_name(model) == "Bluray-720p"

    def test_no_match_falls_back_to_candidate_verbatim(self) -> None:
        # Nothing determined, but Sonarr's candidate carries a real quality:
        # re-emit it verbatim rather than omit the quality key.
        candidate = QualityModel.model_validate(
            {
                "quality": {"id": 7, "name": "Bluray-1080p", "source": "bluray", "resolution": 1080},
                "revision": {"version": 1, "real": 0, "isRepack": False},
            },
        )
        model = resolve_quality(
            ParsedQuality(),
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate,
        )
        assert model is candidate

    def test_nothing_resolves_synthesizes_explicit_unknown(self) -> None:
        # No axes and no candidate quality: emit an explicit Unknown object (never
        # omit the key - the omitted key is what crashed Sonarr's FileNameBuilder).
        model = resolve_quality(
            ParsedQuality(),
            ParsedQuality(),
            ParsedQuality(),
            _DEFS,
            candidate_model=None,
        )
        assert model.quality == Quality(id=0, name="Unknown", source="unknown", resolution=0)
        assert model.revision is not None

    def test_candidate_revision_is_preserved(self) -> None:
        # A repack/proper revision on the candidate carries onto the resolved model.
        sonarr = ParsedQuality(source=QualitySource.WEB, resolution=1080)
        candidate = QualityModel.model_validate(
            {
                "quality": {"name": "WEBDL-1080p", "source": "web", "resolution": 1080},
                "revision": {"version": 2, "real": 0, "isRepack": True},
            },
        )
        model = resolve_quality(sonarr, ParsedQuality(), ParsedQuality(), _DEFS, candidate)
        assert model.revision == Revision(version=2, real=0, isRepack=True)


class TestDeriveLanguages:
    """`derive_languages` returns both audio languages for a dual-audio release, or just the primary otherwise."""

    def test_dual_audio_returns_dual(self) -> None:
        assert derive_languages(True, ["Japanese", "English"], ["Japanese"]) == [
            "Japanese",
            "English",
        ]

    def test_single_audio_returns_single(self) -> None:
        assert derive_languages(False, ["Japanese", "English"], ["Japanese"]) == ["Japanese"]
