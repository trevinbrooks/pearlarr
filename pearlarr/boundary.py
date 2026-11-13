"""How foreign JSON is narrowed: the `Json` shape, the base models, the lenient field types, and validation."""

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Annotated, cast

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

# The recursive JSON value shape.
type Json = bool | int | float | str | Sequence["Json"] | Mapping[str, "Json"] | None


def coerce_int(value: object) -> int | None:
    """Best-effort int, or None for a non-numeric / NaN value."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


# --- pydantic boundary plumbing ----------------------------------------------


class ApiModel(BaseModel):
    """Frozen boundary read model: unknown keys ignored, field-name kwargs allowed.

    `validate_by_name` is required, or constructing an aliased field by field name no-ops to the default.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", validate_by_name=True)


class WireModel(BaseModel):
    """Frozen wire re-emit shape: unknown keys VALIDATE and RE-EMIT (extra="allow").

    Write dumps are `model_dump(exclude_unset=True)`, NEVER `exclude_none`: an explicitly-set None must reach
    the wire, so set every key the body needs.
    """

    model_config = ConfigDict(frozen=True, extra="allow")


class BoundaryContractError(RuntimeError):
    """A strict library read got a non-empty payload with zero valid records."""


def validation_summary(e: ValidationError) -> str:
    """A log-safe summary of a validation failure: field locs + error types only.

    Never `str(e)`, which embeds the raw input values.
    """

    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc']) or '<record>'}: {err['type']}"
        for err in e.errors(include_url=False, include_input=False)
    )


def validate_each[ModelT: ApiModel](
    model: type[ModelT],
    raw: list[object],
    *,
    strict: bool = False,
) -> list[ModelT]:
    """Validate each raw record into `model`, warning on and skipping the ones that fail.

    With `strict=True` a non-empty `raw` that validates to NOTHING raises `BoundaryContractError`.
    """

    validated: list[ModelT] = []
    for index, record in enumerate(raw):
        try:
            validated.append(model.model_validate(record))
        except ValidationError as e:
            # Deferred: a top-level .output import cycles back here through manual_import.
            # Skip-arm only, so the all-valid hot path never touches the import machinery.
            from .output import hub_warn

            hub_warn(f"Skipping malformed {model.__name__} record [{index}] ({validation_summary(e)})")
    if strict and raw and not validated:
        msg = f"none of the {len(raw)} {model.__name__} records validated - refusing to treat it as empty"
        raise BoundaryContractError(msg)
    return validated


def _str_or_none(value: object) -> str | None:
    """Per-field lenient fold: keep a str, fold any other shape to None."""

    return value if isinstance(value, str) else None


def _str_or_blank(value: object) -> str:
    """Per-field lenient fold: keep a str, fold any other shape to ""."""

    return value if isinstance(value, str) else ""


def _datetime_or_none(value: object) -> datetime | None:
    """Per-field lenient fold: a zoned datetime or zoned ISO 8601 string parses, anything else folds to None."""

    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    return parsed if isinstance(parsed, datetime) and parsed.tzinfo is not None else None


def _day_or_none(value: object) -> date | None:
    """Per-field lenient fold: a date, or an ISO date string such as `2020-08-29`, parses. Anything else folds to None."""

    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _fuzzy_date_or_none(value: object) -> date | None:
    """Per-field lenient fold: AniList's `{year, month, day}` parses when it names a real day.

    AniList sends null for a part it doesn't know, and a date with any part missing folds to None.
    """

    if isinstance(value, date):
        return value
    if not isinstance(value, dict):
        return None
    parts = cast("dict[str, object]", value)
    year, month, day = parts.get("year"), parts.get("month"), parts.get("day")
    if not (isinstance(year, int) and isinstance(month, int) and isinstance(day, int)):
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _int_or_zero(value: object) -> int:
    """Per-field lenient fold: best-effort int, folding junk/None to 0."""

    return coerce_int(value) or 0


def stringified(value: object) -> str:
    """Per-field lenient fold: `str(value or "")` (a falsy value reads as "")."""

    return str(value or "")


def none_if_falsy(value: object) -> object:
    """Fold a falsy value (`{}`/None) to None before nested validation."""

    return value or None


def _lax_bool(value: object) -> bool:
    """Per-field lenient fold: real bools and recognized spellings parse, junk folds to False."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().casefold() in ("true", "1")
    return False


# Reusable lenient field shapes.
type LenientStr = Annotated[str | None, BeforeValidator(_str_or_none)]


type BlankStr = Annotated[str, BeforeValidator(_str_or_blank)]


type ZeroInt = Annotated[int, BeforeValidator(_int_or_zero)]


type LenientInt = Annotated[int | None, BeforeValidator(coerce_int)]


type LenientDay = Annotated[date | None, BeforeValidator(_day_or_none)]


type FuzzyDate = Annotated[date | None, BeforeValidator(_fuzzy_date_or_none)]


type LenientTime = Annotated[datetime | None, BeforeValidator(_datetime_or_none)]


type LaxBool = Annotated[bool, BeforeValidator(_lax_bool)]


def validate_skipping_junk[ModelT: ApiModel](model: type[ModelT], value: object) -> object:
    """Nested-array before-validator: validate each entry into `model`, skipping junk without failing the array."""

    if not isinstance(value, list):
        return ()
    kept: list[ModelT] = []
    for entry in cast("list[object]", value):
        try:
            kept.append(model.model_validate(entry))
        except ValidationError:
            continue
    return kept


def validate_all_or_none[ModelT: ApiModel](model: type[ModelT], value: object) -> list[ModelT] | None:
    """Validate every entry into `model`, or None when the value isn't a list or ANY entry fails."""

    # tuple included: direct construction passes the field's own type.
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [model.model_validate(entry) for entry in cast("Sequence[object]", value)]
    except ValidationError:
        return None
