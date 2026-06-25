"""Minimal local type stubs for the (un-stubbed) ``discordwebhook`` package.

Covers ONLY the surface ``discord.py`` uses: constructing ``Discord(url=...)``
and calling ``.post(embeds=[...])``. The package ships no ``py.typed``, so under
strict the import trips ``reportMissingTypeStubs`` and the ``.post`` call cascades
to ``reportUnknownMemberType``; typing just the two members the code touches
clears both.

``post`` is keyword-only (matching the runtime). ``embeds`` is typed
``list[dict[str, object]] | None`` - permissive enough for the nested
embed payload the caller builds (author/title/description/fields/thumbnail, whose
values are strings, ``None``, and nested dicts/lists, all ``object``) while keeping
the member typed rather than ``Unknown``. The return is untyped at the call site
(the caller ignores it), so it's left ``object``.
"""

class Discord:
    def __init__(self, *, url: str) -> None: ...
    def post(
        self,
        *,
        content: str | None = ...,
        username: str | None = ...,
        avatar_url: str | None = ...,
        tts: bool = ...,
        file: object | None = ...,
        embeds: list[dict[str, object]] | None = ...,
        allowed_mentions: object | None = ...,
    ) -> object: ...
