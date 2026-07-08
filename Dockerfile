# Builder: resolve the locked dependency set into /app/.venv with uv, then
# install the project itself (non-editable, so the venv is self-contained).
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv
# Never fetch a uv-managed CPython: the venv must bind this stage's
# /usr/local/bin/python3.14, which the final stage shares.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1
WORKDIR /app
# Manifests first: the dependency layer only rebuilds when the lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# README/LICENSE feed hatchling's project metadata.
COPY README.md LICENSE ./
COPY seadexarr/ seadexarr/
RUN uv sync --frozen --no-dev --no-editable

# Supercronic (the runtime scheduler): pinned release, per-arch sha256-verified.
FROM python:3.14-slim AS supercronic
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# v0.2.47 sha256 sums, computed from the release binaries after verifying the
# published per-arch SHA1s (the project publishes no sha256 manifest).
RUN case "${TARGETARCH}" in \
        amd64) sha256="dcb1403c188a9438c47d4bba82a9c357fc9351ce91627fb2bae627f0f5becfc4" ;; \
        arm64) sha256="e1124aa34294e2bb8ab7002f347f4363ba35097f3daf4d3c44e9d813c1fb2bb8" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL -o /usr/local/bin/supercronic \
        "https://github.com/aptible/supercronic/releases/download/v0.2.47/supercronic-linux-${TARGETARCH}" \
    && echo "${sha256}  /usr/local/bin/supercronic" | sha256sum -c - \
    && chmod +x /usr/local/bin/supercronic

# Final: the venv, supercronic and the entrypoint on the same slim base.
FROM python:3.14-slim
LABEL org.opencontainers.image.source="https://github.com/trevinbrooks/seadexarr"
# tzdata so TZ drives supercronic's local-time schedule; /config pre-created
# writable so a volume-less or named-volume run works out of the box.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --user-group --no-create-home seadexarr \
    && mkdir /config \
    && chown seadexarr:seadexarr /config
COPY --from=builder /app/.venv /app/.venv
COPY --from=supercronic /usr/local/bin/supercronic /usr/local/bin/supercronic
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh
# One mounted volume holds config, caches and logs (see seadexarr paths).
# HOME=/tmp is a backstop for arbitrary-uid runs (the run path never needs it).
ENV PATH="/app/.venv/bin:${PATH}" \
    SEADEX_ARR_DATA_DIR=/config \
    HOME=/tmp
USER seadexarr
HEALTHCHECK --interval=5m --timeout=30s --start-period=30s CMD ["seadexarr", "paths"]
ENTRYPOINT ["/entrypoint.sh"]
