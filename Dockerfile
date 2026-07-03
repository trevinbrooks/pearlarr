# Pinned to the supported minor (see requires-python) rather than :latest, so
# image rebuilds don't silently jump Python versions.
FROM python:3.14-slim
WORKDIR /app
COPY . /app
RUN pip install setuptools
RUN pip install -e .

# One mounted volume holds config, caches and logs (see seadexarr paths).
ENV SEADEX_ARR_DATA_DIR=/config

ENTRYPOINT ["seadexarr"]
