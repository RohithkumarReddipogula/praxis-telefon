# Praxis-Telefon agent for LiveKit Cloud agent hosting (M5, region eu-central).
#
# No secrets in the image: API keys come in at runtime as LiveKit Cloud secrets
# (environment variables); .dockerignore keeps local key files, recordings, results and
# the local database out of the build context. The database is synthetic and is seeded
# at start (SEED_ON_START), so nothing is kept between restarts.
#
# Base image: the tag LiveKit's own template uses (lk agent dockerfile); a version-pinned
# tag did not exist (first deployment 2026-11-08). Dependencies are pinned by uv.lock.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

# The runtime user first: dependencies and model weights are installed as this user,
# so the turn-detector weights land in the cache the agent reads at runtime.
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app && chown agent:agent /app
USER agent
WORKDIR /app

# Dependencies first (cached layer), then the project.
COPY --chown=agent:agent pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=agent:agent src ./src
RUN uv sync --frozen --no-dev

# VAD and turn-detector weights, baked into the image (no download per call).
RUN uv run --no-sync python -m praxis_telefon.agent.main download-files

# Public demo defaults (agent/demo.py); the limits can be changed via secrets/env.
ENV DEMO_MODE=1 \
    SEED_ON_START=1 \
    DATABASE_PATH=/tmp/praxis.db \
    DEMO_USAGE_DB=/tmp/praxis-demo-usage.db \
    LOG_LEVEL=INFO

CMD ["uv", "run", "--no-sync", "python", "-m", "praxis_telefon.agent.main", "start"]
