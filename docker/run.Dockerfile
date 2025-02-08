FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install \
    --no-install-suggests \
    --no-install-recommends \
    -y \
    build-essential \
    curl \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_PATH=/app/venv\
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=1.8.5

WORKDIR /app

RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 -

COPY poetry.lock pyproject.toml /app/

COPY . .

RUN poetry install --no-interaction --no-ansi

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/docker_caches/.bash_history" \
    && echo "$SNIPPET" >> "/root/.bashrc"