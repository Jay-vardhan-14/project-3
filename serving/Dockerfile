# --- build stage: resolve serving deps into /usr/local via uv ---
FROM python:3.11-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra serving --no-dev --no-install-project

# --- runtime stage: non-root, slim ---
FROM python:3.11-slim

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /usr/local /usr/local

WORKDIR /app
COPY serving/app ./app
RUN chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
