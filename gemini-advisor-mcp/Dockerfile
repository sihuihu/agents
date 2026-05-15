FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    "mcp[cli]>=1.0.0" \
    "google-genai>=1.0.0" \
    "uvicorn>=0.29.0" \
    "starlette>=0.37.0"

COPY server.py .

# Cloud Run injects $PORT; default 8080
ENV PORT=8080
EXPOSE 8080

CMD ["python", "server.py", "--sse"]
