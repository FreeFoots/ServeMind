FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend/src
WORKDIR /app
COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/src /app/backend/src
RUN pip install --no-cache-dir '/app/backend[mcp,rag]' && useradd --uid 10001 --create-home app
COPY backend/sql /app/backend/sql
COPY backend/scripts /app/backend/scripts
COPY backend/skills /app/backend/skills
COPY knowledge /app/knowledge
RUN mkdir -p /app/backend/runtime && chown app:app /app/backend/runtime
USER app
EXPOSE 8317
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8317/v1/ready', timeout=3)"
CMD ["uvicorn", "servemind.api.main:app", "--host", "0.0.0.0", "--port", "8317", "--proxy-headers", "--forwarded-allow-ips", "172.30.17.0/24"]
