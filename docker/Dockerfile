FROM node:20-alpine AS frontend-builder
WORKDIR /app/web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY web/frontend/ .
RUN npm run build

FROM python:3.12-slim AS python-builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --user ".[web,dev]"

FROM python:3.12-slim
WORKDIR /app
COPY --from=python-builder /root/.local /root/.local
COPY --from=frontend-builder /app/web/static/dist /app/web/static/dist
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV DATABASE_URL=sqlite:///data/trading.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["trading-agent", "serve"]