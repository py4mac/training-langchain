FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Dépendances d'abord (cache Docker)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Code de l'agent
COPY agent.py langgraph.json ./

# Render injecte $PORT ; 2024 par défaut en local
ENV PORT=2024
EXPOSE 2024

# Serveur en mémoire (les threads sont perdus au redémarrage) : OK pour une démo
CMD ["sh", "-c", "uv run --no-sync langgraph dev --host 0.0.0.0 --port ${PORT} --no-browser --no-reload"]
