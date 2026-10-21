FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HOME=/app \
    DOCSAGENT_MODEL_CACHE=/app/models
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding and reranking models at build time, so the container starts without fetching them.
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=os.environ['DOCSAGENT_MODEL_CACHE'])" \
 && python -c "import os; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir=os.environ['DOCSAGENT_MODEL_CACHE'])"

COPY docsagent ./docsagent
COPY sample_docs ./sample_docs
COPY evals ./evals
RUN python -m docsagent ingest sample_docs

# Run as an unprivileged user (Kubernetes enforces this with runAsNonRoot).
RUN useradd --uid 10001 --home-dir /app --no-create-home --shell /usr/sbin/nologin app \
 && chown -R 10001:10001 /app
USER 10001

# API keys are passed at run time (docker run -e ANTHROPIC_API_KEY ..., or a Kubernetes Secret), never baked in.
EXPOSE 8000
CMD ["uvicorn", "docsagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
