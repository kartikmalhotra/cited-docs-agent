FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time, so the container starts without fetching it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY docsagent ./docsagent
COPY sample_docs ./sample_docs
COPY evals ./evals
RUN python -m docsagent ingest sample_docs

# ANTHROPIC_API_KEY is passed at run time: docker run -e ANTHROPIC_API_KEY ... (never baked in)
EXPOSE 8000
CMD ["uvicorn", "docsagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
