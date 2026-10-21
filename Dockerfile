FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding and reranking models at build time, so the container starts without fetching them.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
 && python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY docsagent ./docsagent
COPY sample_docs ./sample_docs
COPY evals ./evals
RUN python -m docsagent ingest sample_docs

# ANTHROPIC_API_KEY is passed at run time: docker run -e ANTHROPIC_API_KEY ... (never baked in)
EXPOSE 8000
CMD ["uvicorn", "docsagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
