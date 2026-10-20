"""Stage 1, step 2: turn chunks into vectors and keep them in Qdrant, a vector database.

An embedding model turns text into a list of numbers (a vector) so that texts with similar
meaning get similar vectors. Qdrant stores those vectors and quickly finds the ones closest
to a question's vector. That is "semantic search": it matches meaning, not exact words.
"""
from __future__ import annotations

import os
from typing import Iterable, Protocol

from .chunking import Chunk

COLLECTION = "docs"
DEFAULT_EMBED_MODEL = os.environ.get("DOCSAGENT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """A small open-source embedding model that runs on your own computer.

    No API key and no cost. The model (about 70 MB) downloads once on first use.
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.dim = len(self.embed_query("probe"))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # Some models (like BGE) embed questions slightly differently from passages.
        return next(iter(self.model.query_embed(text))).tolist()


class VectorStore:
    """Chunks in, closest chunks out."""

    def __init__(self, embedder: Embedder, path: str = "qdrant_data", client=None):
        from qdrant_client import QdrantClient, models

        self._models = models
        self.embedder = embedder
        # Local mode: Qdrant keeps its data in a folder on disk, so no server is needed.
        # Pass client=QdrantClient(":memory:") in tests.
        self.client = client or QdrantClient(path=path)
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
            )

    def add(self, chunks: Iterable[Chunk], batch_size: int = 64) -> int:
        chunks = list(chunks)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.embedder.embed_passages([c.text for c in batch])
            self.client.upsert(COLLECTION, points=[
                self._models.PointStruct(
                    id=c.id, vector=v,
                    payload={"source": c.source, "index": c.index, "text": c.text,
                             "heading": c.heading, "page": c.page},
                )
                for c, v in zip(batch, vectors)
            ])
        return len(chunks)

    def search(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the k chunks closest in meaning to the question, with a similarity score (0-1)."""
        hits = self.client.query_points(
            COLLECTION, query=self.embedder.embed_query(question), limit=k, with_payload=True,
        ).points
        return [(Chunk(**hit.payload), hit.score) for hit in hits]

    def count(self) -> int:
        return self.client.count(COLLECTION).count
