"""Reranking: a second, sharper model re-orders the search results.

Vector search turns the question and each chunk into vectors separately and compares them.
That is fast, but it never reads the two together. A cross-encoder reads the question and a
chunk side by side and scores how well that chunk answers the question. It is more accurate
but too slow to run over every chunk, so the two are combined:

    vector search: 8 candidates, fast  ──►  cross-encoder: best 4, accurate
"""
from __future__ import annotations

import os
from typing import Protocol

from .chunking import Chunk

DEFAULT_RERANK_MODEL = os.environ.get("DOCSAGENT_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

Hits = list[tuple[Chunk, float]]


class Reranker(Protocol):
    def scores(self, question: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A small open-source cross-encoder that runs locally (no API key, no cost)."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name, cache_dir=os.environ.get("DOCSAGENT_MODEL_CACHE"))

    def scores(self, question: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(question, texts)]


def rerank(question: str, hits: Hits, reranker: Reranker | None, k: int) -> Hits:
    """Order hits by the reranker's score and keep the best k.

    Each hit keeps its original vector-search score, which the agent uses to judge whether
    the match is good enough. Reranker scores are on a different scale, so they only decide
    the order.
    """
    if reranker is None or len(hits) < 2:
        return hits[:k]
    scores = reranker.scores(question, [chunk.text for chunk, _ in hits])
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i] for i in order[:k]]
