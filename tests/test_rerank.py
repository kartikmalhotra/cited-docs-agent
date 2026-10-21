"""Tests for reranking, with a fake reranker (no model download)."""
import unittest

from docsagent.chunking import Chunk
from docsagent.rerank import rerank

A = Chunk("a.md", 0, "general leave information", "Leave")
B = Chunk("b.md", 0, "carry over up to 5 days until 31 March", "Carry-over")
C = Chunk("c.md", 0, "meal allowance", "Meals")


class KeywordReranker:
    """Fake cross-encoder: scores a chunk by how many question words it contains."""

    def scores(self, question, texts):
        words = set(question.lower().split())
        return [sum(w in text for w in words) for text in texts]


class RerankTest(unittest.TestCase):
    def test_reranker_reorders_and_keeps_the_best_k(self):
        hits = [(A, 0.80), (C, 0.70), (B, 0.65)]
        out = rerank("how many days can i carry over", hits, KeywordReranker(), k=2)
        self.assertEqual(out[0][0], B)
        self.assertEqual(len(out), 2)

    def test_vector_scores_are_kept_for_the_agent(self):
        out = rerank("carry over days", [(A, 0.80), (B, 0.65)], KeywordReranker(), k=2)
        self.assertEqual(out[0], (B, 0.65))

    def test_without_a_reranker_only_trims(self):
        hits = [(A, 0.8), (B, 0.7), (C, 0.6)]
        self.assertEqual(rerank("q", hits, None, k=2), hits[:2])


if __name__ == "__main__":
    unittest.main()
