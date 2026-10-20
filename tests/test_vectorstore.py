"""Tests for the vector store, using a tiny fake embedder so they run fast and offline."""
import unittest

try:
    from qdrant_client import QdrantClient
except ImportError:  # qdrant-client not installed yet
    QdrantClient = None

from docsagent.chunking import Chunk


class KeywordEmbedder:
    """Fake embedder: one dimension per keyword. Good enough to test storing and searching."""
    WORDS = ["leave", "expense", "incident", "march", "receipt", "page"]
    dim = len(WORDS)

    def _vec(self, text):
        text = text.lower()
        return [float(text.count(w)) + 0.01 for w in self.WORDS]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@unittest.skipIf(QdrantClient is None, "qdrant-client is not installed")
class VectorStoreTest(unittest.TestCase):
    def test_closest_chunk_comes_first_and_keeps_its_source(self):
        from docsagent.vectorstore import VectorStore

        store = VectorStore(KeywordEmbedder(), client=QdrantClient(":memory:"))
        store.add([
            Chunk("leave.md", 0, "Leave carries over until March.", "Carry-over"),
            Chunk("expense.md", 0, "Every expense needs a receipt.", "Claims"),
            Chunk("incident.md", 0, "Acknowledge the incident page in 5 minutes.", "Paging"),
        ])
        self.assertEqual(store.count(), 3)
        top, score = store.search("Which receipt do I need for an expense?", k=1)[0]
        self.assertEqual(top.source, "expense.md")
        self.assertEqual(top.label(), "expense.md › Claims")
        self.assertGreater(score, 0.5)

    def test_adding_the_same_chunk_twice_does_not_duplicate_it(self):
        from docsagent.vectorstore import VectorStore

        store = VectorStore(KeywordEmbedder(), client=QdrantClient(":memory:"))
        chunk = Chunk("leave.md", 0, "Leave carries over until March.")
        store.add([chunk])
        store.add([chunk])
        self.assertEqual(store.count(), 1)


if __name__ == "__main__":
    unittest.main()
