"""Tests for the REST API, with a fake agent (no model calls)."""
import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:  # fastapi or httpx not installed yet
    TestClient = None

from docsagent.chunking import Chunk
from docsagent.citations import check_citations

CARRY = Chunk("leave-policy.md", 2, "You can carry over up to 5 unused days.", "Carry-over")


class FakeAgent:
    def invoke(self, state):
        hits = [(CARRY, 0.82)]
        draft = "You can carry over up to 5 days [1]. Probably more for seniors."
        return {"hits": hits, "checked": check_citations(draft, [CARRY]), "steps": ["retrieve", "generate", "check"]}


@unittest.skipIf(TestClient is None, "fastapi/httpx is not installed")
class ApiTest(unittest.TestCase):
    def test_ask_returns_answer_sources_and_removed_sentences(self):
        from docsagent import api

        api.set_agent(FakeAgent())
        client = TestClient(api.app)
        body = client.post("/ask", json={"question": "Carry over?"}).json()
        self.assertEqual(body["answer"], "You can carry over up to 5 days [1].")
        self.assertEqual(body["sources"], [{"n": 1, "label": "leave-policy.md › Carry-over", "score": 0.82}])
        self.assertEqual(body["removed"][0]["reason"], "no source")
        self.assertEqual(client.get("/health").json(), {"ok": True})


if __name__ == "__main__":
    unittest.main()
