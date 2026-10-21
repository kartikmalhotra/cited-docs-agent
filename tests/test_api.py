"""Tests for the REST API, with a fake agent and a fake search (no model calls)."""
import json
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


class FakeStreamingAgent(FakeAgent):
    """Like LangGraph's stream_mode="values": the whole state after every step."""

    def stream(self, state, stream_mode="values"):
        yield dict(state)
        yield {**state, "steps": ["retrieve"]}
        yield {**state, **self.invoke(state)}


class FakeStore:
    def search(self, query, k):
        return [(CARRY, 0.82)]


def parse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        event_line, data_line = block.split("\n")
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


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
        self.assertEqual(body["usage"]["calls"], 0)
        self.assertEqual(client.get("/health").json(), {"ok": True})

    def test_stream_sends_each_step_then_the_checked_answer(self):
        from docsagent import api

        api.set_agent(FakeStreamingAgent())
        response = TestClient(api.app).post("/ask/stream", json={"question": "Carry over?"})
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = parse_events(response.text)
        self.assertEqual([name for name, _ in events], ["step", "step", "step", "answer"])
        self.assertEqual([data["step"] for name, data in events if name == "step"], ["retrieve", "generate", "check"])
        self.assertEqual(events[-1][1]["answer"], "You can carry over up to 5 days [1].")   # unsupported sentence never sent

    def test_stream_reports_a_failure_instead_of_cutting_off(self):
        from docsagent import api

        class BrokenAgent:
            def stream(self, state, stream_mode="values"):
                raise RuntimeError("the model provider is down")

        api.set_agent(BrokenAgent())
        with self.assertLogs("docsagent.api", "ERROR"):
            events = parse_events(TestClient(api.app).post("/ask/stream", json={"question": "Carry over?"}).text)
        self.assertEqual(events[-1][0], "error")
        self.assertNotIn("provider is down", events[-1][1]["message"])   # no internal details leak out

    def test_search_returns_sources_without_calling_a_model(self):
        from docsagent import api

        api.set_agent(None, store=FakeStore(), reranker=None)
        body = TestClient(api.app).get("/search", params={"q": "carry over", "k": 2}).json()
        self.assertEqual(body["results"][0]["label"], "leave-policy.md › Carry-over")
        self.assertEqual(body["blocked_sources"], [])

    def test_empty_question_is_rejected(self):
        from docsagent import api

        api.set_agent(FakeAgent())
        self.assertEqual(TestClient(api.app).post("/ask", json={"question": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
