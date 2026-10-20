"""Tests for the LangGraph agent, with a fake search and a fake LLM (no API calls)."""
import unittest

try:
    import langgraph  # noqa: F401
except ImportError:
    langgraph = None

from docsagent.chunking import Chunk

CARRY_OVER = Chunk("leave-policy.md", 2, "You can carry over up to 5 unused days.", "Carry-over")


class FakeStore:
    def __init__(self, score_for):
        self.score_for = score_for
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return [(CARRY_OVER, self.score_for(query))]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, system, user):
        self.calls.append(system.split("\n")[0])
        if "Rewrite" in system:
            return "annual leave carry over"
        return "You can carry over up to 5 days [1]. Everyone loves holidays."


@unittest.skipIf(langgraph is None, "langgraph is not installed")
class AgentTest(unittest.TestCase):
    def test_good_match_goes_straight_to_the_answer_and_drops_the_unsourced_sentence(self):
        from docsagent.agent import ask, build_agent

        store, llm = FakeStore(lambda q: 0.9), FakeLLM()
        state = ask(build_agent(store, llm), "How many days can I carry over?")
        self.assertEqual(state["checked"].text, "You can carry over up to 5 days [1].")
        self.assertEqual(state["checked"].dropped[0][0], "Everyone loves holidays.")
        self.assertEqual([s.split(" ")[0] for s in state["steps"]], ["retrieve", "generate", "check"])

    def test_weak_match_rewrites_the_question_once(self):
        from docsagent.agent import ask, build_agent

        store = FakeStore(lambda q: 0.9 if q == "annual leave carry over" else 0.2)
        state = ask(build_agent(store, FakeLLM()), "Can I keep my unused days?")
        self.assertEqual(store.queries, ["Can I keep my unused days?", "annual leave carry over"])
        self.assertIn("rewrite", state["steps"])
        self.assertTrue(state["checked"].found)

    def test_rewrites_stop_after_one_try(self):
        from docsagent.agent import ask, build_agent

        store = FakeStore(lambda q: 0.1)
        ask(build_agent(store, FakeLLM()), "Something unrelated")
        self.assertEqual(len(store.queries), 2)


if __name__ == "__main__":
    unittest.main()
