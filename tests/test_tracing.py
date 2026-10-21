"""Tests for tracing: one trace per question, a span per step and per model call."""
import os
import unittest
from types import SimpleNamespace as NS
from unittest import mock

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:   # opentelemetry-sdk not installed yet
    InMemorySpanExporter = None
try:
    import langgraph  # noqa: F401
except ImportError:
    langgraph = None

from docsagent import tracing
from docsagent.chunking import Chunk

CARRY = Chunk("leave-policy.md", 2, "You can carry over up to 5 unused days.", "Carry-over")


class FakeStore:
    def search(self, query, k):
        return [(CARRY, 0.9)]


class FakeClaudeClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        return NS(content=[NS(type="text", text="You can carry over up to 5 days [1].")],
                  usage=NS(input_tokens=200, output_tokens=12))


@unittest.skipIf(InMemorySpanExporter is None, "opentelemetry-sdk is not installed")
class TracingTest(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        tracing.setup_tracing(self.exporter)
        self.addCleanup(tracing.reset)

    def spans(self):
        return {span.name: span for span in self.exporter.get_finished_spans()}

    def test_model_call_span_uses_the_genai_conventions(self):
        from docsagent.llm import AnthropicLLM

        AnthropicLLM(model="claude-test", client=FakeClaudeClient())("system", "question")
        span = self.spans()["chat claude-test"]
        self.assertEqual(span.attributes["gen_ai.provider.name"], "anthropic")
        self.assertEqual(span.attributes["gen_ai.request.model"], "claude-test")
        self.assertEqual(span.attributes["gen_ai.usage.input_tokens"], 200)
        self.assertEqual(span.attributes["gen_ai.usage.output_tokens"], 12)

    @unittest.skipIf(langgraph is None, "langgraph is not installed")
    def test_one_trace_per_question_with_a_span_per_step(self):
        from docsagent.agent import ask, build_agent
        from docsagent.llm import AnthropicLLM

        state = ask(build_agent(FakeStore(), AnthropicLLM(model="claude-test", client=FakeClaudeClient())),
                    "How many days can I carry over?")
        spans = self.spans()
        root = spans["ask"]
        for step in ("retrieve", "generate", "check"):
            self.assertEqual(spans[step].parent.span_id, root.context.span_id)
            self.assertEqual(spans[step].context.trace_id, root.context.trace_id)
        self.assertEqual(spans["chat claude-test"].parent.span_id, spans["generate"].context.span_id)
        self.assertEqual(root.attributes["docsagent.llm.input_tokens"], 200)
        self.assertEqual(state["usage"]["calls"], 1)

    @unittest.skipIf(langgraph is None, "langgraph is not installed")
    def test_question_text_is_recorded_only_when_allowed(self):
        from docsagent.agent import ask, build_agent

        agent = build_agent(FakeStore(), lambda system, user: "You can carry over up to 5 days [1].")
        with mock.patch.dict(os.environ, {"DOCSAGENT_TRACE_CONTENT": ""}):
            ask(agent, "I earn 50 lakh a year. How many days can I carry over?")
        self.assertNotIn("docsagent.question", self.spans()["ask"].attributes)
        self.exporter.clear()
        with mock.patch.dict(os.environ, {"DOCSAGENT_TRACE_CONTENT": "1"}):
            ask(agent, "How many days can I carry over?")
        self.assertEqual(self.spans()["ask"].attributes["docsagent.question"], "How many days can I carry over?")


class TracingOffTest(unittest.TestCase):
    def test_tracing_is_off_unless_configured(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(tracing.setup_tracing())
        with tracing.span("step", **{"docsagent.hits": 1}) as span:   # still safe to use
            self.assertFalse(span.is_recording())


if __name__ == "__main__":
    unittest.main()
