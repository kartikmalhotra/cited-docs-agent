"""Tests for the model providers, with fake SDK clients (no API calls, no keys)."""
import os
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from docsagent import llm


class FakeAnthropic:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return NS(content=[NS(type="text", text=" Up to 5 days [1]. ")], usage=NS(input_tokens=120, output_tokens=9))


class FakeOpenAI:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        if self.fail:
            raise TimeoutError("the provider timed out")
        self.calls.append(kwargs)
        return NS(choices=[NS(message=NS(content="Up to 5 days [1]."))], usage=NS(prompt_tokens=100, completion_tokens=7))


class FakeGemini:
    def __init__(self):
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return NS(text="Up to 5 days [1].", usage_metadata=NS(prompt_token_count=90, candidates_token_count=6))


class ProviderTest(unittest.TestCase):
    def test_claude_reply_is_a_plain_string_that_also_carries_token_counts(self):
        client = FakeAnthropic()
        reply = llm.AnthropicLLM(model="claude-test", client=client)("Be brief.", "Carry over?")
        self.assertEqual(reply, "Up to 5 days [1].")
        self.assertIsInstance(reply, str)
        self.assertEqual((reply.usage["provider"], reply.usage["input_tokens"], reply.usage["output_tokens"]),
                         ("anthropic", 120, 9))
        sent = client.calls[0]
        self.assertEqual((sent["model"], sent["temperature"], sent["system"]), ("claude-test", 0, "Be brief."))

    def test_openai_sends_the_system_prompt_first_at_temperature_zero(self):
        client = FakeOpenAI()
        reply = llm.OpenAILLM(model="gpt-4.1-mini", client=client)("Be brief.", "Carry over?")
        self.assertEqual(reply.usage["output_tokens"], 7)
        sent = client.calls[0]
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])
        self.assertEqual(sent["temperature"], 0)

    def test_openai_reasoning_models_get_no_temperature(self):
        client = FakeOpenAI()
        llm.OpenAILLM(model="gpt-5-mini", client=client)("Be brief.", "Carry over?")
        self.assertNotIn("temperature", client.calls[0])
        self.assertIn("max_completion_tokens", client.calls[0])

    def test_gemini_passes_the_system_instruction(self):
        client = FakeGemini()
        reply = llm.GeminiLLM(model="gemini-test", client=client)("Be brief.", "Carry over?")
        self.assertEqual(reply, "Up to 5 days [1].")
        self.assertEqual(client.calls[0]["config"]["system_instruction"], "Be brief.")
        self.assertEqual(reply.usage["input_tokens"], 90)

    def test_each_provider_has_a_default_model(self):
        self.assertEqual(llm.AnthropicLLM(client=FakeAnthropic()).model, llm.DEFAULT_MODELS["anthropic"])
        self.assertEqual(llm.OpenAILLM(client=FakeOpenAI()).model, llm.DEFAULT_MODELS["openai"])
        self.assertEqual(llm.GeminiLLM(client=FakeGemini()).model, llm.DEFAULT_MODELS["gemini"])


class FallbackTest(unittest.TestCase):
    def test_backup_answers_when_the_main_model_fails(self):
        main = llm.OpenAILLM(model="gpt-4.1-mini", client=FakeOpenAI(fail=True))
        backup = llm.AnthropicLLM(model="claude-test", client=FakeAnthropic())
        with self.assertLogs("docsagent.llm", "WARNING"):
            reply = llm.FallbackLLM(main, backup)("Be brief.", "Carry over?")
        self.assertEqual(reply, "Up to 5 days [1].")
        self.assertEqual(reply.usage["provider"], "anthropic")
        self.assertTrue(reply.usage["fallback"])

    def test_backup_is_not_called_when_the_main_model_works(self):
        backup = FakeAnthropic()
        llm.FallbackLLM(llm.OpenAILLM(client=FakeOpenAI()), llm.AnthropicLLM(client=backup))("Be brief.", "Hi")
        self.assertEqual(backup.calls, [])


class MakeLLMTest(unittest.TestCase):
    def setUp(self):
        class Fake:
            def __init__(self, model=None):
                self.model = model

        self.fakes = {name: type(name, (Fake,), {}) for name in ("anthropic", "openai", "gemini")}
        patcher = mock.patch.dict(llm.PROVIDERS, self.fakes)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reads_provider_model_and_fallback_from_the_environment(self):
        env = {"DOCSAGENT_PROVIDER": "openai", "DOCSAGENT_MODEL": "gpt-test", "DOCSAGENT_FALLBACK": "gemini"}
        with mock.patch.dict(os.environ, env):
            made = llm.make_llm()
        self.assertIsInstance(made, llm.FallbackLLM)
        self.assertIsInstance(made.primary, self.fakes["openai"])
        self.assertEqual(made.primary.model, "gpt-test")
        self.assertIsInstance(made.backup, self.fakes["gemini"])

    def test_default_is_claude_with_no_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            made = llm.make_llm()
        self.assertIsInstance(made, self.fakes["anthropic"])
        self.assertIsNone(made.model)   # the provider class picks its own default

    def test_unknown_provider_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown model provider"):
            llm.make_llm(provider="watson")


class UsageTest(unittest.TestCase):
    def test_totals_always_and_cost_only_when_prices_are_set(self):
        calls = [{"input_tokens": 1000, "output_tokens": 100}, {"input_tokens": 500, "output_tokens": 50}]
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm.summarize_usage(calls),
                             {"calls": 2, "input_tokens": 1500, "output_tokens": 150, "cost_usd": None})
        with mock.patch.dict(os.environ, {"DOCSAGENT_PRICE_INPUT": "1", "DOCSAGENT_PRICE_OUTPUT": "5"}):
            self.assertAlmostEqual(llm.summarize_usage(calls)["cost_usd"], 0.00225)


if __name__ == "__main__":
    unittest.main()
