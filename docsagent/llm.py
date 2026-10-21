"""Language model providers behind one small interface: llm(system, user) -> text.

    DOCSAGENT_PROVIDER = anthropic | openai | gemini    (default anthropic)
    DOCSAGENT_MODEL    = model name                     (default depends on the provider)
    DOCSAGENT_FALLBACK = anthropic | openai | gemini    (optional: used when the first one fails)

The OpenAI adapter also works with any OpenAI-compatible server, such as Ollama or vLLM for
open-source models: set OPENAI_BASE_URL (for Ollama, http://localhost:11434/v1).

Every call returns a Reply: a normal string that also carries the provider, model, token
counts and time taken, so the agent can report usage and cost per question. Every call is
also traced as an OpenTelemetry span (tracing.py).
"""
from __future__ import annotations

import logging
import os
import time

from . import tracing

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-flash-latest",
}
TIMEOUT_SECONDS = float(os.environ.get("DOCSAGENT_LLM_TIMEOUT", "30"))
REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")   # OpenAI models that reject temperature/max_tokens


class Reply(str):
    """The model's text (it behaves exactly like a str), plus what the call used."""

    usage: dict

    def __new__(cls, text: str, usage: dict | None = None):
        reply = super().__new__(cls, text)
        reply.usage = usage or {}
        return reply


class _Provider:
    provider = ""

    def __init__(self, model: str | None = None, max_tokens: int = 600, client=None):
        self.model = model or DEFAULT_MODELS[self.provider]
        self.max_tokens = max_tokens
        self.client = client if client is not None else self._client()

    def __call__(self, system: str, user: str) -> Reply:
        with tracing.llm_span(self.provider, self.model) as record:
            start = time.perf_counter()
            text, tokens_in, tokens_out = self._complete(system, user)
            usage = {"provider": self.provider, "model": self.model, "input_tokens": int(tokens_in or 0),
                     "output_tokens": int(tokens_out or 0), "seconds": round(time.perf_counter() - start, 3)}
            record(usage)
        return Reply((text or "").strip(), usage)

    def _client(self):
        raise NotImplementedError

    def _complete(self, system: str, user: str) -> tuple[str, int, int]:
        raise NotImplementedError


class AnthropicLLM(_Provider):
    """Claude. Reads ANTHROPIC_API_KEY."""

    provider = "anthropic"

    def _client(self):
        import anthropic

        return anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=2)

    def _complete(self, system, user):
        message = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return text, message.usage.input_tokens, message.usage.output_tokens


class OpenAILLM(_Provider):
    """OpenAI, or any OpenAI-compatible server (set OPENAI_BASE_URL). Reads OPENAI_API_KEY."""

    provider = "openai"

    def _client(self):
        import openai

        return openai.OpenAI(timeout=TIMEOUT_SECONDS, max_retries=2)

    def _complete(self, system, user):
        options = {"max_tokens": self.max_tokens, "temperature": 0}
        if self.model.startswith(REASONING_PREFIXES):
            options = {"max_completion_tokens": self.max_tokens * 4}   # room for reasoning tokens
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **options,
        )
        usage = response.usage
        return (response.choices[0].message.content or "",
                getattr(usage, "prompt_tokens", 0) if usage else 0,
                getattr(usage, "completion_tokens", 0) if usage else 0)


class GeminiLLM(_Provider):
    """Google Gemini through the Gemini API (GEMINI_API_KEY), or through Vertex AI
    (GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION)."""

    provider = "gemini"

    def _client(self):
        from google import genai

        return genai.Client(http_options={"timeout": int(TIMEOUT_SECONDS * 1000)})

    def _complete(self, system, user):
        # No output cap here: on "thinking" models the cap also counts thinking tokens and can
        # cut the answer off. The system prompt already asks for at most 4 short sentences.
        response = self.client.models.generate_content(
            model=self.model, contents=user, config={"system_instruction": system, "temperature": 0},
        )
        meta = response.usage_metadata
        return (response.text or "",
                (getattr(meta, "prompt_token_count", 0) or 0) if meta else 0,
                (getattr(meta, "candidates_token_count", 0) or 0) if meta else 0)


ClaudeLLM = AnthropicLLM   # the original name, kept for existing code
PROVIDERS = {"anthropic": AnthropicLLM, "claude": AnthropicLLM, "openai": OpenAILLM,
             "gemini": GeminiLLM, "google": GeminiLLM}


class FallbackLLM:
    """Try the main model; if the call fails (outage, rate limit, timeout), use the backup."""

    def __init__(self, primary, backup):
        self.primary, self.backup = primary, backup

    def __call__(self, system: str, user: str):
        try:
            return self.primary(system, user)
        except Exception as error:   # any provider error: switch models instead of failing the question
            log.warning("Model call failed (%s); using the fallback model.", type(error).__name__)
            tracing.event("llm.fallback", {"error.type": type(error).__name__})
            reply = self.backup(system, user)
            if isinstance(reply, Reply):
                reply.usage["fallback"] = True
            return reply


def _provider_class(name: str):
    try:
        return PROVIDERS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown model provider {name!r}. Use anthropic, openai or gemini.") from None


def make_llm(provider: str | None = None, model: str | None = None, fallback: str | None = None):
    """Build the model from arguments, or from DOCSAGENT_PROVIDER / DOCSAGENT_MODEL / DOCSAGENT_FALLBACK."""
    provider = provider or os.environ.get("DOCSAGENT_PROVIDER") or "anthropic"
    fallback = fallback or os.environ.get("DOCSAGENT_FALLBACK") or ""
    llm = _provider_class(provider)(model=model or os.environ.get("DOCSAGENT_MODEL") or None)
    if fallback and _provider_class(fallback) is not type(llm):
        backup = _provider_class(fallback)(model=os.environ.get("DOCSAGENT_FALLBACK_MODEL") or None)
        llm = FallbackLLM(llm, backup)
    return llm


def summarize_usage(calls: list[dict]) -> dict:
    """Totals for one question: model calls, tokens, and cost when prices are set."""
    tokens_in = sum(c.get("input_tokens", 0) for c in calls)
    tokens_out = sum(c.get("output_tokens", 0) for c in calls)
    return {"calls": len(calls), "input_tokens": tokens_in, "output_tokens": tokens_out,
            "cost_usd": estimate_cost(tokens_in, tokens_out)}


def estimate_cost(tokens_in: int, tokens_out: int) -> float | None:
    """USD, from DOCSAGENT_PRICE_INPUT and DOCSAGENT_PRICE_OUTPUT (price per million tokens).

    Returns None when they are not set: prices change often, so none are hard-coded here.
    """
    try:
        price_in = float(os.environ["DOCSAGENT_PRICE_INPUT"])
        price_out = float(os.environ["DOCSAGENT_PRICE_OUTPUT"])
    except (KeyError, ValueError):
        return None
    return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6)
