"""Tracing with OpenTelemetry: one trace per question, one span per agent step and model call.

Off unless you turn it on:
    DOCSAGENT_TRACING=console                  print spans in the terminal
    OTEL_EXPORTER_OTLP_ENDPOINT=http://...     send them to any OpenTelemetry backend
                                               (Jaeger, Grafana Tempo, Langfuse, Arize Phoenix)

Model-call spans follow the OpenTelemetry GenAI conventions (gen_ai.* attributes: provider,
model, input and output tokens), which LLM observability tools read directly.

Question text is recorded only with DOCSAGENT_TRACE_CONTENT=1, because it can contain
personal data. Without it, traces hold timings, counts and scores only.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from opentelemetry import trace

_provider = None   # the TracerProvider set by setup_tracing(); None means tracing is off


def setup_tracing(exporter=None):
    """Turn tracing on. Tests pass an exporter; otherwise the environment decides.

    Returns the TracerProvider, or None when tracing stays off.
    """
    global _provider
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    console = os.environ.get("DOCSAGENT_TRACING", "").lower() == "console"
    if exporter is None and not endpoint and not console:
        return None

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "cited-docs-agent"}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    _provider = provider
    return provider


def reset() -> None:
    """Turn tracing off again (used by tests)."""
    global _provider
    _provider = None


def tracer():
    return (_provider or trace).get_tracer("docsagent")


def content(value: str | None) -> str | None:
    """Text to record on a span, only when DOCSAGENT_TRACE_CONTENT=1."""
    return value if os.environ.get("DOCSAGENT_TRACE_CONTENT") == "1" else None


@contextmanager
def span(name: str, **attributes):
    """A span for one step. Attributes with the value None are skipped."""
    with tracer().start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


@contextmanager
def llm_span(provider: str, model: str):
    """A span for one model call. Yields record(usage) to attach the token counts."""
    attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": provider,
        "gen_ai.system": provider,          # older name of the same attribute, still widely read
        "gen_ai.request.model": model,
        "gen_ai.request.temperature": 0.0,
    }
    with tracer().start_as_current_span(f"chat {model}", kind=trace.SpanKind.CLIENT,
                                        attributes=attributes) as current:
        def record(usage: dict) -> None:
            current.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
            current.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))

        yield record


def event(name: str, attributes: dict | None = None) -> None:
    """Add an event (like a fallback to another model) to the current span."""
    trace.get_current_span().add_event(name, attributes or {})
