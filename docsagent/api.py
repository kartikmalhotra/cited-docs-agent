"""A small REST API around the agent.

    POST /ask          the answer, its sources and what was removed (JSON)
    POST /ask/stream   the same, streamed as server-sent events: each agent step as it
                       finishes, then the checked answer
    GET  /search?q=    the closest document pieces (no model call, no API key needed)
    GET  /health

Run:  uvicorn docsagent.api:app --reload
Then: curl -X POST localhost:8000/ask -H 'content-type: application/json' \
           -d '{"question": "How many leave days can I carry over?"}'
"""
from __future__ import annotations

import json
import logging
import queue
import threading

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import tracing
from .agent import ask, find_sources

log = logging.getLogger(__name__)
app = FastAPI(title="cited-docs-agent", description="Answers from your documents, with a source for every sentence.")
tracing.setup_tracing()   # does nothing unless tracing is configured (see tracing.py)

_UNSET = object()
_agent = None
_store = None
_reranker = _UNSET


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def set_agent(agent, store=None, reranker=None) -> None:
    """Swap in a different agent, store and reranker (tests use fakes)."""
    global _agent, _store, _reranker
    _agent, _store, _reranker = agent, store, reranker


def _get_store():
    global _store
    if _store is None:
        from .vectorstore import FastEmbedder, VectorStore

        _store = VectorStore(FastEmbedder())
    return _store


def _get_reranker():
    global _reranker
    if _reranker is _UNSET:
        from .rerank import CrossEncoderReranker

        _reranker = CrossEncoderReranker()
    return _reranker


def _get_agent():
    global _agent
    if _agent is None:
        from .agent import build_agent
        from .llm import make_llm

        _agent = build_agent(_get_store(), make_llm(), reranker=_get_reranker())
    return _agent


def answer_json(state: dict) -> dict:
    checked, hits = state["checked"], state["hits"]
    return {
        "answer": checked.text,
        "found": checked.found,
        "sources": [{"n": n, "label": hits[n - 1][0].label(), "score": round(hits[n - 1][1], 3)}
                    for n in checked.cited],
        "removed": [{"sentence": s, "reason": r} for s, r in checked.dropped],
        "blocked_sources": [{"label": c.label(), "matched": phrase} for c, phrase in state.get("blocked", [])],
        "steps": state["steps"],
        "usage": state.get("usage"),
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/search")
def search(q: str = Query(min_length=1, max_length=500), k: int = Query(4, ge=1, le=20)) -> dict:
    hits, blocked = find_sources(_get_store(), q, reranker=_get_reranker(), k=k, fetch_k=max(8, 2 * k))
    return {
        "results": [{"label": c.label(), "score": round(s, 3), "text": c.text} for c, s in hits],
        "blocked_sources": [{"label": c.label(), "matched": phrase} for c, phrase in blocked],
    }


@app.post("/ask")
def ask_question(body: Question) -> dict:
    return answer_json(ask(_get_agent(), body.question))


@app.post("/ask/stream")
def ask_stream(body: Question) -> StreamingResponse:
    return StreamingResponse(_events(_get_agent(), body.question), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _events(agent, question: str):
    """Run the agent in a worker thread and pass each step out as soon as it finishes.

    The answer itself is sent only after the source check, so a sentence that gets removed
    is never shown, not even for a moment.
    """
    updates: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            state = ask(agent, question, on_step=lambda step: updates.put(("step", {"step": step})))
            updates.put(("answer", answer_json(state)))
        except Exception:   # tell the client instead of cutting the stream off
            log.exception("The agent failed while answering")
            updates.put(("error", {"message": "Something went wrong while answering. Please try again."}))
        finally:
            updates.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := updates.get()) is not None:
        yield _sse(*item)
