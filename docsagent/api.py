"""A small REST API around the agent.

Run:  uvicorn docsagent.api:app --reload
Then: curl -X POST localhost:8000/ask -H 'content-type: application/json' \
           -d '{"question": "How many leave days can I carry over?"}'
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import ask

app = FastAPI(title="cited-docs-agent", description="Answers from your documents, with a source for every sentence.")
_agent = None


class Question(BaseModel):
    question: str


def set_agent(agent) -> None:
    """Swap in a different agent (tests use a fake one)."""
    global _agent
    _agent = agent


def _get_agent():
    global _agent
    if _agent is None:
        from .agent import ClaudeLLM, build_agent
        from .vectorstore import FastEmbedder, VectorStore

        _agent = build_agent(VectorStore(FastEmbedder()), ClaudeLLM())
    return _agent


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/ask")
def ask_question(body: Question) -> dict:
    state = ask(_get_agent(), body.question)
    checked, hits = state["checked"], state["hits"]
    return {
        "answer": checked.text,
        "found": checked.found,
        "sources": [{"n": n, "label": hits[n - 1][0].label(), "score": round(hits[n - 1][1], 3)}
                    for n in checked.cited],
        "removed": [{"sentence": s, "reason": r} for s, r in checked.dropped],
        "steps": state["steps"],
    }
