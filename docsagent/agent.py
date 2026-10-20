"""Stage 2, step 2: the agent, built as a LangGraph graph.

    START → retrieve → (good match?) ─yes→ generate → check → END
                 ▲            │
                 └─ rewrite ←─no (once)

retrieve  finds the closest chunks in Qdrant.
rewrite   if even the best chunk is a weak match, asks Claude to rephrase the question as a
          search query and tries once more (people ask "can I bring my dog?" when the
          handbook says "pets").
generate  gives Claude the numbered chunks; it must cite [n] after every sentence.
check     removes every sentence that fails the source check (see citations.py).

Each node returns only the fields it changes; LangGraph merges them into the shared state.
"""
from __future__ import annotations

import operator
import os
from typing import Annotated, Callable, TypedDict

from .chunking import Chunk
from .citations import NOT_FOUND, CheckedAnswer, check_citations, number_sources

# An LLM here is any function (system prompt, user message) -> text. That keeps the agent
# testable: tests pass a fake function instead of calling Claude.
LLM = Callable[[str, str], str]

ANSWER_SYSTEM = f"""You answer questions using ONLY the numbered sources given to you.
Rules:
- End every sentence with the number of the source it comes from, like [1] or [1][2].
- Use only facts that are stated in the sources. Never add outside knowledge.
- If the sources do not answer the question, reply exactly: {NOT_FOUND}
- Answer in at most 4 short sentences."""

REWRITE_SYSTEM = """Rewrite the user's question as a short search query for a company
handbook, using the words a policy document would use. Reply with the query only."""


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    hits: list[tuple[Chunk, float]]
    rewrites: int
    draft: str
    checked: CheckedAnswer
    # A reducer: each node's "steps" are appended instead of replacing the list.
    steps: Annotated[list[str], operator.add]


class ClaudeLLM:
    """Calls Claude. The API key is read from the ANTHROPIC_API_KEY environment variable."""

    def __init__(self, model: str | None = None, max_tokens: int = 600):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("DOCSAGENT_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens

    def __call__(self, system: str, user: str) -> str:
        message = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()


def build_agent(store, llm: LLM, k: int = 4, min_score: float = 0.55, max_rewrites: int = 1):
    from langgraph.graph import END, START, StateGraph

    def retrieve(state: AgentState) -> AgentState:
        query = state.get("search_query") or state["question"]
        return {"hits": store.search(query, k), "steps": [f"retrieve ({query})"]}

    def route(state: AgentState) -> str:
        best = max((score for _, score in state["hits"]), default=0.0)
        if best >= min_score or state.get("rewrites", 0) >= max_rewrites:
            return "generate"
        return "rewrite"

    def rewrite(state: AgentState) -> AgentState:
        query = llm(REWRITE_SYSTEM, state["question"]).strip() or state["question"]
        return {"search_query": query, "rewrites": state.get("rewrites", 0) + 1, "steps": ["rewrite"]}

    def generate(state: AgentState) -> AgentState:
        chunks = [chunk for chunk, _ in state["hits"]]
        if not chunks:
            return {"draft": NOT_FOUND, "steps": ["generate (no sources)"]}
        prompt = f"Sources:\n\n{number_sources(chunks)}\n\nQuestion: {state['question']}"
        return {"draft": llm(ANSWER_SYSTEM, prompt), "steps": ["generate"]}

    def check(state: AgentState) -> AgentState:
        chunks = [chunk for chunk, _ in state["hits"]]
        return {"checked": check_citations(state["draft"], chunks), "steps": ["check"]}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("check", check)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", route, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "check")
    graph.add_edge("check", END)
    return graph.compile()


def ask(agent, question: str) -> AgentState:
    return agent.invoke({"question": question, "rewrites": 0, "steps": []})
