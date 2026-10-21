"""Stage 3: measure the agent instead of trusting it.

evals/questions.jsonl holds questions with known answers (and some with no answer in the
documents). For each one we check four things:

  retrieval   did search find the right section?            (no API key needed)
  answer      does the final answer contain the key facts?
  citation    does the answer cite the right section?
  refusal     for questions the documents can't answer, did it say so instead of guessing?

Run:  python -m docsagent eval                   (full agent, needs ANTHROPIC_API_KEY)
      python -m docsagent eval --retrieval-only  (search only, free)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from .citations import NOT_FOUND

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "evals" / "questions.jsonl"


def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace(",", "").split())


def contains_facts(answer: str, must_include: list[str]) -> bool:
    """Every item must appear in the answer; 'a|b' means either a or b."""
    answer = _normalize(answer)
    return all(any(_normalize(option) in answer for option in item.split("|")) for item in must_include)


@dataclass
class CaseResult:
    question: str
    answerable: bool
    retrieved: bool | None = None   # None when not measured
    top1: bool | None = None        # the right section was ranked first
    answered: bool | None = None
    cited: bool | None = None
    refused: bool | None = None
    answer: str = ""
    seconds: float | None = None     # time to answer this question (or to retrieve, in retrieval-only runs)

    @property
    def passed(self) -> bool | None:
        """None when nothing was measured (e.g. an unanswerable question in a retrieval-only run)."""
        checks = [self.retrieved, self.answered, self.cited] if self.answerable else [self.refused]
        measured = [c for c in checks if c is not None]
        return all(measured) if measured else None


def _is_expected(chunk, case: dict) -> bool:
    return chunk.source == case["source"] and chunk.heading == case["section"]


def score_retrieval(case: dict, hits) -> CaseResult:
    result = CaseResult(case["question"], case["answerable"])
    if case["answerable"]:
        result.retrieved = any(_is_expected(chunk, case) for chunk, _ in hits)
        result.top1 = bool(hits) and _is_expected(hits[0][0], case)
    return result


def score_answer(case: dict, state: dict) -> CaseResult:
    result = score_retrieval(case, state["hits"])
    checked = state["checked"]
    result.answer = checked.text
    if case["answerable"]:
        result.answered = checked.found and contains_facts(checked.text, case["must_include"])
        result.cited = any(_is_expected(state["hits"][n - 1][0], case) for n in checked.cited)
    else:
        result.refused = checked.text == NOT_FOUND
    return result


def summarize(results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    def rate(values):
        values = [v for v in values if v is not None]
        return sum(values), len(values)

    return {
        "retrieval": rate(r.retrieved for r in results),
        "top-1": rate(r.top1 for r in results),
        "answer": rate(r.answered for r in results),
        "citation": rate(r.cited for r in results),
        "refusal": rate(r.refused for r in results),
        "overall": rate(r.passed for r in results),
    }


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: p50 is the median, p95 the value 95% of runs stay under."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def latency(results: list[CaseResult]) -> tuple[float, float] | None:
    """(p50, p95) in milliseconds, or None if nothing was timed."""
    times = [r.seconds * 1000 for r in results if r.seconds is not None]
    return (percentile(times, 50), percentile(times, 95)) if times else None


def report(results: list[CaseResult]) -> str:
    mark = {True: "✅", False: "❌", None: "–"}
    lines = ["| # | Question | Retrieval | Top-1 | Answer | Citation | Refusal |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, start=1):
        lines.append(f"| {i} | {r.question} | {mark[r.retrieved]} | {mark[r.top1]} | {mark[r.answered]} | "
                     f"{mark[r.cited]} | {mark[r.refused]} |")
    lines += ["", "| Metric | Score |", "|---|---|"]
    for name, (good, total) in summarize(results).items():
        if total:
            lines.append(f"| {name} | {good}/{total} ({good / total:.0%}) |")
    timing = latency(results)
    if timing:
        lines.append(f"| latency p50 / p95 | {timing[0]:.0f} ms / {timing[1]:.0f} ms |")
    return "\n".join(lines)


def run(cases: list[dict], store, agent=None, reranker=None) -> list[CaseResult]:
    """With an agent: full evaluation. Without one: retrieval only (same search, screening
    and reranking the agent uses)."""
    from .agent import ask, find_sources

    if agent is None and cases:
        find_sources(store, cases[0]["question"], reranker=reranker)  # warm-up, not timed

    results = []
    for case in cases:
        start = time.perf_counter()
        if agent is None:
            hits, _ = find_sources(store, case["question"], reranker=reranker)
            result = score_retrieval(case, hits)
        else:
            result = score_answer(case, ask(agent, case["question"]))
        result.seconds = time.perf_counter() - start
        results.append(result)
    return results
