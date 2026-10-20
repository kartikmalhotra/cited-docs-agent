"""Stage 2, step 1: make the model show its sources, then check them.

The model gets numbered sources and must end every sentence with [n]. Afterwards this code
checks every sentence and removes the ones that fail. That is the hallucination guard:
  1. a sentence with no citation is removed,
  2. a citation to a source that doesn't exist (say [7] when there are 4) is removed,
  3. a sentence with a number (5 days, ₹2,500, 31 March) is removed unless that number
     appears in the source it cites. Made-up numbers are the most common hallucination.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import Chunk

NOT_FOUND = "I couldn't find this in the documents."

_CITATION = re.compile(r"\[(\d+)\]")
_ONLY_CITATIONS = re.compile(r"^(\[\d+\]\s*)+[.!?]?$")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def number_sources(chunks: list[Chunk]) -> str:
    """Format chunks as numbered sources for the prompt: [1] (file › section) text."""
    return "\n\n".join(f"[{i}] ({c.label()})\n{c.text}" for i, c in enumerate(chunks, start=1))


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences, keeping a trailing '[1]' with the sentence before it."""
    # "...31 March. [1] Next" → keep "[1]" with "31 March." and start a new sentence after it.
    text = re.sub(r"([.!?])\s+((?:\[\d+\])+)[ \t]*", r"\1 \2\n", text.strip())
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    sentences: list[str] = []
    for part in parts:
        if sentences and _ONLY_CITATIONS.match(part):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part.lstrip("-•* ").strip())
    return [s for s in sentences if s]


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


@dataclass
class CheckedAnswer:
    kept: list[tuple[str, list[int]]] = field(default_factory=list)  # (sentence, source numbers)
    dropped: list[tuple[str, str]] = field(default_factory=list)     # (sentence, reason)

    @property
    def found(self) -> bool:
        return bool(self.kept)

    @property
    def text(self) -> str:
        return " ".join(s for s, _ in self.kept) if self.kept else NOT_FOUND

    @property
    def cited(self) -> list[int]:
        return sorted({n for _, nums in self.kept for n in nums})


def check_citations(draft: str, sources: list[Chunk]) -> CheckedAnswer:
    result = CheckedAnswer()
    if draft.strip() == NOT_FOUND:
        return result
    for sentence in split_sentences(draft):
        cited = [int(n) for n in _CITATION.findall(sentence)]
        if not cited:
            result.dropped.append((sentence, "no source"))
            continue
        if any(n < 1 or n > len(sources) for n in cited):
            result.dropped.append((sentence, "cites a source that doesn't exist"))
            continue
        claim = _CITATION.sub("", sentence)
        source_numbers = set().union(*(_numbers(sources[n - 1].text) for n in cited))
        missing = _numbers(claim) - source_numbers
        if missing:
            result.dropped.append((sentence, f"number not in its source: {', '.join(sorted(missing))}"))
            continue
        result.kept.append((sentence, sorted(set(cited))))
    return result
