"""Stage 1, step 1: turn documents into small pieces ("chunks") that can be searched and cited.

Why chunk at all? A model answers best from a few short, relevant passages, not a whole
document. Each chunk keeps where it came from (file, section, page) so every answer can
point back to its source.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

# Any fixed UUID works here. It makes chunk ids stable: the same file and position always
# give the same id, so ingesting a folder twice updates chunks instead of duplicating them.
_ID_NAMESPACE = uuid.UUID("6f1c2d3e-5a4b-4c8d-9e0f-1a2b3c4d5e6f")

_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    source: str              # file name, shown in citations
    index: int               # position of this chunk within its file
    text: str
    heading: str = ""        # nearest markdown heading above the text
    page: int | None = None  # PDF page number (1-based); None for text files

    @property
    def id(self) -> str:
        return str(uuid.uuid5(_ID_NAMESPACE, f"{self.source}#{self.index}"))

    def label(self) -> str:
        """Short human label used in citations, e.g. 'leave-policy.md › Carry-over'."""
        where = f"p. {self.page}" if self.page else self.heading
        return f"{self.source} › {where}" if where else self.source


def _paragraphs(text: str):
    """Yield (heading, paragraph) pairs. A heading is remembered for the text under it."""
    heading = ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        match = _HEADING.match(lines[0])
        if match:
            heading = match.group(1).strip()
            block = "\n".join(lines[1:]).strip()
            if not block:
                continue
        yield heading, block


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split a paragraph that is too long, at sentence ends where possible."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(paragraph):
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    # A single sentence longer than max_chars is cut hard as a last resort.
    return [p[i:i + max_chars] for p in pieces for i in range(0, len(p), max_chars)]


def chunk_text(text: str, source: str, page: int | None = None,
               max_chars: int = 800, start_index: int = 0) -> list[Chunk]:
    """Pack paragraphs into chunks of up to max_chars. A chunk never mixes two sections."""
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_heading = ""

    def flush() -> None:
        if buffer:
            chunks.append(Chunk(source, start_index + len(chunks), "\n\n".join(buffer),
                                buffer_heading, page))

    for heading, paragraph in _paragraphs(text):
        for piece in _split_long(paragraph, max_chars):
            size = sum(len(b) + 2 for b in buffer) + len(piece)
            if buffer and (heading != buffer_heading or size > max_chars):
                flush()
                buffer.clear()
            if not buffer:
                buffer_heading = heading
            buffer.append(piece)
    flush()
    return chunks


def read_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return (page_number, text) pairs. Text files are one 'page' with no number."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [(n, page.extract_text() or "") for n, page in enumerate(PdfReader(path).pages, start=1)]
    return [(None, path.read_text(encoding="utf-8", errors="replace"))]


def load_folder(folder: str | Path, max_chars: int = 800) -> list[Chunk]:
    """Chunk every supported file in a folder (and its subfolders)."""
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        file_chunks: list[Chunk] = []
        for page, text in read_pages(path):
            file_chunks += chunk_text(text, path.name, page, max_chars, start_index=len(file_chunks))
        chunks += file_chunks
    return chunks
