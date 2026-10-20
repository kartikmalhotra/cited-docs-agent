"""Tests for chunking. Run with: python -m pytest  (or python -m unittest discover tests)."""
import unittest
from pathlib import Path

from docsagent.chunking import Chunk, chunk_text, load_folder

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "sample_docs"

TEXT = """# Policy

Intro paragraph.

## Carry-over

You can carry over up to 5 days.

Carried-over days expire on 31 March.

## Sick leave

Ten days a year.
"""


class ChunkingTest(unittest.TestCase):
    def test_each_chunk_keeps_its_heading_and_never_mixes_sections(self):
        chunks = chunk_text(TEXT, "policy.md")
        self.assertEqual([c.heading for c in chunks], ["Policy", "Carry-over", "Sick leave"])
        self.assertIn("31 March", chunks[1].text)
        self.assertNotIn("Ten days", chunks[1].text)

    def test_long_paragraph_is_split_under_the_limit(self):
        paragraph = " ".join(f"Sentence number {i} is here." for i in range(60))
        chunks = chunk_text(paragraph, "long.txt", max_chars=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.text) <= 200 for c in chunks))

    def test_ids_are_stable_and_unique(self):
        first = chunk_text(TEXT, "policy.md")
        again = chunk_text(TEXT, "policy.md")
        self.assertEqual([c.id for c in first], [c.id for c in again])
        self.assertEqual(len({c.id for c in first}), len(first))

    def test_label_names_the_file_and_section(self):
        self.assertEqual(Chunk("a.md", 0, "x", heading="Carry-over").label(), "a.md › Carry-over")
        self.assertEqual(Chunk("b.pdf", 0, "x", page=3).label(), "b.pdf › p. 3")

    def test_sample_docs_load(self):
        chunks = load_folder(SAMPLE_DOCS)
        self.assertEqual({c.source for c in chunks},
                         {"leave-policy.md", "expense-policy.md", "incident-response.md"})
        self.assertTrue(any("31 March" in c.text and c.heading == "Carry-over" for c in chunks))


if __name__ == "__main__":
    unittest.main()
