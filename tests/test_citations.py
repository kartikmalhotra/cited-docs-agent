"""Tests for the source check: the part that removes unsupported sentences."""
import unittest

from docsagent.chunking import Chunk
from docsagent.citations import NOT_FOUND, check_citations, number_sources, split_sentences

SOURCES = [
    Chunk("leave-policy.md", 2, "You can carry over up to 5 unused days. They must be used by 31 March.", "Carry-over"),
    Chunk("expense-policy.md", 1, "The daily meal allowance is ₹2,500 within India.", "Meals"),
]


class CitationCheckTest(unittest.TestCase):
    def test_cited_sentences_are_kept(self):
        result = check_citations("You can carry over up to 5 days [1]. Meals are capped at ₹2,500 a day [2].", SOURCES)
        self.assertEqual(len(result.kept), 2)
        self.assertEqual(result.cited, [1, 2])
        self.assertEqual(result.dropped, [])

    def test_sentence_without_a_source_is_removed(self):
        result = check_citations("You can carry over up to 5 days [1]. Most companies allow this.", SOURCES)
        self.assertEqual([s for s, _ in result.dropped], ["Most companies allow this."])
        self.assertEqual(result.text, "You can carry over up to 5 days [1].")

    def test_citation_to_a_missing_source_is_removed(self):
        result = check_citations("Carry-over ends in March [3].", SOURCES)
        self.assertFalse(result.found)
        self.assertEqual(result.text, NOT_FOUND)

    def test_made_up_number_is_removed_even_with_a_citation(self):
        result = check_citations("You can carry over up to 10 days [1].", SOURCES)
        self.assertEqual(result.dropped[0][1], "number not in its source: 10")

    def test_numbers_with_commas_match(self):
        self.assertTrue(check_citations("Meals are ₹2500 a day [2].", SOURCES).found)

    def test_citation_after_the_full_stop_stays_with_its_sentence(self):
        self.assertEqual(split_sentences("Used by 31 March. [1] Next."), ["Used by 31 March. [1]", "Next."])

    def test_not_found_reply_is_respected(self):
        result = check_citations(NOT_FOUND, SOURCES)
        self.assertFalse(result.found)
        self.assertEqual(result.text, NOT_FOUND)

    def test_sources_are_numbered_with_their_labels(self):
        text = number_sources(SOURCES)
        self.assertTrue(text.startswith("[1] (leave-policy.md › Carry-over)"))
        self.assertIn("[2] (expense-policy.md › Meals)", text)


if __name__ == "__main__":
    unittest.main()
