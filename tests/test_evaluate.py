"""Tests for the evaluation scoring (offline, no model calls)."""
import unittest

from docsagent.chunking import Chunk
from docsagent.citations import NOT_FOUND, check_citations
from docsagent.evaluate import contains_facts, load_cases, score_answer, summarize

CARRY = Chunk("leave-policy.md", 2, "You can carry over up to 5 unused days. Use them by 31 March.", "Carry-over")
SICK = Chunk("leave-policy.md", 3, "Employees get 10 days of paid sick leave.", "Sick leave")
CASE = {"question": "How many days carry over?", "answerable": True, "source": "leave-policy.md",
        "section": "Carry-over", "must_include": ["5", "31 march"]}


def state(draft, hits):
    return {"hits": hits, "checked": check_citations(draft, [c for c, _ in hits])}


class EvaluateTest(unittest.TestCase):
    def test_contains_facts_handles_case_commas_and_alternatives(self):
        self.assertTrue(contains_facts("Up to ₹15,000 in the first 3 months", ["15000"]))
        self.assertTrue(contains_facts("Alcohol is not reimbursed", ["never|not"]))
        self.assertFalse(contains_facts("Up to 5 days", ["5", "31 march"]))

    def test_correct_cited_answer_passes(self):
        result = score_answer(CASE, state("Up to 5 days, used by 31 March [1].", [(CARRY, 0.8), (SICK, 0.4)]))
        self.assertTrue(result.retrieved and result.answered and result.cited and result.passed)

    def test_citing_the_wrong_section_fails_the_citation_check(self):
        result = score_answer(CASE, state("Unused days carry over to next year [1].", [(SICK, 0.8), (CARRY, 0.7)]))
        self.assertTrue(result.retrieved)   # the right section was found...
        self.assertFalse(result.cited)      # ...but the answer cites the sick-leave section
        self.assertFalse(result.passed)

    def test_unanswerable_question_must_be_refused(self):
        case = {"question": "Gym membership?", "answerable": False}
        self.assertTrue(score_answer(case, state(NOT_FOUND, [(SICK, 0.3)])).refused)
        self.assertFalse(score_answer(case, state("Yes, gyms are paid [1].", [(SICK, 0.3)])).refused)

    def test_summary_counts_only_measured_checks(self):
        refused = score_answer({"question": "x", "answerable": False}, state(NOT_FOUND, []))
        answered = score_answer(CASE, state("Up to 5 days, used by 31 March [1].", [(CARRY, 0.8)]))
        summary = summarize([refused, answered])
        self.assertEqual(summary["refusal"], (1, 1))
        self.assertEqual(summary["answer"], (1, 1))
        self.assertEqual(summary["overall"], (2, 2))

    def test_eval_set_is_well_formed(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 15)
        for case in cases:
            if case["answerable"]:
                self.assertTrue(case["source"] and case["section"] and case["must_include"])


if __name__ == "__main__":
    unittest.main()
