import unittest

from evaluate_retrieval import calculate_retrieval_metrics


class CalculateRetrievalMetricsTests(unittest.TestCase):
    def test_hit_and_reciprocal_rank_use_first_relevant_result(self):
        hit, reciprocal_rank, rank = calculate_retrieval_metrics(
            ["guide.pdf", "faq.pdf"],
            ["other.pdf", "faq.pdf", "guide.pdf"],
        )

        self.assertTrue(hit)
        self.assertEqual(reciprocal_rank, 0.5)
        self.assertEqual(rank, 2)

    def test_miss_has_zero_reciprocal_rank(self):
        hit, reciprocal_rank, rank = calculate_retrieval_metrics(
            ["guide.pdf"],
            ["other.pdf"],
        )

        self.assertFalse(hit)
        self.assertEqual(reciprocal_rank, 0.0)
        self.assertIsNone(rank)


if __name__ == "__main__":
    unittest.main()