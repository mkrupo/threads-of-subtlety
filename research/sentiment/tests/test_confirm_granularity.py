import unittest
from collections import Counter

import numpy as np

from research.sentiment.confirm_granularity import CURVE_SIZES, SESOI, nested_subsets, verdict


class VerdictTest(unittest.TestCase):
    def test_significant_differences_name_the_better_granularity(self):
        self.assertEqual(verdict(-0.003, [-0.005, -0.001], 0.01), "fine better")
        self.assertEqual(verdict(0.003, [0.001, 0.005], 0.01), "coarse better")

    def test_equivalence_needs_the_whole_interval_inside_the_margin(self):
        self.assertEqual(verdict(0.0005, [-0.0015, 0.0019], 0.6), f"equivalent within ±{SESOI}")
        self.assertEqual(verdict(0.0005, [-0.0015, 0.0021], 0.6), "inconclusive")

    def test_significant_differences_inside_the_margin_get_a_combined_verdict(self):
        self.assertEqual(verdict(-0.001, [-0.0018, -0.0002], 0.04), f"fine better, but within ±{SESOI}")
        self.assertEqual(verdict(0.001, [0.0003, 0.0017], 0.01), f"coarse better, but within ±{SESOI}")


class CurveSubsetTest(unittest.TestCase):
    def setUp(self):
        self.metadata = [{"domain": f"d{i % 4}", "label": (i // 4) % 2} for i in range(16000)]
        self.train = np.arange(12800)

    def test_subsets_are_nested_stratified_and_reproducible(self):
        subsets = nested_subsets(self.metadata, self.train, fold=2, draw=1)
        self.assertEqual(sorted(subsets), list(CURVE_SIZES))
        for small, large in zip(CURVE_SIZES, CURVE_SIZES[1:]):
            self.assertTrue(set(subsets[small]) <= set(subsets[large]))
        for size, members in subsets.items():
            self.assertEqual(len(members), size)
            strata = Counter((self.metadata[i]["domain"], self.metadata[i]["label"]) for i in members)
            self.assertEqual(set(strata.values()), {size // 8})
            self.assertTrue(set(members) <= set(self.train))
        again = nested_subsets(self.metadata, self.train, fold=2, draw=1)
        for size in CURVE_SIZES:
            np.testing.assert_array_equal(subsets[size], again[size])

    def test_draws_differ(self):
        first = nested_subsets(self.metadata, self.train, fold=0, draw=0)
        second = nested_subsets(self.metadata, self.train, fold=0, draw=1)
        self.assertFalse(np.array_equal(first[1000], second[1000]))


if __name__ == "__main__":
    unittest.main()
