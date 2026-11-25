import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from research.sentiment.granularity_diagnostics import (
    GateData, bootstrap_p, holm, log_loss, permutation_p, resampled_means, within_group_spread,
)


class StatisticsTest(unittest.TestCase):
    def test_log_loss_is_clipped_and_matches_definition(self):
        truth = np.array([1, 0, 1])
        values = log_loss(truth, np.array([0.8, 0.25, 1.0]))
        np.testing.assert_allclose(values[:2], [-np.log(0.8), -np.log(0.75)])
        self.assertTrue(np.isfinite(values[2]) and values[2] >= 0)
        self.assertTrue(np.isfinite(log_loss(np.array([1]), np.array([0.0]))).all())

    def test_resampled_means_weight_each_review(self):
        weights = np.array([[1, 1, 0], [0, 0, 3]], dtype=np.float64)
        np.testing.assert_allclose(resampled_means(weights, np.array([2., 4., 9.])), [3., 9.])

    def test_bootstrap_p_is_two_sided(self):
        self.assertEqual(bootstrap_p([1, 2, 3, 4]), 0.0)
        self.assertEqual(bootstrap_p([-1, 1]), 1.0)
        self.assertAlmostEqual(bootstrap_p([-1, 1, 2, 3]), 0.5)

    def test_holm_is_monotone_step_down(self):
        adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)

    def test_permutation_p_counts_observed_value(self):
        self.assertAlmostEqual(permutation_p(1.0, [2.0, 3.0, 0.5], lower_is_better=True), 2 / 4)
        self.assertAlmostEqual(permutation_p(5.0, [2.0, 3.0, 0.5], lower_is_better=False), 1 / 4)


class ScrambleTest(unittest.TestCase):
    def test_scramble_keeps_counts_within_each_coarse_group(self):
        fine = np.array([0, 1, 1, 2, 3, 3, 3, 0, 2, 4])
        coarse = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 2])
        gate = SimpleNamespace(inv=SimpleNamespace(fine_ids=fine, coarse_ids=coarse))
        scrambled = GateData.scramble(gate, np.random.default_rng(3))
        self.assertEqual(Counter(scrambled.tolist()), Counter(fine.tolist()))
        for group in np.unique(coarse):
            self.assertEqual(Counter(scrambled[coarse == group].tolist()), Counter(fine[coarse == group].tolist()))
        self.assertFalse(np.array_equal(scrambled, fine))

    def test_within_group_spread_ignores_singleton_and_empty_groups(self):
        parent = np.array([0, 0, 1, 2, 2])
        counts = np.array([1, 3, 5, 4, 0])
        gates = np.exp(np.array([0., 4., 9., 1., 7.]))
        # Only group 0 has two observed members: weighted mean 3, variance (9 + 3) / 4.
        self.assertAlmostEqual(within_group_spread(gates, counts, parent), 3.0)
        self.assertEqual(within_group_spread(gates, np.array([1, 0, 5, 4, 0]), parent), 0.0)


if __name__ == "__main__":
    unittest.main()
