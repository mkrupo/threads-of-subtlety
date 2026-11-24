import unittest

import numpy as np
from scipy.optimize import check_grad
from sklearn.preprocessing import StandardScaler

from research.sentiment.dependency_sentiment import dependency_edges, group_pairs
from research.sentiment.relation_gate import objective


class RelationGateTest(unittest.TestCase):
    def test_dependency_projection_follows_nuclearity_and_keeps_edu_order(self):
        tree = {"root": 7, "key": "scene", "nodes": {
            "1": {}, "2": {}, "3": {}, "4": {},
            "5": {"left": 1, "right": 2, "nuclearity": "NN", "relation": "Joint"},
            "6": {"left": 3, "right": 4, "nuclearity": "NS", "relation": "Elaboration"},
            "7": {"left": 5, "right": 6, "nuclearity": "SN", "relation": "Concession"},
        }}
        self.assertEqual(dependency_edges(tree), [(1, 0, "joint"), (3, 2, "elaboration"), (0, 2, "concession")])
        self.assertEqual(dependency_edges({"root": 1, "key": "single", "nodes": {"1": {}}}), [])

    def test_pooled_edges_use_frequency_and_gate_within_each_review(self):
        pairs = group_pairs(np.array([[2.], [4.], [7.], [8.]]),
                            np.array([0, 0, 0, 2]), np.array([0, 0, 1, 1]), 3, 2)
        gates = np.array([2., 3.])
        pooled, totals = pairs.weighted_mean(gates)
        np.testing.assert_allclose(pooled[:, 0], [33 / 7, 0, 8])
        np.testing.assert_allclose(totals, [7, 0, 3])
        selection = np.array([2, 1, 0])
        np.testing.assert_allclose(pairs.subset(selection).weighted_mean(gates)[0], pooled[selection])

    def test_objective_gradient(self):
        rng = np.random.default_rng(7)
        base = rng.normal(size=(5, 2))
        truth = np.array([0, 0, 1, 1, 1])
        pairs = group_pairs(rng.normal(size=(12, 3)), np.repeat([0, 1, 2, 4], 3),
                            np.tile([0, 1, 1], 4), len(truth), 2)
        scaler = StandardScaler().fit(pairs.weighted_mean(np.ones(2))[0])
        for use_labels in (False, True):
            start = rng.normal(size=1 + 2 + 3 + 2 * use_labels)

            def calculate(parameters):
                return objective(parameters, base, pairs, scaler, truth, 1.0, use_labels)

            error = check_grad(lambda x: calculate(x)[0], lambda x: calculate(x)[1], start)
            self.assertLess(error, 1e-5)


if __name__ == "__main__":
    unittest.main()
