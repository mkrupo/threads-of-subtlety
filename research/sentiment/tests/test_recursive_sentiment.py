import unittest

import numpy as np
from scipy.optimize import check_grad

from research.sentiment.recursive_sentiment import build_forest, compose, objective


class RecursiveSentimentTest(unittest.TestCase):
    def setUp(self):
        tree = {"key": "review", "root": 7, "nodes": {
            "1": {}, "2": {}, "3": {}, "4": {},
            "5": {"left": 1, "right": 2, "nuclearity": "SN", "relation": "detail"},
            "6": {"left": 3, "right": 4, "nuclearity": "NN", "relation": "joint"},
            "7": {"left": 5, "right": 6, "nuclearity": "NS", "relation": "contrast"},
        }}
        single = {"key": "single", "root": 1, "nodes": {"1": {}}}
        records = [(tree, [-0.6, 0.7, -0.3, -0.9]), (single, [0.6])]
        fine = {"detail": 0, "joint": 1, "contrast": 2}
        coarse = {"support": 0, "contrast": 1}
        mapping = {"detail": "support", "joint": "support", "contrast": "contrast"}
        self.forest = build_forest(records, fine, coarse, mapping)

    def test_combines_child_scores_by_role_before_composing_the_root(self):
        indices = tuple(layer.parameter_indices("fine") for layer in self.forest.layers)
        weights = np.array([0.9, 0.2, 0.3, 0.1, 0.7, 0.4])
        left = np.tanh(0.2 * -0.6 + 0.9 * 0.7)
        right = np.tanh(0.3 * (-0.3 - 0.9))
        roots = compose(self.forest, indices, weights)[self.forest.roots]
        np.testing.assert_allclose(roots, [np.tanh(0.7 * left + 0.4 * right), 0.6])

    def test_equal_rules_give_identical_predictions_for_all_label_groupings(self):
        roots = {}
        for mode, count in (("shared", 1), ("coarse", 2), ("fine", 3)):
            indices = tuple(layer.parameter_indices(mode) for layer in self.forest.layers)
            roots[mode] = compose(self.forest, indices, np.tile([0.75, 0.25], count))[self.forest.roots]
        np.testing.assert_array_equal(roots["shared"], roots["coarse"])
        np.testing.assert_array_equal(roots["shared"], roots["fine"])

    def test_objective_gradient_through_nested_and_single_edu_reviews(self):
        truth = np.array([0, 1])
        indices = tuple(layer.parameter_indices("coarse") for layer in self.forest.layers)
        prior = np.array([0.0, 1.0, 0.5, 0.5, 0.5, 0.5])

        def evaluate(parameters):
            return objective(parameters, self.forest, indices, truth, prior)

        start = np.array([0.1, 1.3, 0.75, 0.25, 0.6, 0.1])
        error = check_grad(lambda x: evaluate(x)[0], lambda x: evaluate(x)[1], start)
        self.assertLess(error, 1e-6)


if __name__ == "__main__":
    unittest.main()
