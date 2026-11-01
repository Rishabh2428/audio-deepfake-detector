from __future__ import annotations

import unittest

import numpy as np

from app.learning import _evaluate


class EvaluationMetricTests(unittest.TestCase):
    class FixedClassifier:
        def __init__(self, scores):
            self.scores = np.asarray(scores, dtype=float)

        def predict_proba(self, _features):
            return np.column_stack((1.0 - self.scores, self.scores))

    def test_perfect_holdout_has_zero_error_rates(self):
        metrics = _evaluate(
            self.FixedClassifier([0.1, 0.2, 0.8, 0.9]),
            np.zeros((4, 1)),
            np.asarray([0, 0, 1, 1]),
            {},
        )
        self.assertEqual(metrics["holdout_eer"], 0.0)
        self.assertEqual(metrics["holdout_false_accept_rate_at_0_5"], 0.0)
        self.assertEqual(metrics["holdout_false_reject_rate_at_0_5"], 0.0)

    def test_reports_balanced_equal_error_and_cutoff_rates(self):
        metrics = _evaluate(
            self.FixedClassifier([0.6, 0.2, 0.8, 0.4]),
            np.zeros((4, 1)),
            np.asarray([0, 0, 1, 1]),
            {},
        )
        self.assertEqual(metrics["holdout_eer"], 0.5)
        self.assertEqual(metrics["holdout_false_accept_rate_at_0_5"], 0.5)
        self.assertEqual(metrics["holdout_false_reject_rate_at_0_5"], 0.5)


if __name__ == "__main__":
    unittest.main()
