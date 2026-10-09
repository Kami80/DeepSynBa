import unittest

import numpy as np
import pandas as pd

from analyze_extended import choose_cases, independent_metrics, seed_summary


class ExtendedAnalysisTests(unittest.TestCase):
    def test_correlations_with_ties_and_known_rmse(self):
        result = independent_metrics([1, 1, 2, 3], [3, 3, 2, 1])
        self.assertAlmostEqual(result["rmse"], np.sqrt(3))
        self.assertAlmostEqual(result["pearson"], -1.0)
        self.assertAlmostEqual(result["spearman"], -1.0)
        self.assertEqual(result["dose_points"], 4)

    def test_report_uses_sample_standard_deviation(self):
        columns = ["validation_rmse", "test_rmse", "test_pearson", "test_spearman", "combination_rmse"]
        table = pd.DataFrame({name: [20, 22, 24] for name in columns})
        result = seed_summary(table)
        self.assertEqual(result["test_rmse"]["mean"], 22)
        self.assertEqual(result["test_rmse"]["sample_sd"], 2)

    def test_diagnostic_cases_follow_declared_error_ranks(self):
        errors = np.arange(100, dtype=float)[::-1]
        cases = choose_cases(errors)
        self.assertEqual([errors[index] for _, index in cases], [10, 50, 89, 98])


if __name__ == "__main__":
    unittest.main()
