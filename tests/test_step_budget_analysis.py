import unittest

import numpy as np
import pandas as pd

from analyze_step_budget import MEASURES, paired_changes, summarize
from run_step_budget import SEEDS, SIZES, VARIANTS


def fixture():
    rows = []
    for method in ("epoch_budget", "update_budget"):
        for size in SIZES:
            for name in VARIANTS:
                for seed, base in zip(SEEDS, (1., 2., 3.)):
                    value = base - .5 if method == "update_budget" else base
                    row = {field: value for field in MEASURES}
                    row.update(method=method, training_matrices=size, variant=name, seed=seed,
                               parameters=100, stop_reason="update_limit" if method == "update_budget" else "epoch_limit")
                    rows.append(row)
    return pd.DataFrame(rows)


class BudgetAnalysisTests(unittest.TestCase):
    def test_fixed_seed_pairing_uses_paired_difference_sd(self):
        table = fixture().sample(frac=1., random_state=17)
        paired, summary = paired_changes(table)
        self.assertEqual(len(paired), 18)
        np.testing.assert_array_equal(paired.rmse_change, -.5)
        np.testing.assert_array_equal(summary.rmse_change_mean, -.5)
        # Individual seed RMSE has SD=1; identical matched changes have SD=0.
        np.testing.assert_array_equal(summary.rmse_change_sd, 0.)
        aggregated = summarize(table)
        np.testing.assert_array_equal(aggregated.rmse_sd, 1.)
        np.testing.assert_array_equal(aggregated.limit_reached, 3)

    def test_missing_and_duplicate_seeds_are_not_silently_averaged(self):
        table = fixture()
        for corrupt in (table.iloc[:-1], pd.concat([table, table.iloc[:1]], ignore_index=True)):
            with self.assertRaises(ValueError):
                summarize(corrupt)
            with self.assertRaises((ValueError, pd.errors.MergeError)):
                paired_changes(corrupt)


if __name__ == "__main__":
    unittest.main()
