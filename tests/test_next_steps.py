import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from analyze_next_steps import aggregate
from next_steps_statistics import (cluster_sums, paired_cluster_bootstrap,
                                   resampled_mean_rmse)
from run_next_steps import eligible_holdout, nested_rows, pair_set, validate_file
from prepare_small_data import sha256


class HoldoutTests(unittest.TestCase):
    def test_all_previously_seen_pairs_are_excluded_in_both_orders(self):
        source = pd.DataFrame({"source_row": range(6), "drug1": ["A", "B", "E", "G", "I", "K"],
            "drug2": ["B", "A", "F", "H", "J", "L"], "cell": ["x", "y", "x", "x", "x", "x"],
            "split": [3, 3, 3, 3, 1, 2]})
        previous = source.iloc[[0, 4, 5]].copy()
        actual = eligible_holdout(source, previous)
        self.assertEqual(actual.source_row.tolist(), [2, 3])
        self.assertFalse(pair_set(actual) & pair_set(previous))

    def test_nested_sizes_are_repeatable_and_only_use_training_rows(self):
        frame = pd.DataFrame({"split": [1] * 20 + [2] * 5 + [3] * 5})
        first = nested_rows(frame, (3, 7, 20), seed=5)
        second = nested_rows(frame, (3, 7, 20), seed=5)
        for size in first:
            np.testing.assert_array_equal(first[size], second[size])
            self.assertTrue((frame.loc[first[size], "split"] == 1).all())
        self.assertTrue(set(first[3]) < set(first[7]) < set(first[20]))
        np.testing.assert_array_equal(first[20], np.arange(20))
        with self.assertRaises(ValueError):
            nested_rows(frame, (3, 21))

    def test_frozen_hash_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            path.write_bytes(b"original checkpoint")
            expected = sha256(path)
            validate_file(path, expected)
            path.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "Frozen file changed"):
                validate_file(path, expected)


class SummaryTests(unittest.TestCase):
    def test_summary_uses_sample_sd_and_rejects_missing_groups(self):
        records = pd.DataFrame([{"variant": name, "seed": seed, "parameters": 100,
            "rmse": value, "pearson": .5, "spearman": .4, "combination_rmse": value + 1}
            for name in ("reference", "positive_swap", "small_swap")
            for seed, value in zip((42, 43, 44), (1., 2., 3.))])
        summary = aggregate(records)
        np.testing.assert_array_equal(summary.rmse_mean, [2, 2, 2])
        np.testing.assert_array_equal(summary.rmse_sd, [1, 1, 1])
        with self.assertRaises(ValueError):
            aggregate(records.iloc[:-1])
        with self.assertRaises(ValueError):
            aggregate(records.loc[records.variant != "small_swap"])


class BootstrapTests(unittest.TestCase):
    def test_unequal_clusters_keep_all_cells_and_doses(self):
        truth = np.zeros((3, 2, 2))
        # Cluster 0 has one matrix; cluster 1 has two matrices.
        prediction = np.broadcast_to(np.array([1., 3., 5.])[None, None, :, None, None], (1, 2, 3, 2, 2))
        sums, counts, points = cluster_sums(truth, prediction, [0, 1, 1])
        result = resampled_mean_rmse(sums, counts, points, [[0, 0], [1, 1], [0, 1]])
        np.testing.assert_allclose(result[:, 0], [1, np.sqrt(17), np.sqrt(35 / 3)])

    def test_mean_of_seed_rmses_is_distinct_from_ensemble_rmse(self):
        truth = np.zeros((2, 1, 1))
        prediction = np.array([[[[[1.]], [[1.]]], [[[-3.]], [[-3.]]]]])
        sums, counts, points = cluster_sums(truth, prediction, [0, 1])
        value = resampled_mean_rmse(sums, counts, points, [[0, 1]])
        self.assertEqual(value[0, 0], 2.)

    def test_paired_resampling_preserves_identical_model_differences(self):
        truth = np.zeros((5, 1, 1))
        base = np.arange(1., 6.).reshape(1, 1, 5, 1, 1)
        predictions = np.repeat(base, 3, axis=0)
        result, samples = paired_cluster_bootstrap(truth, predictions, [0, 0, 1, 2, 2],
            ["reference", "positive_swap", "small_swap"], replicates=50, seed=19)
        self.assertEqual(result["clusters"], 3)
        for comparison in result["comparisons"]:
            np.testing.assert_array_equal(comparison["percentile_95_ci"], [0., 0.])
        _, repeated = paired_cluster_bootstrap(truth, predictions, [0, 0, 1, 2, 2],
            ["reference", "positive_swap", "small_swap"], replicates=50, seed=19)
        np.testing.assert_array_equal(samples, repeated)

    def test_non_finite_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            cluster_sums(np.zeros((1, 1, 1)), np.full((3, 1, 1, 1, 1), np.nan), [0])


if __name__ == "__main__":
    unittest.main()
