"""Scientific correctness and regression checks for the small-data workflow."""
import unittest
import pickle
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml

from deepsynba import DeepSynBa
from prepare_small_data import ROOT, audit_split, finite_array, sample_split
from train_small import load_data, metrics, SmallDataset


class SplitTests(unittest.TestCase):
    def test_reversed_drug_pair_cannot_cross_splits(self):
        frame = pd.DataFrame({"drug1": ["A", "B "], "drug2": ["B", "A"],
                              "cell": ["X", "Y"], "split": [1, 3]})
        with self.assertRaisesRegex(ValueError, "overlapping pairs"):
            audit_split(frame)

    def test_sampling_is_repeatable_and_retains_original_labels(self):
        frame = pd.DataFrame({"source_row": range(30), "split": np.repeat([1, 2, 3], 10)})
        counts = {"train": 6, "val": 2, "test": 2}
        first = sample_split(frame, counts, seed=42)
        pd.testing.assert_frame_equal(first, sample_split(frame, counts, seed=42))
        for row in first.itertuples():
            self.assertEqual(row.split, frame.loc[row.source_row, "split"])

    def test_missing_data_is_rejected(self):
        with self.assertRaises(ValueError):
            finite_array([float("nan"), 1], (2,), "test")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = yaml.safe_load((ROOT / "config.yaml").read_text())
        config["model_params"].update(gene_channel=8, emb_size=16)
        cls.model = DeepSynBa(config).eval()

    def test_single_example_matches_its_batched_prediction(self):
        inputs = (torch.randn(2, 1, 768), torch.randn(2, 1, 768), torch.randn(2, 976, 1),
                  torch.tensor([[0, 0.1, 1, 10.0]]).repeat(2, 1),
                  torch.tensor([[0, 0.01, 0.1, 1.0]]).repeat(2, 1))
        with torch.no_grad():
            batch = self.model(*inputs)["dose_response"]
            single = self.model(*(x[:1] for x in inputs))["dose_response"]
        self.assertEqual(single.shape, (1, 4, 4))
        torch.testing.assert_close(single, batch[:1], atol=2e-4, rtol=1e-5)

    def test_synba_surface_matches_direct_equation(self):
        x1, x2 = torch.tensor([0.0, 0.1, 1.0]), torch.tensor([0.0, 0.2, 2.0])
        e1, e2, e3 = torch.tensor(0.7), torch.tensor(0.6), torch.tensor(0.1)
        logc1, logc2 = torch.tensor(-1.0), torch.tensor(-0.5)
        h1, h2, alpha = torch.tensor(1.5), torch.tensor(0.8), torch.tensor(2.0)
        actual = self.model.synba_likelihood_2d(x1, x2, e1, e2, e3, logc1, logc2,
                                               h1, h2, torch.tensor(0.0), alpha)
        a = torch.exp(logc1 * h1 + logc2 * h2)
        b = (x1 + 1e-6) ** h1 * torch.exp(logc2 * h2)
        c = torch.exp(logc1 * h1) * (x2 + 1e-6) ** h2
        d = alpha * (x1 + 1e-6) ** h1 * (x2 + 1e-6) ** h2
        expected = 100 * (a + b * e1 + c * e2 + d * e3) / (a + b + c + d)
        torch.testing.assert_close(actual, expected)


class MetricTests(unittest.TestCase):
    def test_pooled_rmse_and_constant_correlations(self):
        result = metrics([0, 2, 4, 6], [2, 2, 2, 2])
        self.assertAlmostEqual(result["rmse"], np.sqrt(6))
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["spearman"])


@unittest.skipUnless((ROOT / "data/small/manifest.json").exists(), "Prepare the small dataset first")
class RealDataTests(unittest.TestCase):
    def test_real_subset_shapes_and_split_integrity(self):
        arrays, metadata, manifest = load_data(ROOT / "data/small")
        self.assertEqual(len(metadata), manifest["selected_examples"])
        self.assertEqual(audit_split(metadata)["pairs_crossing_splits"], 0)
        first = SmallDataset(arrays, "train")[0]
        self.assertEqual([tuple(x.shape) for x in first],
                         [(1, 768), (1, 768), (976, 1), (4,), (4,), (4, 4)])
        self.assertTrue(all(torch.isfinite(x).all() for x in first))

    def test_selected_responses_and_dose_units_match_the_source(self):
        arrays, metadata, _ = load_data(ROOT / "data/small")
        with zipfile.ZipFile(ROOT / "data.zip") as bundle:
            responses = pickle.loads(bundle.read("data/nci_synergy_dose_response.pickle"))
        for split in (1, 2, 3):
            index = metadata.index[metadata.split == split][0]
            row = metadata.loc[index]
            original = responses[f"{row.drug1}_{row.drug2}_{row.cell}"]
            np.testing.assert_array_equal(arrays["response"][index].numpy(),
                                          original["matrix"].astype(np.float32))
            for stored, source in (("dose1", "x1_dose"), ("dose2", "x2_dose")):
                np.testing.assert_array_equal(arrays[stored][index].numpy(),
                                              original[source].astype(np.float32) * 1e6)


if __name__ == "__main__":
    unittest.main()
