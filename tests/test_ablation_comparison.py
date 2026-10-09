import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from ablation_model import (EPSILON, POSITIVE_KEYS, PositiveOutput, build_model,
                            swap_examples, training_objective, variant_catalog)
from compare_variants import freeze_selection
from deepsynba import DeepSynBa
from prepare_small_data import ROOT
from analyze_comparison import MEASURES, aggregate_results, factorial_contrasts


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = yaml.safe_load((ROOT / "config.yaml").read_text())
        cls.config["model_params"].update(gene_channel=8, emb_size=16)
        cls.variants = {row["id"]: row for row in variant_catalog()}

    def test_factorial_cells_are_complete_and_unique(self):
        variants = variant_catalog()
        self.assertEqual(len(variants), 10)
        self.assertEqual(len({v["id"] for v in variants}), 10)
        self.assertEqual(len({(v["positive"], v["small"], v["swap"]) for v in variants[:8]}), 8)

    def test_reference_preserves_initial_parameters_and_predictions(self):
        torch.manual_seed(42)
        reference = DeepSynBa(self.config).eval()
        torch.manual_seed(42)
        actual, _ = build_model(self.config, self.variants["reference"])
        actual.eval()
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
        inputs = (torch.randn(2, 1, 768), torch.randn(2, 1, 768), torch.randn(2, 976, 1),
                  torch.tensor([[0, 0.1, 1, 10.0]]).repeat(2, 1),
                  torch.tensor([[0, 0.2, 2, 20.0]]).repeat(2, 1))
        with torch.no_grad():
            torch.testing.assert_close(actual(*inputs)["dose_response"],
                                       reference(*inputs)["dose_response"], rtol=0, atol=0)

    def test_positive_output_has_gradient_for_negative_inputs(self):
        inputs = torch.tensor([-10., -2., 0., 2.], requires_grad=True)
        outputs = PositiveOutput()(inputs)
        self.assertTrue((outputs >= EPSILON).all())
        outputs.sum().backward()
        self.assertTrue((inputs.grad > 0).all())

    def test_positive_initialization_and_other_heads(self):
        model, _ = build_model(self.config, self.variants["positive"])
        for key in POSITIVE_KEYS:
            layers = model.predictors[key].prediction_head
            self.assertIsInstance(layers[-1], PositiveOutput)
            torch.testing.assert_close(layers[-1](layers[-2].bias), torch.ones(1))
        self.assertIsInstance(model.predictors["e1_mean"].prediction_head[-1], nn.Sigmoid)
        self.assertIsInstance(model.predictors["logC1_mean"].prediction_head[-1], nn.Linear)

    def test_variant_reload_and_single_example_inference(self):
        variant = dict(self.variants["positive"], bias=False)
        model, config = build_model(self.config, variant)
        model.eval()
        restored, _ = build_model(config, variant)
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.eval()
        inputs = (torch.randn(2, 1, 768), torch.randn(2, 1, 768), torch.randn(2, 976, 1),
                  torch.tensor([[0, .1, 1., 10.]]).repeat(2, 1),
                  torch.tensor([[0, .2, 2., 20.]]).repeat(2, 1))
        with torch.no_grad():
            outputs = model(*inputs)
            reloaded = restored(*inputs)
            single = restored(*(value[:1] for value in inputs))
        self.assertNotIn("bias", outputs)
        for key in outputs:
            torch.testing.assert_close(outputs[key], reloaded[key], rtol=0, atol=0)
            torch.testing.assert_close(single[key], outputs[key][:1], rtol=1e-5, atol=2e-4)
        self.assertEqual(single["dose_response"].shape, (1, 4, 4))

    def test_smaller_model_parameter_count_and_no_bias(self):
        model, _ = build_model(self.config, self.variants["small"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3499409)
        model, _ = build_model(self.config, self.variants["positive_small_swap_no_bias"])
        self.assertFalse(model.bias)
        self.assertFalse(hasattr(model, "bias_predictor1"))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2903177)

    def test_swapping_changes_both_doses_and_transposes_target_without_mutation(self):
        first = torch.tensor([[[1.]], [[2.]]])
        second = first + 10
        cell = first + 20
        d1, d2 = torch.arange(8).reshape(2, 4), torch.arange(8, 16).reshape(2, 4)
        target = torch.arange(32).reshape(2, 4, 4)
        original = target.clone()
        inputs = (first, second, cell, d1, d2)
        mask = torch.tensor([True, False])
        changed, truth = swap_examples(inputs, target, mask)
        torch.testing.assert_close(changed[0][0], second[0])
        torch.testing.assert_close(changed[0][1], first[1])
        torch.testing.assert_close(changed[3][0], d2[0])
        torch.testing.assert_close(truth[0], target[0].T)
        torch.testing.assert_close(truth[1], target[1])
        torch.testing.assert_close(target, original)
        restored, restored_truth = swap_examples(changed, truth, mask)
        for actual, expected in zip(restored, inputs):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(restored_truth, target)

    def test_bias_penalty_is_separate_from_reported_prediction_error(self):
        outputs = {"dose_response": torch.tensor([1., 3.]), "bias": torch.tensor([2., 4.])}
        loss, mse, penalty = training_objective(outputs, torch.zeros(2), 0.1)
        self.assertEqual(mse.item(), 5)
        self.assertEqual(penalty.item(), 10)
        self.assertEqual(loss.item(), 6)
        plain, plain_mse, _ = training_objective({"dose_response": outputs["dose_response"]}, torch.zeros(2), 0)
        self.assertEqual(plain.item(), plain_mse.item())

    def test_selection_uses_mean_validation_not_best_seed_or_test(self):
        reports = []
        for name, values in (("unstable", [1, 20, 20]), ("stable", [10, 10, 10])):
            for seed, value in zip((42, 43, 44), values):
                reports.append({"variant": {"id": name}, "seed": seed, "best_epoch": 1,
                                "checkpoint_sha256": "hash", "validation_predictions_sha256": "pred",
                                "validation": {"all_16": {"rmse": value}}, "parameters": 10,
                                "test_rmse": 0 if name == "unstable" else 100})
        selected = freeze_selection(reports)
        self.assertEqual(selected["selected_variant"], "stable")
        self.assertEqual(selected["selected_seed_for_examples"], 42)
        self.assertEqual(selected["ranking"][0]["validation_rmse_sd"], 0)

    def test_aggregation_and_factorial_contrasts(self):
        variants = variant_catalog()
        records = []
        for variant in variants:
            for seed, offset in zip((42, 43, 44), (-1., 0., 1.)):
                value = 20 - 2 * variant["positive"] - variant["small"] - .5 * variant["swap"] + offset
                if variant["bias_l2"]:
                    value += .2
                if not variant["bias"]:
                    value += .3
                records.append({"variant": variant["id"], "seed": seed, "parameters": 100,
                                **{key: value for key in MEASURES}})
        table = pd.DataFrame(records)
        summary = aggregate_results(table, variants)
        self.assertEqual(summary.variant.iloc[0], "positive_small_swap")
        self.assertEqual(summary.validation_rmse_sd.iloc[0], 1)
        contrasts, details = factorial_contrasts(table, variants)
        np.testing.assert_allclose([row["validation_rmse"]["mean_delta"] for row in contrasts],
                                   [-2., -1., -.5, .2, .3])
        np.testing.assert_allclose([row["validation_rmse"]["sample_sd"] for row in contrasts], 0, atol=1e-12)
        self.assertEqual(len(details), (4 + 4 + 4 + 1 + 1) * 3 * 2)

    def test_missing_seed_is_not_silently_averaged(self):
        records = [{"variant": "reference", "seed": seed, "parameters": 100,
                    **{key: 10. for key in MEASURES}} for seed in (42, 43)]
        with self.assertRaisesRegex(AssertionError, "three predeclared seeds"):
            aggregate_results(pd.DataFrame(records), [variant_catalog()[0]])


if __name__ == "__main__":
    unittest.main()
