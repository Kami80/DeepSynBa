"""Boundary/behavior checks for training measured in optimizer updates."""
import unittest

from train_small import metrics  # deterministic CUDA environment before torch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from run_step_budget import UpdateBudget, fit_with_updates


class TinyResponseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 16)
        self.train_flags = []

    def forward(self, value):
        self.train_flags.append(self.training)
        return {"dose_response": self.linear(value).reshape(-1, 4, 4)}


def tiny_fit(batches, budget, lr=.02, on_validation=None):
    torch.manual_seed(5)
    model = TinyResponseModel()
    inputs = torch.ones((batches * 2, 1))
    targets = torch.zeros((batches * 2, 4, 4))
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    val_loader = DataLoader(TensorDataset(inputs[:2], targets[:2]), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=.5, total_iters=3)
    rows = []

    def record(row, current_model):
        rows.append(dict(row))
        if on_validation is not None:
            on_validation(row, current_model)

    fitted = fit_with_updates(model, optimizer, scheduler, train_loader, val_loader,
        budget=budget, on_validation=record, device="cpu")
    return fitted, rows, model


class StepBudgetTests(unittest.TestCase):
    def test_checks_cross_epoch_boundaries_and_restore_training_mode(self):
        fitted, rows, model = tiny_fit(3, UpdateBudget(8, 2, 20))
        self.assertEqual(fitted["updates_completed"], 8)
        self.assertEqual(fitted["examples_seen"], 16)
        self.assertEqual([(row["update"], row["epoch"], row["batch_in_epoch"]) for row in rows],
                         [(2, 1, 2), (4, 2, 1), (6, 2, 3), (8, 3, 2)])
        self.assertEqual(model.train_flags, [True, True, False] * 4)
        self.assertEqual(fitted["stop_reason"], "update_limit")

    def test_schedule_and_patience_depend_on_updates_for_different_epoch_sizes(self):
        first, a, _ = tiny_fit(3, UpdateBudget(12, 2, 2), lr=0.)
        second, b, _ = tiny_fit(5, UpdateBudget(12, 2, 2), lr=0.)
        for fitted, rows in ((first, a), (second, b)):
            self.assertEqual(fitted["best_update"], 2)
            self.assertEqual(fitted["updates_completed"], 6)
            self.assertEqual(fitted["updates_completed"] - fitted["best_update"], 4)
            self.assertEqual(fitted["stop_reason"], "early_stopping")
            self.assertEqual([row["stale_evaluations"] for row in rows], [0, 1, 2])
        self.assertNotEqual(first["epoch_equivalent"], second["epoch_equivalent"])
        _, lr_a, _ = tiny_fit(3, UpdateBudget(10, 2, 20))
        _, lr_b, _ = tiny_fit(5, UpdateBudget(10, 2, 20))
        np.testing.assert_allclose([row["lr"] for row in lr_a], [.02, .02 * 5/6, .02 * 2/3, .01, .01])
        np.testing.assert_array_equal([row["lr"] for row in lr_a], [row["lr"] for row in lr_b])

    def test_callback_observes_checkpoint_that_matches_recorded_validation(self):
        def check(row, model):
            with torch.no_grad():
                prediction = model(torch.ones((2, 1)))["dose_response"].numpy()
            actual = metrics(np.zeros((2, 4, 4)), prediction)["rmse"]
            self.assertAlmostEqual(row["val_rmse"], actual, places=12)
        fitted, rows, _ = tiny_fit(3, UpdateBudget(8, 2, 20), on_validation=check)
        best = min(rows, key=lambda row: row["val_rmse"])
        self.assertEqual(fitted["best_update"], best["update"])
        self.assertAlmostEqual(fitted["best_validation_mse"] ** .5, best["val_rmse"], places=12)

    def test_production_lr_decay_preserves_old_3000_matrix_schedule(self):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=.5, total_iters=30)
        learning_rates = []
        for update in range(1, 2401):
            learning_rates.append(optimizer.param_groups[0]["lr"])
            parameter.grad = torch.tensor(0.)
            optimizer.step()
            if update % 24 == 0:
                scheduler.step()
        self.assertEqual(learning_rates[0], 1e-4)
        self.assertEqual(learning_rates[23], 1e-4)
        self.assertAlmostEqual(learning_rates[24], 1e-4 * (1 - .5/30), places=14)
        np.testing.assert_allclose(learning_rates[720:], 5e-5, rtol=0, atol=1e-18)

    def test_invalid_or_incomplete_budget_is_rejected(self):
        for values in ((0, 24, 8), (2400, 0, 8), (2400, 24, 0), (25, 24, 8)):
            with self.assertRaises(ValueError):
                UpdateBudget(*values)


if __name__ == "__main__":
    unittest.main()
