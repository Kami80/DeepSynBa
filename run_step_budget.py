"""Validation-only comparison with a fixed optimizer-update budget.

The inspected holdouts are never loaded. Each fit starts from its original seed
on the exact earlier nested subset; only the budget/schedule timing changes.
"""
import argparse
import csv
from dataclasses import dataclass
import gc
from pathlib import Path
import platform
import time

# Configure deterministic cuBLAS before importing torch.
from train_small import ROOT, dump_json, load_data, metrics, predict, seed_everything
import numpy as np
import pandas as pd
import torch
import yaml

from ablation_model import build_model, swap_examples, variant_catalog
from compare_variants import utc_now
from prepare_small_data import sha256
from run_next_steps import make_loaders, read_json, validate_file

VARIANTS = ("positive_swap", "small_swap")
SIZES = (1000, 2000, 3000)
SEEDS = (42, 43, 44)
SOURCE_FILES = ("run_step_budget.py", "run_next_steps.py", "ablation_model.py",
                "train_small.py", "prepare_small_data.py", "compare_variants.py",
                "deepsynba.py", "blocks/predictor.py", "blocks/drug_encoder.py",
                "config.yaml")


@dataclass(frozen=True)
class UpdateBudget:
    max_updates: int = 2400
    validation_interval: int = 24
    patience_evaluations: int = 8

    def __post_init__(self):
        if min(self.max_updates, self.validation_interval, self.patience_evaluations) < 1:
            raise ValueError("Update budget, validation interval and patience must be positive")
        if self.max_updates % self.validation_interval:
            raise ValueError("The budget must end at a scheduled validation check")


def fit_with_updates(model, optimizer, scheduler, train_loader, val_loader, *,
                     budget, on_validation, device, gradient_clip=5.0, augment=None):
    """Count actual optimizer steps, including checks in the middle of an epoch.

    on_validation receives the current model before the next training step.
    The scheduler advances after validation, exactly as in the old 3,000-row fit.
    """
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Both training and validation loaders must be nonempty")
    best_loss, best_update, best_examples = float("inf"), None, None
    stale, update, examples, epoch = 0, 0, 0, 0
    block_error, block_points, block_started = 0.0, 0, time.perf_counter()
    history = []
    while update < budget.max_updates and stale < budget.patience_evaluations:
        epoch += 1
        for batch_index, batch in enumerate(train_loader, start=1):
            # predict() sets eval mode. Restore train mode even after a mid-epoch check.
            model.train()
            *inputs, target = [value.to(device, non_blocking=True) for value in batch]
            if augment is not None:
                inputs, target = augment(inputs, target)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(*inputs)["dose_response"]
            loss = torch.nn.functional.mse_loss(predicted, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            update += 1
            examples += len(target)
            block_error += loss.item() * target.numel()
            block_points += target.numel()
            if update % budget.validation_interval == 0:
                observed, predicted = predict(model, val_loader, torch.device(device))
                residual = observed.astype(np.float64) - predicted
                val_loss = float(np.mean(residual ** 2))
                if not np.isfinite(val_loss):
                    raise FloatingPointError("Non-finite validation loss")
                improved = val_loss < best_loss
                if improved:
                    best_loss, best_update, best_examples, stale = val_loss, update, examples, 0
                else:
                    stale += 1
                row = {"update": update, "epoch": epoch, "batch_in_epoch": batch_index,
                    "epoch_equivalent": update / len(train_loader), "examples_seen": examples,
                    "train_rmse": (block_error / block_points) ** .5,
                    "val_rmse": val_loss ** .5,
                    "val_combination_rmse": float(np.sqrt(np.mean(residual[:, 1:, 1:] ** 2))),
                    "lr": optimizer.param_groups[0]["lr"], "stale_evaluations": stale,
                    "improved": improved, "seconds": time.perf_counter() - block_started}
                history.append(row)
                on_validation(row, model)
                scheduler.step()
                block_error, block_points, block_started = 0.0, 0, time.perf_counter()
            if update >= budget.max_updates or stale >= budget.patience_evaluations:
                break
    return {"history": history, "updates_completed": update, "best_update": best_update,
            "examples_seen": examples, "best_examples_seen": best_examples,
            "epochs_started": epoch, "epoch_equivalent": update / len(train_loader),
            "best_validation_mse": best_loss,
            "stop_reason": "early_stopping" if stale >= budget.patience_evaluations else "update_limit"}


def prepare(output, baseline):
    if output.exists():
        raise FileExistsError("Preserve prior experiments: choose a new output directory")
    baseline = baseline.resolve()
    validate_file(baseline / "protocol.json",
                  (baseline / "protocol.sha256").read_text(encoding="utf-8").strip())
    old = read_json(baseline / "protocol.json")
    for name, expected in old["source_sha256"].items():
        validate_file(ROOT / name, expected)
    arrays, metadata, manifest = load_data(ROOT / "data/small")
    if (manifest["dataset_sha256"], manifest["metadata_sha256"]) != (
            old["dataset_sha256"], old["dataset_metadata_sha256"]):
        raise ValueError("The training/validation dataset differs from the prior experiment")
    selection = read_json(baseline / "learning/selection.json")
    baselines = []
    for size in SIZES:
        for name in VARIANTS:
            for seed in SEEDS:
                record = next(row for row in selection["runs"] if
                    (row["training_matrices"], row["variant"], row["seed"]) == (size, name, seed))
                folder = Path(record["folder"])
                frozen = dict(record)
                if "history_sha256" not in frozen:
                    frozen["history_sha256"] = next(row["history_sha256"]
                        for row in old["frozen_checkpoints"] if (row["variant"], row["seed"]) == (name, seed))
                for filename, key in (("best.pt", "checkpoint_sha256"),
                        ("val_predictions.npz", "validation_predictions_sha256"),
                        ("history.csv", "history_sha256")):
                    validate_file(folder / filename, frozen[key])
                baselines.append(frozen)
    subsets = {str(size): old["nested_subsets"][str(size)] for size in SIZES}
    for size, subset in subsets.items():
        rows = subset["array_rows"]
        if len(rows) != int(size) or len(set(rows)) != int(size):
            raise ValueError("Invalid subset size or duplicate training matrices")
        np.testing.assert_array_equal(arrays["source_row"][rows].numpy(), subset["source_rows"])
        if not (metadata.iloc[rows].split == 1).all():
            raise ValueError("A nested training subset includes non-training rows")
    for smaller, larger in zip(SIZES, SIZES[1:]):
        if not set(subsets[str(smaller)]["array_rows"]) < set(subsets[str(larger)]["array_rows"]):
            raise ValueError("The previous subsets are not nested")
    budget = UpdateBudget()
    variants = {item["id"]: item for item in variant_catalog()}
    protocol = {"created_utc": utc_now(), "experiment": "Equal maximum optimizer-update budget",
        "baseline_experiment": str(baseline), "baseline_protocol_sha256": sha256(baseline / "protocol.json"),
        "baseline_selection_sha256": sha256(baseline / "learning/selection.json"),
        "dataset_sha256": manifest["dataset_sha256"], "metadata_sha256": manifest["metadata_sha256"],
        "training_sizes": list(SIZES), "seeds": list(SEEDS),
        "variants": [variants[name] for name in VARIANTS], "nested_subsets": subsets,
        "baseline_runs": baselines, "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "hyperparameters": {"max_updates": budget.max_updates, "validation_interval": budget.validation_interval,
            "patience_evaluations": budget.patience_evaluations, "patience_updates": 192,
            "batch_size": 128, "grad_clip": 5.0, "cpu_threads": 8,
            "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-5},
            "scheduler": {"name": "LinearLR", "start_factor": 1.0, "end_factor": .5, "total_iters": 30},
            "scheduler_step_every_updates": 24, "scheduler_decay_finished_at_update": 720},
        "primary_endpoint": "Per size/model paired-seed difference in validation all-16 RMSE: update-budget minus earlier epoch-budget. Negative favors update-budget.",
        "secondary_endpoints": "Validation combination-9 RMSE, Pearson, Spearman; actual updates, examples, training seconds, peak allocated CUDA MiB and parameter count.",
        "selection_policy": "Each seed checkpoint minimizes all-16 validation MSE at updates 24,48,...,2400. Strict improvement; stop after eight failed checks (192 updates). Rank the six configurations by mean validation RMSE over all three seeds; ties by SD, parameters, data size, variant ID. Never choose a seed by accuracy.",
        "evaluation_policy": "Validation-only adaptive budget study. Do not load, predict, select on, or reuse either inspected holdout. No new test claim or generalization interval.",
        "uncertainty": "Report mean and sample SD of three fixed seed fits on one fixed nested data split. No confidence intervals, significance tests, ensemble averaging or split variability estimates.",
        "comparability": "Same initial seeds, nested source rows, architecture, augmentation, float32 AdamW and batches. Schedule, validation and patience share update units. The 3000-matrix setting is unchanged; require all six old trajectories/predictions to reproduce. At 1000/2000 matrices the intervention bundles a larger budget with slower LR decay and stopping/evaluation timing.",
        "compute_policy": "Maximum budget is equal, realized updates can differ because of early stopping. Training seconds include validation/checkpoint I/O, exclude model creation and final reload. Prior/new wall times come from separate sessions and are descriptive, not a controlled timing benchmark.",
        "training_order_sizes": [3000, 1000, 2000],
        "training_order": "3000 first to verify the unchanged controls, then 1000,2000 matrices; positive_swap,small_swap; seeds 42,43,44. Every fit retrained from scratch.",
        "units": "Response in original units (untreated reference 100); concentrations micromolar.",
        "augmentation": "Drug/dose swap plus target transpose, Bernoulli(0.5) using seed+431069; no inference averaging."}
    output.mkdir(parents=True)
    dump_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(sha256(output / "protocol.json"), encoding="utf-8")
    dump_json(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "pandas": pd.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "platform": platform.platform(), "deterministic": True})
    print("FROZEN: 18 new fits; 2400 updates maximum; validation every 24; patience 192 updates", flush=True)


def load_protocol(output):
    validate_file(output / "protocol.json", (output / "protocol.sha256").read_text(encoding="utf-8").strip())
    protocol = read_json(output / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        validate_file(ROOT / name, expected)
    baseline = Path(protocol["baseline_experiment"])
    validate_file(baseline / "protocol.json", protocol["baseline_protocol_sha256"])
    validate_file(baseline / "learning/selection.json", protocol["baseline_selection_sha256"])
    arrays, metadata, manifest = load_data(ROOT / "data/small")
    if (manifest["dataset_sha256"], manifest["metadata_sha256"]) != (
            protocol["dataset_sha256"], protocol["metadata_sha256"]):
        raise ValueError("Frozen training/validation dataset changed")
    return protocol, arrays, metadata


def verify_run_files(record):
    folder = Path(record["folder"])
    for filename, key in (("best.pt", "checkpoint_sha256"),
            ("val_predictions.npz", "validation_predictions_sha256"), ("history.csv", "history_sha256")):
        validate_file(folder / filename, record[key])


def train_one(output, variant, seed, size, arrays, protocol):
    folder = output / "training" / f"n_{size}" / variant["id"] / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    seed_everything(seed, 8)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    settings = protocol["hyperparameters"]
    budget = UpdateBudget(settings["max_updates"], settings["validation_interval"], settings["patience_evaluations"])
    subset = protocol["nested_subsets"][str(size)]
    loaders = make_loaders(arrays, seed, subset["array_rows"])
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    base["trainer_args"]["data_dir"] = str(ROOT / "data/small")
    model, config = build_model(base, variant)
    model.to("cuda")
    optimizer_args, scheduler_args = dict(settings["optimizer"]), dict(settings["scheduler"])
    optimizer_name, scheduler_name = optimizer_args.pop("name"), scheduler_args.pop("name")
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_args)
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_args)
    swap_rng = torch.Generator().manual_seed(seed + 431069)

    def augment(inputs, target):
        return swap_examples(inputs, target, torch.rand(len(target), generator=swap_rng) < .5)

    fields = ["update", "epoch", "batch_in_epoch", "epoch_equivalent", "examples_seen", "train_rmse",
              "val_rmse", "val_combination_rmse", "lr", "stale_evaluations", "improved", "seconds"]
    print(f"TRAIN n={size} {variant['id']} seed={seed}; {len(loaders['train'])} batches/epoch", flush=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (folder / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def on_validation(row, current_model):
            writer.writerow(row)
            handle.flush()
            if row["improved"]:
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in current_model.state_dict().items()},
                    "model_config": config, "variant": variant, "best_update": row["update"],
                    "epoch": row["epoch"], "batch_in_epoch": row["batch_in_epoch"],
                    "val_mse": row["val_rmse"] ** 2, "dataset_sha256": protocol["dataset_sha256"],
                    "seed": seed, "training_source_rows": subset["source_rows"],
                    "protocol_sha256": sha256(output / "protocol.json")}, folder / "best.pt")
            if row["update"] % 600 == 0 or row["stale_evaluations"] >= budget.patience_evaluations:
                print(f"  n={size} {variant['id']} s{seed} update={row['update']} val={row['val_rmse']:.3f}", flush=True)

        fitted = fit_with_updates(model, optimizer, scheduler, loaders["train"], loaders["val"],
            budget=budget, on_validation=on_validation, device="cuda", gradient_clip=settings["grad_clip"],
            augment=augment if variant["swap"] else None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    observed, predicted = predict(model, loaders["val"], torch.device("cuda"))
    validation = {"all_16": metrics(observed, predicted),
                  "combination_9": metrics(observed[:, 1:, 1:], predicted[:, 1:, 1:])}
    np.testing.assert_allclose(validation["all_16"]["rmse"], fitted["best_validation_mse"] ** .5, rtol=1e-10)
    val_rows = loaders["val"].dataset.rows
    np.savez_compressed(folder / "val_predictions.npz", observed=observed, predicted=predicted,
        source_row=arrays["source_row"][val_rows].numpy(),
        dose1=arrays["dose1"][val_rows].numpy(), dose2=arrays["dose2"][val_rows].numpy())
    baseline = next(item for item in protocol["baseline_runs"] if
        (item["training_matrices"], item["variant"], item["seed"]) == (size, variant["id"], seed))
    control_reproduced = None
    if size == 3000:
        verify_run_files(baseline)
        prior = pd.read_csv(Path(baseline["folder"]) / "history.csv")
        current = pd.read_csv(folder / "history.csv")
        if len(prior) != len(current):
            raise AssertionError("The unchanged 3000-row stopping trajectory did not reproduce")
        np.testing.assert_array_equal(current["update"], prior["epoch"] * 24)
        np.testing.assert_allclose(current[["train_rmse", "val_rmse", "lr"]],
                                   prior[["train_rmse", "val_rmse", "lr"]], rtol=0, atol=1e-10)
        with np.load(Path(baseline["folder"]) / "val_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["predicted"], predicted)
        control_reproduced = True
    report = {key: value for key, value in fitted.items() if key != "history"}
    report.update({"variant": variant["id"], "seed": seed, "training_matrices": size,
        "parameters": sum(value.numel() for value in model.parameters()), "batches_per_epoch": len(loaders["train"]),
        "maximum_updates": budget.max_updates, "validation": validation, "training_seconds": seconds,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "control_3000_exactly_reproduced": control_reproduced,
        "checkpoint_sha256": sha256(folder / "best.pt"),
        "validation_predictions_sha256": sha256(folder / "val_predictions.npz"),
        "history_sha256": sha256(folder / "history.csv"), "folder": str(folder),
        "dataset_sha256": protocol["dataset_sha256"], "protocol_sha256": sha256(output / "protocol.json")})
    dump_json(folder / "metrics.json", report)
    print(f"DONE n={size} {variant['id']} seed={seed}: val={validation['all_16']['rmse']:.3f}, "
          f"updates={fitted['updates_completed']}, best={fitted['best_update']}, {seconds:.1f}s", flush=True)
    del optimizer, scheduler, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return report


def train(output):
    protocol, arrays, _ = load_protocol(output)
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen experiment requires the existing CUDA environment")
    if (output / "completed.json").exists():
        raise FileExistsError("This experiment is already complete")
    reports = []
    for size in protocol["training_order_sizes"]:
        for variant in protocol["variants"]:
            for seed in protocol["seeds"]:
                folder = output / "training" / f"n_{size}" / variant["id"] / f"seed_{seed}"
                if (folder / "metrics.json").exists():
                    report = read_json(folder / "metrics.json")
                    verify_run_files(report)
                    if report["protocol_sha256"] != sha256(output / "protocol.json"):
                        raise ValueError("Completed job has a different protocol")
                    print(f"REUSE completed job n={size} {variant['id']} s{seed}", flush=True)
                else:
                    report = train_one(output, variant, seed, size, arrays, protocol)
                reports.append(report)
    ranking = []
    for size in SIZES:
        for name in VARIANTS:
            group = [row for row in reports if (row["training_matrices"], row["variant"]) == (size, name)]
            values = [row["validation"]["all_16"]["rmse"] for row in group]
            ranking.append({"variant": name, "training_matrices": size,
                "validation_rmse_mean": float(np.mean(values)), "validation_rmse_sd": float(np.std(values, ddof=1)),
                "parameters": group[0]["parameters"]})
    ranking.sort(key=lambda row: (row["validation_rmse_mean"], row["validation_rmse_sd"],
                                 row["parameters"], row["training_matrices"], row["variant"]))
    load_protocol(output)
    dump_json(output / "selection.json", {"frozen_utc": utc_now(), "ranking": ranking,
        "selected_configuration": ranking[0], "runs": reports})
    dump_json(output / "completed.json", {"completed_utc": utc_now(), "new_training_jobs": len(reports),
        "new_training_seconds": sum(row["training_seconds"] for row in reports),
        "total_optimizer_updates": sum(row["updates_completed"] for row in reports),
        "selection_sha256": sha256(output / "selection.json"), "runs": reports})
    print(f"COMPLETE all {len(reports)} new fits; validation winner={ranking[0]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/step_budget_20261001")
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/next_steps_20261001")
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage in ("prepare", "all"):
        prepare(output, args.baseline)
    if args.stage in ("train", "all"):
        train(output)


if __name__ == "__main__":
    main()
