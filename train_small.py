"""Train/evaluate the released DeepSynBa model on the prepared small dataset."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

# Required by deterministic CUDA matrix multiplication; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from deepsynba import DeepSynBa
from prepare_small_data import ROOT, SPLITS, audit_split, sha256


class SmallDataset(Dataset):
    def __init__(self, arrays, split):
        self.arrays = arrays
        self.rows = torch.where(arrays["split"] == SPLITS[split])[0]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        a = self.arrays
        return (
            a["drug_features"][a["drug1_index"][row]],
            a["drug_features"][a["drug2_index"][row]],
            a["cell_features"][a["cell_index"][row]],
            a["dose1"][row], a["dose2"][row], a["response"][row],
        )


def load_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, key in (("dataset.npz", "dataset_sha256"), ("metadata.csv", "metadata_sha256")):
        if sha256(directory / name) != manifest[key]:
            raise ValueError(f"{name} differs from the recorded subset; rebuild in a new directory")
    with np.load(directory / "dataset.npz", allow_pickle=False) as saved:
        arrays = {name: torch.from_numpy(saved[name]) for name in saved.files}
    metadata = pd.read_csv(directory / "metadata.csv")
    audit_split(metadata)
    for name in ("source_row", "split"):
        if not np.array_equal(metadata[name], arrays[name].numpy()):
            raise ValueError(f"Metadata and tensors disagree on {name}")
    return arrays, metadata, manifest


def metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64).ravel()
    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite values in evaluation")
    variable = np.ptp(truth) > 0 and np.ptp(prediction) > 0
    return {
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "pearson": float(np.corrcoef(truth, prediction)[0, 1]) if variable else None,
        "spearman": float(spearmanr(truth, prediction).statistic) if variable else None,
        "dose_points": int(truth.size),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    truth, predictions = [], []
    for batch in loader:
        *inputs, target = [tensor.to(device, non_blocking=True) for tensor in batch]
        predicted = model(*inputs)["dose_response"]
        truth.append(target.cpu().numpy())
        predictions.append(predicted.cpu().numpy())
    return np.concatenate(truth), np.concatenate(predictions)


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_predictions(output, split, truth, prediction, dataset, arrays, metadata):
    rows = dataset.rows.numpy()
    np.savez_compressed(output / f"{split}_predictions.npz", observed=truth, predicted=prediction,
                        dose1=arrays["dose1"][rows].numpy(), dose2=arrays["dose2"][rows].numpy(),
                        source_row=arrays["source_row"][rows].numpy())
    metadata.iloc[rows].to_csv(output / f"{split}_metadata.csv", index=False)


def run(args):
    config_path = args.config.resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        settings["max_epochs"] = args.epochs
    if args.seed is not None:
        settings["seed"] = args.seed
    if args.device is not None:
        settings["device"] = args.device
    if args.data_dir is not None:
        settings["data_dir"] = str(args.data_dir.resolve())
    if settings["max_epochs"] < 1 or settings["batch_size"] < 2 or settings["patience"] < 1:
        raise ValueError("Use positive epochs/patience and batch_size >= 2 for BatchNorm")
    seed_everything(settings["seed"], settings["cpu_threads"])
    device_name = settings["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install the CUDA PyTorch build or use --device cpu")
    device = torch.device(device_name)
    data_dir = (config_path.parent / settings["data_dir"]).resolve()
    arrays, metadata, manifest = load_data(data_dir)
    datasets = {split: SmallDataset(arrays, split) for split in SPLITS}
    if len(datasets["train"]) < 2 or min(len(datasets[s]) for s in ("val", "test")) < 1:
        raise ValueError("Need at least two training examples and nonempty validation/test splits")
    batch_size = min(settings["batch_size"], len(datasets["train"]))
    loaders = {
        name: DataLoader(
            dataset, batch_size=batch_size, shuffle=name == "train",
            # Only drop a final singleton batch, which BatchNorm cannot train on.
            drop_last=name == "train" and len(dataset) % batch_size == 1,
            num_workers=settings["num_workers"], pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(settings["seed"]),
        ) for name, dataset in datasets.items()
    }
    base_path = config_path.parent / settings["base_config"]
    model_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    model_config["trainer_args"]["data_dir"] = str(data_dir)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["dataset_sha256"] != manifest["dataset_sha256"]:
            raise ValueError("Checkpoint was trained with a different subset")
        model_config = checkpoint["model_config"]
    model = DeepSynBa(model_config).to(device)
    output = args.output or ROOT / "runs" / datetime.now().strftime("small_%Y%m%d_%H%M%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "PyYAML": yaml.__version__, "device": str(device), "device_name": device_label,
        "cuda_runtime": torch.version.cuda, "platform": platform.platform(),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_source_sha256": {name: sha256(ROOT / name) for name in
                                 ("deepsynba.py", "blocks/predictor.py", "blocks/drug_encoder.py")},
    }
    dump_json(output / "environment.json", environment)
    dump_json(output / "data_manifest.json", manifest)
    installed = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "requirements-lock.txt").write_text(installed, encoding="utf-8")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump({"small_run": settings, "released_model": model_config}, sort_keys=False), encoding="utf-8")
    print(f"Device: {device_label}; model parameters: {environment['model_parameters']:,}", flush=True)
    print(f"Examples: { {name: len(ds) for name, ds in datasets.items()} }; output: {output}", flush=True)
    training_started = time.perf_counter()
    epochs_completed = 0
    if checkpoint is None:
        optimizer_config = dict(model_config["optimizer"])
        optimizer_name = optimizer_config.pop("name")
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_config)
        scheduler_config = dict(model_config["scheduler"])
        scheduler_name = scheduler_config.pop("name")
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_config)
        best_loss, stale = float("inf"), 0
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_rmse", "val_rmse", "lr", "seconds"])
            writer.writeheader()
            for epoch in range(1, settings["max_epochs"] + 1):
                started = time.perf_counter()
                model.train()
                total_squared_error, total_points = 0.0, 0
                for batch in loaders["train"]:
                    *inputs, target = [tensor.to(device, non_blocking=True) for tensor in batch]
                    optimizer.zero_grad(set_to_none=True)
                    predicted = model(*inputs)["dose_response"]
                    loss = torch.nn.functional.mse_loss(predicted, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_norm_clip"],
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    total_squared_error += loss.item() * target.numel()
                    total_points += target.numel()
                val_true, val_pred = predict(model, loaders["val"], device)
                val_loss = float(np.mean((val_true.astype(np.float64) - val_pred) ** 2))
                if not np.isfinite(val_loss):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
                row = {"epoch": epoch, "train_rmse": (total_squared_error / total_points) ** 0.5,
                       "val_rmse": val_loss ** 0.5, "lr": optimizer.param_groups[0]["lr"],
                       "seconds": time.perf_counter() - started}
                writer.writerow(row)
                handle.flush()
                print(f"Epoch {epoch:02d}: train RMSE={row['train_rmse']:.3f}, "
                      f"val RMSE={row['val_rmse']:.3f}, {row['seconds']:.1f}s", flush=True)
                epochs_completed = epoch
                if val_loss < best_loss:
                    best_loss, stale = val_loss, 0
                    torch.save({
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_config": model_config, "epoch": epoch, "val_mse": val_loss,
                        "dataset_sha256": manifest["dataset_sha256"], "seed": settings["seed"],
                    }, output / "best.pt")
                else:
                    stale += 1
                scheduler.step()
                if stale >= settings["patience"]:
                    print(f"Early stopping after {stale} epochs without validation improvement", flush=True)
                    break
        checkpoint_path = output / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        checkpoint_path = args.checkpoint.resolve()
    training_seconds = time.perf_counter() - training_started
    model.load_state_dict(checkpoint["state_dict"])
    # Evaluate training and validation in the same (dropout-disabled) mode.
    train_eval_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=device.type == "cuda")
    train_true, train_pred = predict(model, train_eval_loader, device)
    val_true, val_pred = predict(model, loaders["val"], device)
    save_predictions(output, "val", val_true, val_pred, datasets["val"], arrays, metadata)
    # Fit this simple baseline using training responses only.
    mean_surface = arrays["response"][datasets["train"].rows].numpy().mean(axis=0)
    report = {
        "run_kind": "evaluate_checkpoint" if args.checkpoint else "small_data_reproduction",
        "checkpoint": str(checkpoint_path), "best_epoch": checkpoint["epoch"],
        "epochs_completed": epochs_completed, "best_validation_rmse": checkpoint["val_mse"] ** 0.5,
        "train_examples": len(train_true), "validation_examples": len(val_true),
        "seed": checkpoint["seed"],
        "metrics_definition": "pooled over matrix entries, original response units; no clipping",
        "train_eval_all_16_doses": metrics(train_true, train_pred),
        "validation_all_16_doses": metrics(val_true, val_pred),
        "validation_9_nonzero_combinations": metrics(val_true[:, 1:, 1:], val_pred[:, 1:, 1:]),
        "validation_mean_surface_baseline": metrics(val_true, np.broadcast_to(mean_surface, val_true.shape)),
        "test_evaluated": not args.defer_test,
        "stopping_reason": "checkpoint_evaluation" if args.checkpoint else
                           ("early_stopping" if epochs_completed < settings["max_epochs"] else "epoch_limit"),
        "training_seconds": training_seconds,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None,
        "limitation": "Single seed and reduced data; not a replication of the paper's full-data benchmark",
    }
    if not args.defer_test:
        test_true, test_pred = predict(model, loaders["test"], device)
        baseline = np.broadcast_to(mean_surface, test_true.shape)
        report.update({
            "test_examples": len(test_true),
            "test_all_16_doses": metrics(test_true, test_pred),
            "test_9_nonzero_combinations": metrics(test_true[:, 1:, 1:], test_pred[:, 1:, 1:]),
            "training_mean_surface_baseline": metrics(test_true, baseline),
            "baseline_9_nonzero_combinations": metrics(test_true[:, 1:, 1:], baseline[:, 1:, 1:]),
        })
        save_predictions(output, "test", test_true, test_pred, datasets["test"], arrays, metadata)
    dump_json(output / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.small.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, help="Training seed; does not change the prepared data subset")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", type=Path, help="New directory for results; existing directories are never overwritten")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a saved small-run checkpoint without training")
    parser.add_argument("--defer-test", action="store_true",
                        help="Save training/validation results only; defer test evaluation until models are fixed")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
