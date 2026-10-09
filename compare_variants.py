"""Predeclared small-data factorial experiment, with validation-only selection."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

# Import this before CUDA is initialized: it sets the deterministic cuBLAS environment.
from train_small import (ROOT, SmallDataset, dump_json, load_data, metrics,
                         predict, seed_everything)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from ablation_model import (EPSILON, POSITIVE_KEYS, build_model, swap_examples,
                            training_objective, variant_catalog)
from prepare_small_data import sha256

SEEDS = (42, 43, 44)
SOURCE_FILES = ("ablation_model.py", "compare_variants.py", "train_small.py", "config.yaml",
                "config.small.yaml", "deepsynba.py", "blocks/predictor.py", "blocks/drug_encoder.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def loaders_for(arrays, seed):
    datasets = {name: SmallDataset(arrays, name) for name in ("train", "val", "test")}
    loaders = {name: DataLoader(dataset, batch_size=128, shuffle=name == "train",
                               num_workers=0, pin_memory=True,
                               generator=torch.Generator().manual_seed(seed))
               for name, dataset in datasets.items()}
    return datasets, loaders


@torch.no_grad()
def collect_diagnostics(model, loader, arrays, directory, split):
    model.eval()
    pieces = {name: [] for name in ("observed", "predicted", "swapped_transposed", "bias",
                                    *model.target_keys)}
    for batch in loader:
        *inputs, target = [value.to("cuda", non_blocking=True) for value in batch]
        outputs = model(*inputs)
        swapped = model(inputs[1], inputs[0], inputs[2], inputs[4], inputs[3])
        values = {"observed": target, "predicted": outputs["dose_response"],
                  "swapped_transposed": swapped["dose_response"].transpose(1, 2),
                  "bias": outputs.get("bias", torch.zeros_like(target)),
                  **{key: outputs[key] for key in model.target_keys}}
        for key, value in values.items():
            pieces[key].append(value.cpu().numpy())
    saved = {key: np.concatenate(value) for key, value in pieces.items()}
    rows = loader.dataset.rows.numpy()
    saved.update({key: arrays[key][rows].numpy() for key in ("source_row", "dose1", "dose2")})
    np.savez_compressed(directory / f"{split}_predictions.npz", **saved)
    result = {"all_16": metrics(saved["observed"], saved["predicted"]),
              "combination_9": metrics(saved["observed"][:, 1:, 1:], saved["predicted"][:, 1:, 1:]),
              "swap_rmse": metrics(saved["predicted"], saved["swapped_transposed"])["rmse"],
              "bias_rms": float(np.sqrt(np.mean(saved["bias"].astype(np.float64) ** 2))),
              "parameters": {}}
    for key in POSITIVE_KEYS:
        value = saved[key].ravel()
        result["parameters"][key] = {
            "min": float(value.min()), "max": float(value.max()), "mean": float(value.mean()),
            "zero_fraction": float((value == 0).mean()),
            "below_0_01_fraction": float((value < 0.01).mean()),
        }
    if not all(np.isfinite(value).all() for value in saved.values()):
        raise FloatingPointError("Non-finite prediction or parameter output")
    return result


def make_protocol(manifest, epochs):
    return {
        "started_utc": utc_now(), "seeds": list(SEEDS), "variants": variant_catalog(),
        "dataset_sha256": manifest["dataset_sha256"], "max_epochs": epochs,
        "patience": 8, "batch_size": 128, "gradient_clip": 5.0,
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-5},
        "scheduler": {"name": "LinearLR", "start_factor": 1.0, "end_factor": 0.5, "total_iters": 30},
        "counts": {key: value["examples"] for key, value in manifest["coverage"].items()},
        "positive_package": {"keys": list(POSITIVE_KEYS), "activation": "Softplus + epsilon",
                             "epsilon": EPSILON, "final_weight_multiplier": 0.01,
                             "initial_output_target": 1.0},
        "small_dimensions": {"gene_channel": 128, "emb_size": 512},
        "augmentation": "Independent Bernoulli(0.5) per training example; swap drugs/doses and transpose targets. No inference averaging.",
        "bias_penalty": "MSE(response, target) + 0.1 * mean(bias**2), only in the bias_l2 variant",
        "selection": "Minimum mean validation RMSE across three seeds; tie breaks: SD, parameter count, ID. Minimum validation MSE checkpoint per seed.",
        "diagnostic_example_selection": "Reference seed-44 validation matrix RMSE ranks at 10th, 50th, 90th, 99th percentiles; same examples in every comparison.",
        "uncertainty": "Mean +/- sample SD across three seeds on ONE fixed split, not confidence intervals",
        "test_policy": "Freeze variant ranking and all checkpoint hashes before any test evaluation in this suite. Prior reference test results are already known: exploratory reused test set, not a fresh benchmark.",
        "comparison": "Full 2x2x2 factorial for positive package/size/swap; two conditional bias ablations on the all-three configuration. Not a complete bias factorial.",
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "original_reference": str(ROOT / "runs/extended_100"),
    }


def train_one(output, variant, seed, arrays, manifest, base, epochs):
    folder = output / variant["id"] / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    seed_everything(seed, 8)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    datasets, loaders = loaders_for(arrays, seed)
    model, model_config = build_model(base, variant)
    model = model.to("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
    swap_rng = torch.Generator().manual_seed(seed + 431069)
    print(f"START {variant['id']} seed={seed}, parameters={parameter_count:,}", flush=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    best_loss, stale, best_epoch = float("inf"), 0, None
    fields = ["epoch", "train_rmse", "val_rmse", "lr", "seconds", "bias_penalty_unweighted",
              *[f"{key}_gradient_norm" for key in POSITIVE_KEYS],
              *[f"{key}_zero_gradient_fraction" for key in POSITIVE_KEYS]]
    with (folder / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            total_error, points, total_penalty, batches = 0.0, 0, 0.0, 0
            gradient_sum = torch.zeros(3, device="cuda")
            zero_gradient_sum = torch.zeros(3, device="cuda")
            for batch in loaders["train"]:
                *inputs, target = [value.to("cuda", non_blocking=True) for value in batch]
                if variant["swap"]:
                    mask = torch.rand(len(target), generator=swap_rng) < 0.5
                    inputs, target = swap_examples(inputs, target, mask)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(*inputs)
                objective, mse, penalty = training_objective(outputs, target, variant["bias_l2"])
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"Non-finite loss: {variant['id']} seed {seed} epoch {epoch}")
                objective.backward()
                with torch.no_grad():
                    norms = torch.stack([model.predictors[key].prediction_head[-2].weight.grad.norm()
                                         for key in POSITIVE_KEYS])
                    gradient_sum += norms
                    zero_gradient_sum += (norms == 0).float()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                total_error += mse.item() * target.numel()
                total_penalty += penalty.item() * target.numel()
                points += target.numel()
                batches += 1
            truth, prediction = predict(model, loaders["val"], torch.device("cuda"))
            val_loss = float(np.mean((truth.astype(np.float64) - prediction) ** 2))
            if not np.isfinite(val_loss):
                raise FloatingPointError("Non-finite validation loss")
            row = dict(epoch=epoch, train_rmse=(total_error / points) ** 0.5,
                       val_rmse=val_loss ** 0.5, lr=optimizer.param_groups[0]["lr"],
                       seconds=time.perf_counter() - epoch_started,
                       bias_penalty_unweighted=total_penalty / points)
            for index, key in enumerate(POSITIVE_KEYS):
                row[f"{key}_gradient_norm"] = float(gradient_sum[index] / batches)
                row[f"{key}_zero_gradient_fraction"] = float(zero_gradient_sum[index] / batches)
            writer.writerow(row)
            handle.flush()
            if val_loss < best_loss:
                best_loss, stale, best_epoch = val_loss, 0, epoch
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "model_config": model_config, "variant": variant, "epoch": epoch,
                            "val_mse": val_loss, "dataset_sha256": manifest["dataset_sha256"], "seed": seed},
                           folder / "best.pt")
            else:
                stale += 1
            scheduler.step()
            if epoch % 10 == 0 or stale >= 8 or epoch == epochs:
                print(f"  {variant['id']} s{seed} epoch={epoch} val={row['val_rmse']:.3f}, best={best_loss**0.5:.3f}", flush=True)
            if stale >= 8:
                break
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    validation = collect_diagnostics(model, loaders["val"], arrays, folder, "val")
    np.testing.assert_allclose(validation["all_16"]["rmse"], best_loss ** 0.5, rtol=1e-10, atol=1e-10)
    train_loader = DataLoader(datasets["train"], batch_size=128, shuffle=False, pin_memory=True)
    train_true, train_pred = predict(model, train_loader, torch.device("cuda"))
    report = {"variant": variant, "seed": seed, "best_epoch": best_epoch,
              "epochs_completed": epoch, "stop_reason": "early_stopping" if stale >= 8 else "epoch_limit",
              "parameters": parameter_count, "training_seconds": training_seconds,
              "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
              "dataset_sha256": manifest["dataset_sha256"], "test_evaluated": False,
              "validation": validation, "train_eval": metrics(train_true, train_pred),
              "checkpoint_sha256": sha256(folder / "best.pt"),
              "validation_predictions_sha256": sha256(folder / "val_predictions.npz"),
              "checkpoint": str(folder / "best.pt")}
    if variant["id"] == "reference":
        previous = ROOT / f"runs/extended_100/seed_{seed}"
        original = pd.read_csv(previous / "history.csv")
        current = pd.read_csv(folder / "history.csv")
        columns = ["epoch", "train_rmse", "val_rmse", "lr"]
        np.testing.assert_allclose(current[columns], original[columns].iloc[:len(current)], rtol=0, atol=1e-10)
        report["reference_history_matches_previous"] = True
    dump_json(folder / "metrics.json", report)
    print(f"DONE {variant['id']} seed={seed}: best epoch={best_epoch}, val={validation['all_16']['rmse']:.3f}, {training_seconds:.1f}s", flush=True)
    del optimizer, scheduler, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return report


def freeze_selection(reports):
    groups = {}
    selections = []
    for report in reports:
        name = report["variant"]["id"]
        groups.setdefault(name, []).append(report)
        selections.append({"variant": name, "seed": report["seed"], "epoch": report["best_epoch"],
                           "checkpoint_sha256": report["checkpoint_sha256"],
                           "validation_predictions_sha256": report["validation_predictions_sha256"],
                           "validation_rmse": report["validation"]["all_16"]["rmse"]})
    ranking = []
    for name, rows in groups.items():
        values = [row["validation"]["all_16"]["rmse"] for row in rows]
        ranking.append({"variant": name, "validation_rmse_mean": float(np.mean(values)),
                        "validation_rmse_sd": float(np.std(values, ddof=1)), "parameters": rows[0]["parameters"]})
    ranking.sort(key=lambda row: (row["validation_rmse_mean"], row["validation_rmse_sd"], row["parameters"], row["variant"]))
    winner = ranking[0]["variant"]
    selected_seed = min(groups[winner], key=lambda row: (row["validation"]["all_16"]["rmse"], row["seed"]))["seed"]
    return {"frozen_utc": utc_now(), "selected_variant": winner, "selected_seed_for_examples": selected_seed,
            "ranking": ranking, "checkpoints": selections}


def evaluate_one(output, item, arrays):
    folder = output / item["variant"] / f"seed_{item['seed']}"
    if sha256(folder / "best.pt") != item["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    seed_everything(item["seed"], 8)
    _, loaders = loaders_for(arrays, item["seed"])
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model, _ = build_model(checkpoint["model_config"], checkpoint["variant"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to("cuda").eval()
    # Independent reload check before test inference.
    _, prediction = predict(model, loaders["val"], torch.device("cuda"))
    with np.load(folder / "val_predictions.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(prediction, stored["predicted"])
    test = collect_diagnostics(model, loaders["test"], arrays, folder, "test")
    dump_json(folder / "test_metrics.json", {"evaluated_utc": utc_now(), "test": test,
              "validation_reload_exact": True, "checkpoint_sha256": item["checkpoint_sha256"],
              "test_predictions_sha256": sha256(folder / "test_predictions.npz")})
    print(f"TEST {item['variant']} seed={item['seed']}: RMSE={test['all_16']['rmse']:.3f}", flush=True)
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/comparison_small")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This fixed protocol requires CUDA")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays, _, manifest = load_data(ROOT / "data/small")
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    base["trainer_args"]["data_dir"] = str(ROOT / "data/small")
    protocol = make_protocol(manifest, 100)
    dump_json(output / "protocol.json", protocol)
    dump_json(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": np.__version__, "pandas": pd.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "platform": platform.platform()})
    (output / "requirements-lock.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    # Fail loudly if the source model or optimizer no longer matches the original baseline.
    for name, expected in read_json(ROOT / "runs/extended_100/seed_42/environment.json")["model_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Reference model source changed: {name}")
    if base["optimizer"] != protocol["optimizer"] or base["scheduler"] != protocol["scheduler"]:
        raise ValueError("Original optimizer/scheduler no longer matches the fixed protocol")
    reports = []
    for variant in protocol["variants"]:
        for seed in SEEDS:
            reports.append(train_one(output, variant, seed, arrays, manifest, base, protocol["max_epochs"]))
    if list(output.glob("*/seed_*/test_predictions.npz")):
        raise AssertionError("A test was evaluated before selection was frozen")
    selection = freeze_selection(reports)
    dump_json(output / "selection.json", selection)
    print(f"FROZEN validation-selected variant: {selection['selected_variant']}", flush=True)
    for item in selection["checkpoints"]:
        evaluate_one(output, item, arrays)
    for name, expected in protocol["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise AssertionError(f"Experiment source changed during suite: {name}")
    dump_json(output / "completed.json", {"completed_utc": utc_now(), "status": "complete", "runs": len(reports),
                                          "total_training_seconds": sum(row["training_seconds"] for row in reports)})
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
