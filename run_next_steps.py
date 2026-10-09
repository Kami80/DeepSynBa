"""Frozen same-source confirmation and nested small-data learning curves.

The previous project is read-only. New protocols, data and results live here.
Run --stage prepare first; confirm and learning use that immutable protocol.
"""
import argparse
import csv
import gc
import json
import pickle
from pathlib import Path
import platform
import sys
import time
import zipfile

# This import sets deterministic CUDA settings before importing torch.
from train_small import (ROOT, SmallDataset, dump_json, load_data, metrics,
                         predict, seed_everything)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from ablation_model import build_model, swap_examples, variant_catalog
from compare_variants import utc_now
from prepare_small_data import (ARCHIVE_SHA256, audit_split, finite_array,
                                sha256, sorted_pair_columns)

FINALISTS = ("reference", "positive_swap", "small_swap")
SEEDS = (42, 43, 44)
SIZES = (500, 1000, 2000, 3000)
SOURCE_FILES = ("run_next_steps.py", "ablation_model.py", "train_small.py",
                "prepare_small_data.py", "compare_variants.py", "deepsynba.py",
                "blocks/predictor.py", "blocks/drug_encoder.py", "config.yaml")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pair_set(frame):
    return {tuple(pair) for pair in sorted_pair_columns(frame)}


def eligible_holdout(source, previous):
    """Exclude entire unordered pairs from ALL previously examined partitions."""
    audit_split(source)
    audit_split(previous)
    seen = pair_set(previous)
    mask = np.array([tuple(pair) not in seen for pair in sorted_pair_columns(source)])
    result = source.loc[(source.split == 3) & mask].copy()
    if pair_set(result) & seen or set(result.source_row) & set(previous.source_row):
        raise ValueError("Previously examined experiments leaked into the holdout")
    return result


def nested_rows(metadata, sizes=SIZES, seed=20261002):
    training = metadata.index[metadata.split == 1].to_numpy()
    if not sizes or tuple(sorted(set(sizes))) != tuple(sizes):
        raise ValueError("Training sizes must be positive, increasing and unique")
    if sizes[0] < 2 or sizes[-1] > len(training):
        raise ValueError("Invalid nested training sizes")
    order = np.random.default_rng(seed).permutation(training)
    return {int(size): np.sort(order[:size]) for size in sizes}


def coverage(frame):
    return {"matrices": len(frame), "pairs": len(pair_set(frame)),
            "drugs": len(set(frame.drug1.str.strip()) | set(frame.drug2.str.strip())),
            "cells": int(frame.cell.nunique())}


def validate_file(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Frozen file changed: {path}")


def prepare(output, previous_root):
    if output.exists():
        raise FileExistsError("Use a new output directory; existing experiments are preserved")
    arrays, previous, manifest = load_data(ROOT / "data/small")
    previous_root = previous_root.resolve()
    comparison = previous_root / "runs/comparison_small"
    selection = read_json(comparison / "selection.json")
    original_protocol = read_json(comparison / "protocol.json")
    if selection["selected_variant"] != "positive_swap":
        raise ValueError("The saved finalist selection differs from the documented plan")
    for name, expected in original_protocol["source_sha256"].items():
        validate_file(ROOT / name, expected)
    validate_file(ROOT / "data.zip", ARCHIVE_SHA256)
    with zipfile.ZipFile(ROOT / "data.zip") as bundle:
        source = pd.read_csv(bundle.open("data/nci_synergy_metadata.csv"))
    source.insert(0, "source_row", np.arange(len(source)))
    # Metadata only: response values have not been deserialized or predicted yet.
    pool = eligible_holdout(source, previous)
    selected = pool.sample(n=1000, random_state=20261001).sort_index().reset_index(drop=True)
    subsets = nested_rows(previous)
    frozen = []
    for variant in FINALISTS:
        for seed in SEEDS:
            item = next(row for row in selection["checkpoints"]
                        if row["variant"] == variant and row["seed"] == seed)
            folder = comparison / variant / f"seed_{seed}"
            validate_file(folder / "best.pt", item["checkpoint_sha256"])
            validate_file(folder / "val_predictions.npz", item["validation_predictions_sha256"])
            saved_metrics = read_json(folder / "metrics.json")
            if saved_metrics["dataset_sha256"] != manifest["dataset_sha256"]:
                raise ValueError("Saved models were trained on a different dataset")
            frozen.append({**item, "folder": str(folder),
                           "metrics_sha256": sha256(folder / "metrics.json"),
                           "history_sha256": sha256(folder / "history.csv")})
    output.mkdir(parents=True)
    data = output / "holdout"
    data.mkdir()
    selected.to_csv(data / "metadata.csv", index=False)
    subset_records = {}
    validation_drugs = set(previous.loc[previous.split == 2, "drug1"].str.strip()) | set(
        previous.loc[previous.split == 2, "drug2"].str.strip())
    validation_cells = set(previous.loc[previous.split == 2, "cell"])
    for size, rows in subsets.items():
        frame = previous.iloc[rows]
        trained_drugs = set(frame.drug1.str.strip()) | set(frame.drug2.str.strip())
        subset_records[str(size)] = {"array_rows": rows.tolist(),
            "source_rows": frame.source_row.to_list(), "coverage": coverage(frame),
            "validation_drugs_absent_from_training": sorted(validation_drugs - trained_drugs),
            "validation_cells_absent_from_training": sorted(validation_cells - set(frame.cell))}
    protocol = {"created_utc": utc_now(), "previous_root": str(previous_root),
        "source_archive_sha256": ARCHIVE_SHA256,
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_metadata_sha256": manifest["metadata_sha256"],
        "original_selection_sha256": sha256(comparison / "selection.json"),
        "finalists": list(FINALISTS), "seeds": list(SEEDS), "training_sizes": list(SIZES),
        "holdout_sampling_seed": 20261001, "nested_sampling_seed": 20261002,
        "bootstrap_seed": 20261003, "bootstrap_replicates": 5000,
        "eligible_pool": coverage(pool), "previously_seen_pairs": len(pair_set(previous)),
        "holdout": coverage(selected), "holdout_metadata_sha256": sha256(data / "metadata.csv"),
        "frozen_checkpoints": frozen, "nested_subsets": subset_records,
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "primary_endpoint": "Mean of seed-specific all-16-entry RMSEs: positive_swap minus reference, at 3000 training matrices",
        "secondary_endpoints": "Compact minus reference and compact minus positive_swap; combination-9 RMSE; Pearson and Spearman",
        "uncertainty": "Paired drug-pair cluster percentile bootstrap, 5000 draws, marginal 95% CIs. Keep all cell matrices and dose entries within each resampled pair. Fixed three models/seeds; no seed resampling, no familywise adjustment. Shared drugs/cells can remain dependent across pair clusters.",
        "holdout_policy": "1000 uniform complete matrices from original test split after excluding ALL pairs present in prior train/val/test subset. Same-source unseen-pair confirmation, not external validation. Never use it for tuning or seed/configuration selection.",
        "learning_curve_policy": "One fixed nested random ordering of the existing 3000 training matrices. Same 1000 validation matrices, configs, seeds, optimizer and early stopping. Reuse the nine frozen 3000-matrix fits; train 27 smaller-subset fits. Freeze every validation checkpoint before any learning-curve holdout evaluation. Report all sizes; no minimum-size selection using holdout.",
        "hyperparameters": {"batch_size": 128, "max_epochs": 100, "patience": 8,
            "grad_clip": 5.0, "cpu_threads": 8,
            "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-5},
            "scheduler": {"name": "LinearLR", "start_factor": 1.0, "end_factor": 0.5, "total_iters": 30}},
        "training_order": "500,1000,2000 matrices; reference,positive_swap,small_swap; seeds 42,43,44",
        "units": "Response in original scale (untreated reference 100); concentrations in micromolar",
        "references": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC5965657/",
                       "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
                       "https://scikit-learn.org/stable/common_pitfalls.html"]}
    # Freeze the sampling identities, endpoints, all model hashes AND learning rules
    # before accessing any new response values.
    dump_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(sha256(output / "protocol.json"), encoding="utf-8")
    drugs = sorted(set(previous.drug1.str.strip()) | set(previous.drug2.str.strip()))
    cells = sorted(previous.cell.unique())
    drug_index = {name: index for index, name in enumerate(drugs)}
    cell_index = {name: index for index, name in enumerate(cells)}
    with zipfile.ZipFile(ROOT / "data.zip") as bundle:
        responses = pickle.loads(bundle.read("data/nci_synergy_dose_response.pickle"))
    matrices, first_doses, second_doses = [], [], []
    for row in selected.itertuples(index=False):
        key = f"{row.drug1}_{row.drug2}_{row.cell}"
        response = responses[key]
        matrices.append(finite_array(response["matrix"], (4, 4), key))
        for key_name, doses in (("x1_dose", first_doses), ("x2_dose", second_doses)):
            converted = finite_array(response[key_name], (4,), key) * 1e6
            if np.any(converted < 0) or np.any(np.diff(converted) < 0):
                raise ValueError("Invalid holdout dose ordering")
            doses.append(converted)
    held_arrays = {"drug_features": arrays["drug_features"].numpy(),
        "cell_features": arrays["cell_features"].numpy(),
        "drug1_index": np.array([drug_index[x.strip()] for x in selected.drug1]),
        "drug2_index": np.array([drug_index[x.strip()] for x in selected.drug2]),
        "cell_index": np.array([cell_index[x] for x in selected.cell]),
        "response": np.stack(matrices), "dose1": np.stack(first_doses), "dose2": np.stack(second_doses),
        "source_row": selected.source_row.to_numpy(dtype=np.int64), "split": np.full(len(selected), 3, dtype=np.int64)}
    np.savez_compressed(data / "dataset.npz", **held_arrays)
    dump_json(data / "manifest.json", {"source_archive_sha256": ARCHIVE_SHA256,
        "dataset_sha256": sha256(data / "dataset.npz"),
        "metadata_sha256": sha256(data / "metadata.csv"), "coverage": coverage(selected),
        "training_dataset_sha256": manifest["dataset_sha256"], "dose_units": "micromolar",
        "previous_pair_overlap": len(pair_set(selected) & pair_set(previous)),
        "previous_row_overlap": len(set(selected.source_row) & set(previous.source_row)),
        "protocol_sha256": sha256(output / "protocol.json")})
    dump_json(output / "environment.json", {"python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "platform": platform.platform()})
    print(f"FROZEN: {coverage(selected)}; eligible pool={coverage(pool)}", flush=True)


def load_protocol(output):
    validate_file(output / "protocol.json", (output / "protocol.sha256").read_text(encoding="utf-8").strip())
    protocol = read_json(output / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        validate_file(ROOT / name, expected)
    arrays, metadata, manifest = load_data(ROOT / "data/small")
    if manifest["dataset_sha256"] != protocol["dataset_sha256"]:
        raise ValueError("Training dataset changed")
    validate_file(output / "holdout/metadata.csv", protocol["holdout_metadata_sha256"])
    held, held_metadata, held_manifest = load_data(output / "holdout")
    if held_manifest["protocol_sha256"] != sha256(output / "protocol.json"):
        raise ValueError("Holdout belongs to a different protocol")
    if pair_set(held_metadata) & pair_set(metadata):
        raise ValueError("Holdout pair leakage")
    return protocol, arrays, metadata, held


def make_loaders(arrays, seed, training_rows=None, device="cuda"):
    datasets = {name: SmallDataset(arrays, name) for name in ("train", "val")}
    if training_rows is not None:
        datasets["train"].rows = torch.tensor(training_rows, dtype=torch.long)
    return {name: DataLoader(dataset, batch_size=128, shuffle=name == "train",
        drop_last=name == "train" and len(dataset) % 128 == 1, num_workers=0,
        pin_memory=device == "cuda", generator=torch.Generator().manual_seed(seed))
        for name, dataset in datasets.items()}


def held_loader(arrays, device):
    return DataLoader(SmallDataset(arrays, "test"), batch_size=128, shuffle=False,
                      num_workers=0, pin_memory=device == "cuda")


def save_held_prediction(model, held, folder, device):
    observed, predicted = predict(model, held_loader(held, device), torch.device(device))
    np.savez_compressed(folder / "holdout_predictions.npz", observed=observed, predicted=predicted,
        source_row=held["source_row"].numpy(), dose1=held["dose1"].numpy(), dose2=held["dose2"].numpy())
    return {"all_16": metrics(observed, predicted),
            "combination_9": metrics(observed[:, 1:, 1:], predicted[:, 1:, 1:]),
            "predictions_sha256": sha256(folder / "holdout_predictions.npz")}


def confirm(output, device):
    protocol, arrays, _, held = load_protocol(output)
    folder = output / "confirmation"
    folder.mkdir(exist_ok=False)
    reports = []
    for item in protocol["frozen_checkpoints"]:
        seed_everything(item["seed"], 8)
        original = Path(item["folder"])
        validate_file(original / "best.pt", item["checkpoint_sha256"])
        validate_file(original / "val_predictions.npz", item["validation_predictions_sha256"])
        checkpoint = torch.load(original / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["variant"]["id"] != item["variant"] or checkpoint["seed"] != item["seed"]:
            raise ValueError("Frozen checkpoint identity mismatch")
        if checkpoint["dataset_sha256"] != protocol["dataset_sha256"]:
            raise ValueError("Checkpoint training provenance differs")
        model, _ = build_model(checkpoint["model_config"], checkpoint["variant"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        _, prediction = predict(model, make_loaders(arrays, item["seed"], device=device)["val"], torch.device(device))
        with np.load(original / "val_predictions.npz", allow_pickle=False) as stored:
            if device == "cuda":
                np.testing.assert_array_equal(prediction, stored["predicted"])
            else:
                np.testing.assert_allclose(prediction, stored["predicted"], rtol=1e-5, atol=2e-4)
        destination = folder / item["variant"] / f"seed_{item['seed']}"
        destination.mkdir(parents=True)
        report = {"variant": item["variant"], "seed": item["seed"], "training_matrices": 3000,
            "checkpoint_sha256": item["checkpoint_sha256"], "validation_reload": "exact" if device == "cuda" else "tolerance",
            "evaluated_utc": utc_now(), **save_held_prediction(model, held, destination, device)}
        dump_json(destination / "metrics.json", report)
        reports.append(report)
        print(f"CONFIRM {item['variant']} seed={item['seed']}: RMSE={report['all_16']['rmse']:.3f}", flush=True)
        del model, checkpoint
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    dump_json(folder / "completed.json", {"completed_utc": utc_now(), "runs": reports})


def train_one(output, variant, seed, size, arrays, protocol):
    folder = output / "learning" / f"n_{size}" / variant["id"] / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    seed_everything(seed, 8)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    settings = protocol["hyperparameters"]
    rows = protocol["nested_subsets"][str(size)]["array_rows"]
    loaders = make_loaders(arrays, seed, rows)
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    base["trainer_args"]["data_dir"] = str(ROOT / "data/small")
    model, config = build_model(base, variant)
    model.to("cuda")
    optimizer_args = dict(settings["optimizer"])
    optimizer_name = optimizer_args.pop("name")
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_args)
    scheduler_args = dict(settings["scheduler"])
    scheduler_name = scheduler_args.pop("name")
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_args)
    swap_rng = torch.Generator().manual_seed(seed + 431069)
    best_loss, stale, best_epoch = float("inf"), 0, None
    torch.cuda.synchronize()
    started = time.perf_counter()
    print(f"TRAIN n={size} {variant['id']} seed={seed}", flush=True)
    with (folder / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_rmse", "val_rmse", "lr", "seconds"])
        writer.writeheader()
        for epoch in range(1, settings["max_epochs"] + 1):
            epoch_started = time.perf_counter()
            model.train()
            total_error, points = 0.0, 0
            for batch in loaders["train"]:
                *inputs, target = [value.to("cuda", non_blocking=True) for value in batch]
                if variant["swap"]:
                    mask = torch.rand(len(target), generator=swap_rng) < 0.5
                    inputs, target = swap_examples(inputs, target, mask)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(*inputs)["dose_response"]
                loss = torch.nn.functional.mse_loss(predictions, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                total_error += loss.item() * target.numel()
                points += target.numel()
            observed, predicted = predict(model, loaders["val"], torch.device("cuda"))
            validation_mse = float(np.mean((observed.astype(np.float64) - predicted) ** 2))
            if not np.isfinite(validation_mse):
                raise FloatingPointError("Non-finite validation loss")
            writer.writerow({"epoch": epoch, "train_rmse": (total_error / points) ** 0.5,
                "val_rmse": validation_mse ** 0.5, "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - epoch_started})
            handle.flush()
            if validation_mse < best_loss:
                best_loss, stale, best_epoch = validation_mse, 0, epoch
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "model_config": config, "variant": variant, "epoch": epoch, "val_mse": validation_mse,
                    "dataset_sha256": protocol["dataset_sha256"], "seed": seed,
                    "training_source_rows": protocol["nested_subsets"][str(size)]["source_rows"],
                    "protocol_sha256": sha256(output / "protocol.json")}, folder / "best.pt")
            else:
                stale += 1
            scheduler.step()
            if epoch % 25 == 0 or stale >= settings["patience"] or epoch == settings["max_epochs"]:
                print(f"  n={size} {variant['id']} s{seed} epoch={epoch} best val={best_loss**0.5:.3f}", flush=True)
            if stale >= settings["patience"]:
                break
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    observed, predicted = predict(model, loaders["val"], torch.device("cuda"))
    np.testing.assert_allclose(metrics(observed, predicted)["rmse"], best_loss**0.5, rtol=1e-10)
    np.savez_compressed(folder / "val_predictions.npz", observed=observed, predicted=predicted,
        source_row=arrays["source_row"][loaders["val"].dataset.rows].numpy())
    report = {"variant": variant["id"], "seed": seed, "training_matrices": size,
        "parameters": sum(value.numel() for value in model.parameters()), "best_epoch": best_epoch,
        "epochs_completed": epoch, "stop_reason": "early_stopping" if stale >= settings["patience"] else "epoch_limit",
        "validation": metrics(observed, predicted), "training_seconds": seconds,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "checkpoint_sha256": sha256(folder / "best.pt"),
        "validation_predictions_sha256": sha256(folder / "val_predictions.npz"),
        "history_sha256": sha256(folder / "history.csv"), "folder": str(folder), "reused": False}
    dump_json(folder / "metrics.json", report)
    print(f"DONE n={size} {variant['id']} seed={seed}: val={report['validation']['rmse']:.3f}, {seconds:.1f}s", flush=True)
    del optimizer, scheduler, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return report


def learning(output):
    protocol, arrays, _, held = load_protocol(output)
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen learning-curve protocol requires CUDA")
    if not (output / "confirmation/completed.json").exists():
        raise ValueError("Complete the nine frozen-checkpoint confirmation jobs first")
    directory = output / "learning"
    directory.mkdir(exist_ok=True)
    if (directory / "selection.json").exists():
        raise FileExistsError("Learning checkpoint selection is already frozen")
    variants = {row["id"]: row for row in variant_catalog()}
    reports = []
    for size in SIZES[:-1]:
        for name in FINALISTS:
            for seed in SEEDS:
                folder = directory / f"n_{size}" / name / f"seed_{seed}"
                if (folder / "metrics.json").exists():
                    # Restart only between completed jobs, never mid-optimizer state.
                    report = read_json(folder / "metrics.json")
                    validate_file(folder / "best.pt", report["checkpoint_sha256"])
                    validate_file(folder / "val_predictions.npz", report["validation_predictions_sha256"])
                    validate_file(folder / "history.csv", report["history_sha256"])
                    print(f"REUSE completed job n={size} {name} seed={seed}", flush=True)
                else:
                    report = train_one(output, variants[name], seed, size, arrays, protocol)
                reports.append(report)
    for item in protocol["frozen_checkpoints"]:
        original = Path(item["folder"])
        validate_file(original / "best.pt", item["checkpoint_sha256"])
        validate_file(original / "metrics.json", item["metrics_sha256"])
        saved = read_json(original / "metrics.json")
        reports.append({"variant": item["variant"], "seed": item["seed"], "training_matrices": 3000,
            "parameters": saved["parameters"], "best_epoch": saved["best_epoch"],
            "epochs_completed": saved["epochs_completed"], "stop_reason": saved["stop_reason"],
            "validation": saved["validation"]["all_16"], "training_seconds": saved["training_seconds"],
            "peak_cuda_memory_mib": saved["peak_cuda_memory_mb"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "validation_predictions_sha256": item["validation_predictions_sha256"],
            "folder": item["folder"], "reused": True})
    if list(directory.glob("n_*/*/seed_*/holdout_predictions.npz")):
        raise AssertionError("Learning holdout inference occurred before the freeze")
    dump_json(directory / "selection.json", {"frozen_utc": utc_now(), "runs": reports})
    print("FROZEN all 36 learning-curve checkpoints (27 new, 9 reused)", flush=True)
    for report in reports:
        if report["reused"]:
            destination = output / "confirmation" / report["variant"] / f"seed_{report['seed']}"
            held_report = read_json(destination / "metrics.json")
        else:
            destination = Path(report["folder"])
            validate_file(destination / "best.pt", report["checkpoint_sha256"])
            checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
            if checkpoint["training_source_rows"] != protocol["nested_subsets"][str(report["training_matrices"])]["source_rows"]:
                raise ValueError("Checkpoint nested training identities differ")
            model, _ = build_model(checkpoint["model_config"], checkpoint["variant"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to("cuda").eval()
            _, predicted = predict(model, make_loaders(arrays, report["seed"])["val"], torch.device("cuda"))
            with np.load(destination / "val_predictions.npz", allow_pickle=False) as stored:
                np.testing.assert_array_equal(predicted, stored["predicted"])
            held_report = {"evaluated_utc": utc_now(), "validation_reload_exact": True,
                           **save_held_prediction(model, held, destination, "cuda")}
            dump_json(destination / "holdout_metrics.json", held_report)
            del model, checkpoint
            gc.collect()
            torch.cuda.empty_cache()
        report["holdout"] = held_report
        print(f"CURVE n={report['training_matrices']} {report['variant']} s{report['seed']}: RMSE={held_report['all_16']['rmse']:.3f}", flush=True)
    load_protocol(output)
    dump_json(directory / "completed.json", {"completed_utc": utc_now(), "runs": reports,
        "new_training_jobs": 27, "reused_training_jobs": 9,
        "new_training_seconds": sum(row["training_seconds"] for row in reports if not row["reused"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/next_steps_20261001")
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--stage", choices=("prepare", "confirm", "learning", "all"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output = args.output.resolve()
    if args.stage in ("prepare", "all"):
        if args.previous_root is None:
            parser.error("--previous-root is required to freeze the previous checkpoint paths")
        prepare(output, args.previous_root)
    if args.stage in ("confirm", "all"):
        confirm(output, args.device)
    if args.stage in ("learning", "all"):
        learning(output)


if __name__ == "__main__":
    main()
