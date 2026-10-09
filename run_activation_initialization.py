"""Frozen 2x2 activation/initialization experiment: twelve validation-only CUDA fits."""
import argparse
import csv
import gc
from pathlib import Path
import platform
import time

from train_small import ROOT, dump_json, load_data, metrics, seed_everything
import numpy as np
import pandas as pd
import torch
import yaml

from activation_initialization import CONDITIONS, SEEDS, build_factorial_model, collect_outputs, head_summary
from ablation_model import POSITIVE_KEYS, swap_examples
from compare_variants import utc_now
from prepare_small_data import sha256
from run_next_steps import make_loaders, read_json, validate_file
from run_step_budget import UpdateBudget, fit_with_updates, verify_run_files

SOURCE_FILES = ("activation_initialization.py", "run_activation_initialization.py",
    "run_step_budget.py", "run_next_steps.py", "ablation_model.py",
    "train_small.py", "prepare_small_data.py", "compare_variants.py", "deepsynba.py",
    "blocks/predictor.py", "blocks/drug_encoder.py", "config.yaml")
ROW_ARRAYS = ("source_row", "split", "drug1_index", "drug2_index", "cell_index", "dose1", "dose2", "response")
REQUIRED_RUN_FILES = ("best.pt", "history.csv", "val_predictions.npz", "initial_outputs.npz", "val_outputs.npz", "diagnostics.json")


def prepare(output, baseline):
    if output.exists():
        raise FileExistsError("Preserve previous work: use a new experiment directory")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared experiment requires the existing CUDA runtime")
    baseline = baseline.resolve()
    validate_file(baseline / "protocol.json", (baseline / "protocol.sha256").read_text().strip())
    old = read_json(baseline / "protocol.json")
    for name, expected in old["source_sha256"].items():
        validate_file(ROOT / name, expected)
    validate_file(baseline / "selection.json", read_json(baseline / "completed.json")["selection_sha256"])
    # The shared original archive is verified/read during preparation only.
    # Training and analysis use a new archive containing only the selected
    # 2,000 training matrices and 1,000 validation matrices, with no test rows.
    arrays, metadata, manifest = load_data(ROOT / "data/small")
    if (manifest["dataset_sha256"], manifest["metadata_sha256"]) != (old["dataset_sha256"], old["metadata_sha256"]):
        raise ValueError("Original data identities changed")
    subset = old["nested_subsets"]["2000"]
    train_rows = np.asarray(subset["array_rows"], dtype=np.int64)
    val_rows = torch.where(arrays["split"] == 2)[0].numpy()
    if len(set(train_rows)) != 2000 or len(val_rows) != 1000 or not (metadata.iloc[train_rows].split == 1).all():
        raise ValueError("Incorrect training/validation selection")
    np.testing.assert_array_equal(arrays["source_row"][train_rows].numpy(), subset["source_rows"])
    rows = np.concatenate([train_rows, val_rows])
    selected = {key: value[rows].numpy() if key in ROW_ARRAYS else value.numpy() for key, value in arrays.items()}
    reduced_metadata = metadata.iloc[rows].reset_index(drop=True)
    controls = []
    for seed in SEEDS:
        record = next(row for row in read_json(baseline / "selection.json")["runs"]
            if (row["variant"], row["training_matrices"], row["seed"]) == ("positive_swap", 2000, seed))
        verify_run_files(record)
        controls.append(record)
    output.mkdir(parents=True)
    data = output / "data"
    data.mkdir()
    np.savez_compressed(data / "train_validation.npz", **selected)
    with np.load(data / "train_validation.npz", allow_pickle=False) as exported:
        for key, value in arrays.items():
            expected = value[rows].numpy() if key in ROW_ARRAYS else value.numpy()
            np.testing.assert_array_equal(exported[key], expected)
    reduced_metadata.to_csv(data / "metadata.csv", index=False)
    protocol = {"created_utc": utc_now(), "experiment": "Separate activation and output initialization",
        "conditions": list(CONDITIONS), "seeds": list(SEEDS), "training_matrices": 2000,
        "validation_matrices": 1000, "parameters_expected": 40944401,
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "original_dataset_sha256": manifest["dataset_sha256"], "original_metadata_sha256": manifest["metadata_sha256"],
        "dataset_sha256": sha256(data / "train_validation.npz"), "metadata_sha256": sha256(data / "metadata.csv"),
        "training_source_rows": subset["source_rows"], "validation_source_rows": arrays["source_row"][val_rows].tolist(),
        "baseline_experiment": str(baseline), "baseline_protocol_sha256": sha256(baseline / "protocol.json"),
        "baseline_selection_sha256": sha256(baseline / "selection.json"), "controls": controls,
        "hyperparameters": old["hyperparameters"], "affected_heads": list(POSITIVE_KEYS),
        "activation_factor": "Released ReLU versus Softplus + 0.0001 on h1,h2,alpha only.",
        "initialization_factor": "Original Kaiming final weights/zero biases versus the output-calibration package: same weights times 0.01; bias 1 for ReLU or log(expm1(1-0.0001)) for Softplus. Calibration targets output one; it combines weight scaling and activation-specific bias, not an isolated bias intervention.",
        "fixed_settings": "Full 40,944,401-parameter architecture, additive bias, exact previous 2000 training/1000 validation rows, float32 deterministic CUDA, batch128, AdamW1e-4/weight_decay1e-5, clip5, common 2400-update cap, validation24, patience192, LinearLR30 checks ending5e-5 after720 updates.",
        "augmentation": old["augmentation"],
        "primary_endpoint": "Mean seed-specific all-16 validation RMSE for the four factorial cells, with paired conditional activation and calibration effects and interaction.",
        "contrast_definitions": {"activation_original": "softplus_original - relu_original",
            "activation_near_one": "softplus_near_one - relu_near_one", "initialization_relu": "relu_near_one - relu_original",
            "initialization_softplus": "softplus_near_one - softplus_original",
            "interaction": "(softplus_near_one - relu_near_one) - (softplus_original - relu_original)",
            "average_activation": "half the sum of the two conditional activation effects",
            "average_initialization": "half the sum of the two conditional initialization effects"},
        "secondary_endpoints": "Combination-9 validation RMSE, Pearson/Spearman, initial/selected h1,h2,alpha zero fractions and distributions, actual/best updates, memory, training time.",
        "checkpoint_policy": old["selection_policy"].split("Rank the six")[0],
        "selection_policy": "Rank all four cells by mean all-16 validation RMSE over seeds42/43/44, then SD and condition ID. Do not select an individual seed.",
        "uncertainty": "Mean and sample SD across three paired fixed seeds; descriptive single-split factorial effects. No significance tests or confidence/generalization intervals. Early stopping can change realized update counts.",
        "evaluation_policy": "Adaptive validation-only mechanism study. Preparation reads the shared original archive to export only training/validation rows. No holdout loaders, predictions, responses, inference or rankings enter training/analysis; both examined holdouts remain development evidence.",
        "control_policy": "Softplus/near-one exactly matches the previous positive_swap model at2000 matrices. Require all three seeded controls to reproduce stored validation predictions and training/LR/stopping histories before completing the other cells.",
        "training_order": ["softplus_near_one", "relu_original", "softplus_original", "relu_near_one"],
        "compute_policy": "Same update maximum, not fixed realized training length or equal FLOPs. Seconds are descriptive and include validation/checkpoint I/O; extra initial/final head diagnostics are outside timed training. No speedup claim.",
        "units": "Responses in original units; untreated reference100. Doses micromolar."}
    dump_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(sha256(output / "protocol.json"), encoding="utf-8")
    dump_json(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "pandas": pd.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "platform": platform.platform(), "deterministic": True})
    load_protocol(output)
    print("FROZEN: four factorial cells x three seeds; 2000 training / 1000 validation; no test rows in fit archive", flush=True)


def load_protocol(output):
    validate_file(output / "protocol.json", (output / "protocol.sha256").read_text().strip())
    protocol = read_json(output / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        validate_file(ROOT / name, expected)
    data = output / "data"
    validate_file(data / "train_validation.npz", protocol["dataset_sha256"])
    validate_file(data / "metadata.csv", protocol["metadata_sha256"])
    with np.load(data / "train_validation.npz", allow_pickle=False) as saved:
        arrays = {key: torch.from_numpy(saved[key]) for key in saved.files}
    metadata = pd.read_csv(data / "metadata.csv")
    if set(metadata.split) != {1, 2} or len(metadata) != 3000 or len(set(metadata.source_row)) != 3000:
        raise ValueError("Fit archive contains invalid partitions/duplicate matrices")
    for key in ("split", "source_row"):
        np.testing.assert_array_equal(arrays[key].numpy(), metadata[key])
    for split, field in ((1, "training_source_rows"), (2, "validation_source_rows")):
        np.testing.assert_array_equal(arrays["source_row"][arrays["split"] == split].numpy(), protocol[field])
    pair_sets = [{tuple(sorted((row.drug1, row.drug2))) for row in
        metadata.loc[metadata.split == split].itertuples()} for split in (1, 2)]
    if pair_sets[0] & pair_sets[1]:
        raise ValueError("Training/validation drug-pair leakage")
    baseline = Path(protocol["baseline_experiment"])
    validate_file(baseline / "protocol.json", protocol["baseline_protocol_sha256"])
    validate_file(baseline / "selection.json", protocol["baseline_selection_sha256"])
    return protocol, arrays, metadata


def verify_completed_run(record):
    folder = Path(record["folder"])
    for filename, expected in record["file_sha256"].items():
        validate_file(folder / filename, expected)


def train_one(output, condition, seed, arrays, protocol):
    folder = output / "training" / condition["id"] / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    seed_everything(seed, 8)
    torch.cuda.empty_cache()
    settings = protocol["hyperparameters"]
    loaders = make_loaders(arrays, seed)
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    base["trainer_args"]["data_dir"] = str(output / "data")
    model, config = build_factorial_model(base, condition)
    model.to("cuda")
    initial_truth, initial = collect_outputs(model, loaders["val"], "cuda")
    np.savez_compressed(folder / "initial_outputs.npz", **initial,
        source_row=np.asarray(protocol["validation_source_rows"]))
    optimizer_args = dict(settings["optimizer"])
    scheduler_args = dict(settings["scheduler"])
    optimizer_args.pop("name"); scheduler_args.pop("name")
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_args)
    budget = UpdateBudget(settings["max_updates"], settings["validation_interval"], settings["patience_evaluations"])
    swap_rng = torch.Generator().manual_seed(seed + 431069)
    def augment(inputs, target):
        return swap_examples(inputs, target, torch.rand(len(target), generator=swap_rng) < 0.5)
    fields = ["update", "epoch", "batch_in_epoch", "epoch_equivalent", "examples_seen", "train_rmse",
        "val_rmse", "val_combination_rmse", "lr", "stale_evaluations", "improved", "seconds"]
    fields += [key + "_last_batch_grad_norm" for key in POSITIVE_KEYS]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    print(f"TRAIN {condition['id']} seed={seed}; max2400 updates", flush=True)
    with (folder / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        def on_validation(row, current_model):
            saved_row = dict(row)
            for key in POSITIVE_KEYS:
                grad = current_model.predictors[key].prediction_head[-2].weight.grad
                saved_row[key + "_last_batch_grad_norm"] = None if grad is None else float(grad.norm().item())
            writer.writerow(saved_row)
            handle.flush()
            if row["improved"]:
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in current_model.state_dict().items()},
                    "model_config": config, "condition": condition, "seed": seed,
                    "best_update": row["update"], "val_mse": row["val_rmse"] ** 2,
                    "dataset_sha256": protocol["dataset_sha256"], "training_source_rows": protocol["training_source_rows"],
                    "protocol_sha256": sha256(output / "protocol.json")}, folder / "best.pt")
            if row["update"] % 600 == 0 or row["stale_evaluations"] == budget.patience_evaluations:
                print(f"  {condition['id']} s{seed} update={row['update']} val={row['val_rmse']:.3f}", flush=True)
        fitted = fit_with_updates(model, optimizer, scheduler, loaders["train"], loaders["val"],
            budget=budget, on_validation=on_validation, device="cuda", gradient_clip=settings["grad_clip"], augment=augment)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    observed, outputs = collect_outputs(model, loaders["val"], "cuda")
    predicted = outputs.pop("dose_response")
    validation = {"all_16": metrics(observed, predicted), "combination_9": metrics(observed[:,1:,1:], predicted[:,1:,1:])}
    np.testing.assert_allclose(validation["all_16"]["rmse"], fitted["best_validation_mse"] ** 0.5, rtol=1e-10)
    np.testing.assert_array_equal(initial_truth, observed)
    rows = loaders["val"].dataset.rows
    np.savez_compressed(folder / "val_predictions.npz", observed=observed, predicted=predicted,
        source_row=arrays["source_row"][rows].numpy(), dose1=arrays["dose1"][rows].numpy(), dose2=arrays["dose2"][rows].numpy())
    np.savez_compressed(folder / "val_outputs.npz", **outputs, source_row=arrays["source_row"][rows].numpy())
    dump_json(folder / "diagnostics.json", {"initial": head_summary(initial), "best": head_summary(outputs),
        "initial_validation_rmse": metrics(initial_truth, initial["dose_response"])["rmse"],
        "gradient_note": "History records each head's final-linear post-clipping gradient norm on the last batch before each check, not all training batches."})
    control = None
    if condition["id"] == "softplus_near_one":
        old = next(row for row in protocol["controls"] if row["seed"] == seed)
        verify_run_files(old)
        with np.load(Path(old["folder"]) / "val_predictions.npz", allow_pickle=False) as previous:
            np.testing.assert_array_equal(previous["predicted"], predicted)
        columns = ["update", "epoch", "batch_in_epoch", "examples_seen", "train_rmse", "val_rmse", "lr", "stale_evaluations"]
        np.testing.assert_allclose(pd.read_csv(folder / "history.csv")[columns],
            pd.read_csv(Path(old["folder"]) / "history.csv")[columns], rtol=0, atol=1e-10)
        control = True
    record = {key: value for key, value in fitted.items() if key != "history"}
    record.update({"condition": condition, "seed": seed, "training_matrices": 2000,
        "parameters": sum(parameter.numel() for parameter in model.parameters()), "batches_per_epoch": len(loaders["train"]),
        "validation": validation, "training_seconds": seconds, "peak_cuda_memory_mib": peak_memory,
        "control_exactly_reproduced": control, "folder": str(folder), "dataset_sha256": protocol["dataset_sha256"],
        "protocol_sha256": sha256(output / "protocol.json"),
        "file_sha256": {name: sha256(folder / name) for name in REQUIRED_RUN_FILES}})
    if record["parameters"] != protocol["parameters_expected"]:
        raise ValueError("Unexpected parameter count")
    dump_json(folder / "metrics.json", record)
    print(f"DONE {condition['id']} seed={seed}: val={validation['all_16']['rmse']:.3f}, updates={fitted['updates_completed']}, best={fitted['best_update']}, {seconds:.1f}s", flush=True)
    del model, optimizer, scheduler, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return record


def train(output):
    protocol, arrays, _ = load_protocol(output)
    if (output / "completed.json").exists():
        raise FileExistsError("Experiment already completed")
    reports = []
    for name in protocol["training_order"]:
        condition = next(row for row in CONDITIONS if row["id"] == name)
        for seed in protocol["seeds"]:
            folder = output / "training" / name / f"seed_{seed}"
            if (folder / "metrics.json").exists():
                report = read_json(folder / "metrics.json")
                verify_completed_run(report)
                if report["protocol_sha256"] != sha256(output / "protocol.json"):
                    raise ValueError("Completed run belongs to a different protocol")
                print(f"REUSE completed {name} seed={seed}", flush=True)
            else:
                report = train_one(output, condition, seed, arrays, protocol)
            reports.append(report)
    ranking = []
    for condition in CONDITIONS:
        values = [row["validation"]["all_16"]["rmse"] for row in reports if row["condition"] == condition]
        ranking.append({**condition, "validation_rmse_mean": float(np.mean(values)),
            "validation_rmse_sd": float(np.std(values, ddof=1))})
    ranking.sort(key=lambda row: (row["validation_rmse_mean"],row["validation_rmse_sd"],row["id"]))
    load_protocol(output)
    dump_json(output / "selection.json", {"frozen_utc": utc_now(), "ranking": ranking, "selected_condition": ranking[0], "runs": reports})
    dump_json(output / "completed.json", {"completed_utc": utc_now(), "new_training_jobs": len(reports),
        "new_training_seconds": sum(row["training_seconds"] for row in reports),
        "total_optimizer_updates": sum(row["updates_completed"] for row in reports),
        "selection_sha256": sha256(output / "selection.json"), "runs": reports})
    print(f"COMPLETE: 12 fits; validation winner={ranking[0]['id']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/activation_initialization_20261008")
    parser.add_argument("--baseline",type=Path,default=ROOT/"runs/step_budget_20261001")
    parser.add_argument("--stage",choices=("prepare","train","all"),default="all")
    args=parser.parse_args()
    output=args.output.resolve()
    if args.stage in ("prepare","all"):
        prepare(output,args.baseline)
    if args.stage in ("train","all"):
        train(output)


if __name__=="__main__":
    main()
