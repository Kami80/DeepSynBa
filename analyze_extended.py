"""Verify and visualize the fixed three-seed DeepSynBa experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from prepare_small_data import ROOT, sha256

BLUE, ORANGE, GREY = "#255B83", "#C47026", "#626B73"
RESIDUAL_MAP = LinearSegmentedColormap.from_list("residual", [BLUE, "#FAFAF8", ORANGE])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "text.color": "#20272C", "axes.labelcolor": "#20272C",
                     "axes.edgecolor": "#8B949B", "grid.color": "#E2E5E7",
                     "grid.linewidth": 0.7, "lines.linewidth": 1.7,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def independent_metrics(observed, predicted):
    """Recompute metrics independently of the training runner's metric helper."""
    x, y = np.asarray(observed, dtype=np.float64).ravel(), np.asarray(predicted, dtype=np.float64).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Predictions must be finite and aligned with measurements")

    def correlation(a, b):
        a, b = a - a.mean(), b - b.mean()
        denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
        return float(np.dot(a, b) / denominator) if denominator else None

    return {"rmse": float(np.linalg.norm(x - y) / np.sqrt(x.size)),
            "pearson": correlation(x, y), "spearman": correlation(rankdata(x), rankdata(y)),
            "dose_points": int(x.size)}


def assert_metrics(actual, recorded):
    for key in ("rmse", "pearson", "spearman"):
        if actual[key] is None or recorded[key] is None:
            if actual[key] != recorded[key]:
                raise AssertionError(f"Metric mismatch: {key}")
        else:
            np.testing.assert_allclose(actual[key], recorded[key], rtol=1e-9, atol=1e-9)
    if actual["dose_points"] != recorded["dose_points"]:
        raise AssertionError("Metric denominator mismatch")


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def seed_summary(table):
    fields = ["validation_rmse", "test_rmse", "test_pearson", "test_spearman", "combination_rmse"]
    return {name: {"mean": float(table[name].mean()), "sample_sd": float(table[name].std(ddof=1)),
                   "minimum": float(table[name].min()), "maximum": float(table[name].max())}
            for name in fields}


def choose_cases(errors, quantiles=(0.1, 0.5, 0.9, 0.99)):
    order = np.argsort(errors, kind="stable")
    return [(q, int(order[int(round(q * (len(order) - 1)))])) for q in quantiles]


def parameter_diagnostics(experiment, seeds):
    import torch
    from torch.utils.data import DataLoader
    from deepsynba import DeepSynBa
    from train_small import load_data, SmallDataset, seed_everything

    seed_everything(42, 8)
    arrays, _, _ = load_data(ROOT / "data/small")
    loader = DataLoader(SmallDataset(arrays, "val"), batch_size=128, shuffle=False)
    diagnostics = {}
    for seed in seeds:
        checkpoint = torch.load(experiment / f"seed_{seed}/best.pt", map_location="cpu", weights_only=True)
        model = DeepSynBa(checkpoint["model_config"]).to("cuda").eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        parameters = {key: [] for key in model.target_keys}
        response_predictions = []
        with torch.no_grad():
            for batch in loader:
                outputs = model(*(value.to("cuda") for value in batch[:-1]))
                response_predictions.append(outputs["dose_response"].cpu().numpy())
                for name in parameters:
                    parameters[name].append(outputs[name].cpu().numpy().ravel())
        with np.load(experiment / f"seed_{seed}/evaluation/val_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(np.concatenate(response_predictions), saved["predicted"])
        diagnostics[str(seed)] = {}
        for name, pieces in parameters.items():
            values = np.concatenate(pieces)
            diagnostics[str(seed)][name] = {"minimum": float(values.min()), "maximum": float(values.max()),
                                           "mean": float(values.mean()), "zero_fraction": float((values == 0).mean())}
        del model
    return diagnostics


def learning_curves(directory, runs, baseline):
    fig, axes = plt.subplots(1, len(runs), figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, item in zip(axes, runs):
        history, metrics = item["history"], item["training"]
        ax.plot(history.epoch, history.train_rmse, color=BLUE, linestyle="--", label="Training (during optimization)")
        ax.plot(history.epoch, history.val_rmse, color=ORANGE, label="Validation (evaluation mode)")
        ax.axhline(baseline, color=GREY, linestyle=":", linewidth=1.3, label="Validation mean-surface baseline")
        ax.axvline(30, color="#ABB1B5", linestyle=":", linewidth=1.1)
        ax.scatter([metrics["best_epoch"]], [metrics["best_validation_rmse"]], color=ORANGE,
                   edgecolor="white", s=55, zorder=4)
        ax.set_title(f"Seed {metrics['seed']} · best epoch {metrics['best_epoch']}")
        ax.set_xlabel("Training epoch")
        ax.set_xlim(1, 100)
        ax.set_xticks([1, 30, 50, 75, 100])
        ax.grid(axis="y")
    axes[0].set_ylabel("RMSE (original response units)")
    fig.suptitle("Learning curves with a 100-epoch limit", fontsize=17, x=0.07, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.025))
    fig.text(0.07, 0.005, "3,000 training and 1,000 validation matrices per seed. Dotted vertical line: original 30-epoch limit.",
             fontsize=9, color=GREY)
    fig.tight_layout(rect=(0.02, 0.13, 0.99, 0.94))
    save_figure(fig, directory, "learning_curves")


def seed_comparison(directory, table, baseline):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, key, label, baseline_key in zip(axes, ["test_rmse", "test_pearson", "test_spearman"],
                                           ["Test RMSE", "Test Pearson correlation", "Test Spearman correlation"],
                                           ["rmse", "pearson", "spearman"]):
        values = table[key].to_numpy()
        ax.scatter(np.arange(3), values, color=BLUE, s=55, zorder=4)
        ax.errorbar(3.3, values.mean(), yerr=values.std(ddof=1), fmt="D", color=ORANGE,
                    markersize=7, capsize=6, label="Mean ± sample SD", zorder=4)
        ax.axhline(baseline[baseline_key], color=GREY, linestyle=":", label="Mean-surface baseline")
        for i, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.set_xticks([0, 1, 2, 3.3], [str(s) for s in table.seed] + ["Mean ± SD"])
        ax.set_xlabel("Training seed")
        ax.set_title(label)
        ax.set_xlim(-0.5, 4.1)
        ax.set_ylim(0, max(baseline[baseline_key], values.max()) * 1.17 if key == "test_rmse" else 1)
        ax.grid(axis="y")
    axes[0].set_ylabel("Original response units")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Test performance across three training seeds", fontsize=17, x=0.07, ha="left")
    fig.text(0.07, 0.02, "Fixed 1,000-matrix test set (16,000 entries). SD measures seed variation, not a confidence interval.",
             fontsize=9, color=GREY)
    fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.94))
    save_figure(fig, directory, "seed_comparison")


def validation_quality(directory, saved, seed):
    observed, predicted = saved["observed"], saved["predicted"]
    errors = np.sqrt(np.mean((observed.astype(np.float64) - predicted) ** 2, axis=(1, 2)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    margin = (high - low) * 0.04
    low, high = low - margin, high + margin
    density = axes[0].hexbin(observed.ravel(), predicted.ravel(), gridsize=42, mincnt=1,
                             bins="log", cmap="Blues", extent=(low, high, low, high), linewidths=0)
    axes[0].plot([low, high], [low, high], color="#282D31", linestyle="--", linewidth=1.1)
    axes[0].set(xlim=(low, high), ylim=(low, high), xlabel="Measured response (untreated = 100)",
                ylabel="Predicted response (untreated = 100)", title="All 16,000 validation dose entries")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(density, ax=axes[0], label="Entries per hexagon (log scale)", shrink=0.82)
    axes[1].hist(errors, bins=30, color=BLUE, edgecolor="white", linewidth=0.5)
    axes[1].axvline(np.median(errors), color=ORANGE, linestyle="--", label=f"Median = {np.median(errors):.2f}")
    axes[1].set(xlabel="RMSE per complete 4 × 4 matrix", ylabel="Validation matrices",
                title="Error distribution across 1,000 matrices")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y")
    fig.suptitle(f"Validation prediction quality · seed {seed}", fontsize=17, x=0.06, ha="left")
    fig.text(0.06, 0.01, "Model chosen by validation RMSE. Every validation point is included; measurements and predictions are not clipped.",
             fontsize=9, color=GREY)
    fig.tight_layout(rect=(0.01, 0.065, 0.99, 0.94))
    save_figure(fig, directory, "validation_quality")
    return errors


def dose_response_examples(directory, saved, metadata, seed, errors):
    choices = choose_cases(errors)
    indices = [i for _, i in choices]
    all_values = np.concatenate([saved["observed"][indices].ravel(), saved["predicted"][indices].ravel()])
    vmin, vmax = float(all_values.min()), float(all_values.max())
    residuals = saved["predicted"] - saved["observed"]
    error_limit = float(np.abs(residuals[indices]).max())
    fig = plt.figure(figsize=(14, 18), layout="constrained")
    grid = fig.add_gridspec(8, 3, height_ratios=[0.2, 1] * 4)
    response_axes, residual_axes, records = [], [], []
    for row_number, (quantile, index) in enumerate(choices):
        meta = metadata.iloc[index]
        title_ax = fig.add_subplot(grid[row_number * 2, :])
        title_ax.set_axis_off()
        title = (f"{quantile:.0%} error percentile · {meta.drug1} + {meta.drug2} · {meta.cell} "
                 f"· matrix RMSE {errors[index]:.2f}")
        title_ax.text(0.02, 0.4, textwrap.fill(title, width=115), ha="left", va="center", fontsize=11)
        for col, (name, values) in enumerate((("Measured", saved["observed"][index]),
                                               ("Predicted", saved["predicted"][index]),
                                               ("Predicted − measured", residuals[index]))):
            ax = fig.add_subplot(grid[row_number * 2 + 1, col])
            signed = col == 2
            norm_min, norm_max = (-error_limit, error_limit) if signed else (vmin, vmax)
            im = ax.imshow(values, origin="lower", cmap=RESIDUAL_MAP if signed else "cividis",
                           vmin=norm_min, vmax=norm_max, interpolation="nearest", aspect="equal")
            ax.set_title(name)
            ax.set_xticks(range(4), [f"{x:.2g}" for x in saved["dose2"][index]])
            ax.set_yticks(range(4), [f"{x:.2g}" for x in saved["dose1"][index]])
            ax.set_xlabel("Drug 2 concentration (µM)")
            ax.set_ylabel("Drug 1 concentration (µM)")
            for a in range(4):
                for b in range(4):
                    normalized = (values[a, b] - norm_min) / (norm_max - norm_min)
                    color = "white" if ((signed and (normalized < 0.2 or normalized > 0.8)) or
                                         (not signed and normalized < 0.45)) else "#20272C"
                    ax.text(b, a, f"{values[a, b]:.1f}", ha="center", va="center", color=color, fontsize=10)
            if signed:
                residual_axes.append(ax)
                residual_image = im
            else:
                response_axes.append(ax)
                response_image = im
        records.append({"quantile": quantile, "validation_index": index,
                        "source_row": int(meta.source_row), "drug1": meta.drug1, "drug2": meta.drug2,
                        "cell": meta.cell, "matrix_rmse": float(errors[index]),
                        "observed": saved["observed"][index].tolist(),
                        "predicted": saved["predicted"][index].tolist(),
                        "dose1_micromolar": saved["dose1"][index].tolist(),
                        "dose2_micromolar": saved["dose2"][index].tolist()})
    fig.colorbar(response_image, ax=response_axes, location="bottom", shrink=0.7, pad=0.04,
                 label="Response (untreated reference = 100); common scale across examples")
    fig.colorbar(residual_image, ax=residual_axes, location="bottom", shrink=0.9, pad=0.04,
                 label="Residual (response units)")
    fig.suptitle(f"Validation dose-response matrices · seed {seed}\n"
                 "Examples span low, median, high, and extreme error; equally spaced cells represent measured dose levels",
                 fontsize=15)
    save_figure(fig, directory, "validation_matrices")
    return records


def analyze(experiment):
    experiment = Path(experiment).resolve()
    protocol, selection = read_json(experiment / "protocol.json"), read_json(experiment / "selection.json")
    read_json(experiment / "completed.json")
    directory = experiment / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    with np.load(ROOT / "data/small/dataset.npz", allow_pickle=False) as source:
        source_arrays = {k: source[k] for k in source.files}
    if sha256(ROOT / "data/small/dataset.npz") != protocol["dataset_sha256"]:
        raise AssertionError("Experiment data changed")
    train_surface = source_arrays["response"][source_arrays["split"] == 1].mean(axis=0)
    runs, rows, input_hashes = [], [], {}
    reference_predictions = {}
    for seed in protocol["seeds"]:
        folder = experiment / f"seed_{seed}"
        training = read_json(folder / "metrics.json")
        result = read_json(folder / "evaluation/metrics.json")
        history = pd.read_csv(folder / "history.csv")
        frozen = next(row for row in selection["selections"] if row["seed"] == seed)
        if sha256(folder / "best.pt") != frozen["checkpoint_sha256"]:
            raise AssertionError("Frozen checkpoint changed")
        if training["test_evaluated"] or (folder / "test_predictions.npz").exists():
            raise AssertionError("Training phase evaluated test data")
        best_row = history.loc[history.val_rmse.idxmin()]
        assert int(best_row.epoch) == training["best_epoch"] == frozen["best_epoch"]
        np.testing.assert_allclose(best_row.val_rmse, result["validation_all_16_doses"]["rmse"], atol=1e-10)
        for split, code in (("val", 2), ("test", 3)):
            artifact = folder / "evaluation" / f"{split}_predictions.npz"
            with np.load(artifact, allow_pickle=False) as saved:
                saved = {k: saved[k] for k in saved.files}
            mask = source_arrays["split"] == code
            np.testing.assert_array_equal(saved["source_row"], source_arrays["source_row"][mask])
            np.testing.assert_array_equal(saved["observed"], source_arrays["response"][mask])
            for key in ("dose1", "dose2"):
                np.testing.assert_array_equal(saved[key], source_arrays[key][mask])
            scope = "validation" if split == "val" else "test"
            assert_metrics(independent_metrics(saved["observed"], saved["predicted"]), result[f"{scope}_all_16_doses"])
            assert_metrics(independent_metrics(saved["observed"][:, 1:, 1:], saved["predicted"][:, 1:, 1:]),
                           result[f"{scope}_9_nonzero_combinations"])
            if split == "val":
                with np.load(folder / "val_predictions.npz", allow_pickle=False) as training_saved:
                    np.testing.assert_array_equal(saved["predicted"], training_saved["predicted"])
            if seed == selection["diagnostic_seed"]:
                reference_predictions[split] = saved
            input_hashes[str(artifact.relative_to(experiment))] = sha256(artifact)
        base_truth = source_arrays["response"][source_arrays["split"] == 3]
        baseline = independent_metrics(base_truth, np.broadcast_to(train_surface, base_truth.shape))
        assert_metrics(baseline, result["training_mean_surface_baseline"])
        rows.append({"seed": seed, "best_epoch": training["best_epoch"], "epochs_completed": len(history),
                     "stop_reason": training["stopping_reason"],
                     "train_eval_rmse": training["train_eval_all_16_doses"]["rmse"],
                     "validation_rmse": result["validation_all_16_doses"]["rmse"],
                     "test_rmse": result["test_all_16_doses"]["rmse"],
                     "test_pearson": result["test_all_16_doses"]["pearson"],
                     "test_spearman": result["test_all_16_doses"]["spearman"],
                     "combination_rmse": result["test_9_nonzero_combinations"]["rmse"],
                     "training_seconds": training["training_seconds"]})
        runs.append({"training": training, "result": result, "history": history})
        for relative in ("metrics.json", "history.csv", "evaluation/metrics.json", "environment.json"):
            input_hashes[str((folder / relative).relative_to(experiment))] = sha256(folder / relative)
    table = pd.DataFrame(rows)
    table.to_csv(directory / "seed_results.csv", index=False)
    summary = seed_summary(table)
    old = read_json(ROOT / "runs/small_gpu/metrics.json")
    original_history = pd.read_csv(ROOT / "runs/small_gpu/history.csv")
    seed42 = next(item for item in runs if item["training"]["seed"] == 42)
    np.testing.assert_allclose(seed42["history"].iloc[:30][["train_rmse", "val_rmse"]],
                               original_history[["train_rmse", "val_rmse"]], rtol=0, atol=1e-10)
    selected_seed = selection["diagnostic_seed"]
    selected_row = table.loc[table.seed == selected_seed].iloc[0]
    learning_curves(figures, runs, runs[0]["training"]["validation_mean_surface_baseline"]["rmse"])
    seed_comparison(figures, table, baseline)
    errors = validation_quality(figures, reference_predictions["val"], selected_seed)
    metadata = pd.read_csv(experiment / f"seed_{selected_seed}/evaluation/val_metadata.csv")
    np.testing.assert_array_equal(metadata.source_row, reference_predictions["val"]["source_row"])
    cases = dose_response_examples(figures, reference_predictions["val"], metadata, selected_seed, errors)
    write_json(directory / "validation_examples.json", cases)
    heads = parameter_diagnostics(experiment, protocol["seeds"])
    write_json(directory / "parameter_diagnostics.json", heads)
    comparison = {"original_30_epoch_seed42_test_rmse": old["test_all_16_doses"]["rmse"],
                  "extended_seed42_test_rmse": float(table.loc[table.seed == 42, "test_rmse"].iloc[0]),
                  "original_30_epoch_seed42_validation_rmse": old["best_validation_rmse"],
                  "extended_seed42_validation_rmse": float(table.loc[table.seed == 42, "validation_rmse"].iloc[0])}
    comparison["matched_seed42_test_rmse_reduction_percent"] = 100 * (1 - comparison["extended_seed42_test_rmse"] /
                                                                        comparison["original_30_epoch_seed42_test_rmse"])
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "experiment": str(experiment),
              "dataset_sha256": protocol["dataset_sha256"], "seeds": protocol["seeds"],
              "diagnostic_seed_selected_on_validation": selected_seed, "summary": summary,
              "test_mean_surface_baseline": baseline, "comparison_to_30_epochs": comparison,
              "validation_matrix_rmse_quantiles": {str(q): float(np.quantile(errors, q)) for q in (0.1, 0.5, 0.9, 0.99)},
              "total_training_seconds": float(table.training_seconds.sum()),
              "uncertainty_definition": "Sample SD across three training seeds on one fixed dataset; not a confidence interval",
              "independent_verification": {"all_test_and_validation_metrics_recomputed": True,
                                            "dose_units_and_row_alignment_verified": True,
                                            "selected_checkpoints_unchanged": True,
                                            "validation_predictions_reproduced_exactly": True,
                                            "seed42_first30_epochs_match_prior_run": True},
              "validation_parameter_diagnostics": heads,
              "input_sha256": input_hashes}
    write_json(directory / "summary.json", report)
    baseline_improvement = 100 * (1 - summary["test_rmse"]["mean"] / baseline["rmse"])
    selected_test = float(selected_row.test_rmse)
    lines = ["# DeepSynBa: extended small-data results", "",
             f"Completed three runs with a 100-epoch limit on the unchanged 5,000-example dataset. "
             f"Test RMSE across training seeds is **{summary['test_rmse']['mean']:.3f} ± {summary['test_rmse']['sample_sd']:.3f}** "
             f"(mean ± sample SD; three seeds). The training-mean surface baseline has RMSE **{baseline['rmse']:.3f}**. "
             f"The mean RMSE is {baseline_improvement:.1f}% lower than that baseline.", "",
             "## Per-seed results", "",
             "| Seed | Epochs run | Selected epoch | Train RMSE (eval) | Validation RMSE | Test RMSE | Pearson | Spearman | Combination-only RMSE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in table.itertuples():
        lines.append(f"| {row.seed} | {row.epochs_completed} | {row.best_epoch} | {row.train_eval_rmse:.3f} | "
                     f"{row.validation_rmse:.3f} | {row.test_rmse:.3f} | {row.test_pearson:.3f} | "
                     f"{row.test_spearman:.3f} | {row.combination_rmse:.3f} |")
    lines += ["", "All-entry metrics pool 16,000 measured dose entries from 1,000 test matrices. "
              "Combination-only metrics use the 9,000 nonzero-dose interior entries. Lower RMSE is better.", "",
              f"Mean ± sample SD: Pearson **{summary['test_pearson']['mean']:.3f} ± {summary['test_pearson']['sample_sd']:.3f}**; "
              f"Spearman **{summary['test_spearman']['mean']:.3f} ± {summary['test_spearman']['sample_sd']:.3f}**; "
              f"combination-only RMSE **{summary['combination_rmse']['mean']:.3f} ± {summary['combination_rmse']['sample_sd']:.3f}**.", "",
              "![Test performance across seeds](figures/seed_comparison.png)", "",
              "## Longer training and generalization", "",
              f"For the matched seed 42, validation RMSE changed from {comparison['original_30_epoch_seed42_validation_rmse']:.3f} "
              f"to {comparison['extended_seed42_validation_rmse']:.3f}. Test RMSE changed from "
              f"{comparison['original_30_epoch_seed42_test_rmse']:.3f} to {comparison['extended_seed42_test_rmse']:.3f} "
              f"({comparison['matched_seed42_test_rmse_reduction_percent']:.1f}% reduction). Its first 30 training and validation "
              "epochs reproduce the prior GPU run numerically, so this comparison isolates the extra training budget for that seed.", "",
              "The learning curves show online training RMSE with dropout enabled and validation RMSE in evaluation mode. "
              "Those curves are not directly equivalent estimators. The table's training RMSE is recomputed in evaluation mode "
              "at the selected checkpoint, making its train/validation gap more comparable. A gap indicates generalization "
              "error on this split; it does not alone establish that training longer causes overfitting.", "",
              "![Training and validation learning curves](figures/learning_curves.png)", "",
              "## Validation prediction inspection", "",
              f"Seed **{selected_seed}** was selected for diagnostics using validation RMSE before any new test evaluation. "
              f"Its test RMSE is {selected_test:.3f}; this is a validation-selected result, not selection of the best test score. "
              "The distribution below includes all validation examples.", "",
              "![Validation predictions and matrix error distribution](figures/validation_quality.png)", "",
              f"The median validation matrix RMSE is {np.median(errors):.2f}; the 90th percentile is "
              f"{np.quantile(errors, 0.9):.2f}. Accuracy varies materially across combinations. In the low-error illustrated "
              "case, measured and predicted responses both stay close to the untreated reference. In the high-error and "
              "extreme-error examples, the model misses substantial measured response decreases. These observations describe "
              "the selected diagnostics, not a representative sample of all combinations.", "",
              "The four matrices below were chosen by predefined 10th, 50th, 90th, and 99th percentile ranks of validation "
              "matrix RMSE. This is an error-stratified diagnostic view, not a random sample or a claim that every prediction "
              "looks like the low-error example. Measurements are not clipped. Response color scales are shared, residual "
              "scales are symmetric, and dose labels are in micromolar. Cell positions are equally spaced measured dose levels; "
              "the layout does not imply linear concentration spacing.", "",
              "![Measured, predicted, and residual dose-response matrices](figures/validation_matrices.png)", "",
              "## Parameter diagnostics and seed sensitivity", "",
              "The validation predictions were also inspected at the parameter heads, without changing the model or selecting "
              "a new configuration. The table gives the percentage of 1,000 validation examples with an exactly zero output. "
              "Only the two Hill slopes and interaction parameter are shown; the unused sigma head is excluded.", "",
              "| Seed | Zero Hill slope h1 | Zero Hill slope h2 | Zero interaction alpha |", "| --- | ---: | ---: | ---: |"]
    for seed in protocol["seeds"]:
        fractions = [heads[str(seed)][name]["zero_fraction"] for name in ("h1_mean", "h2_mean", "alpha_mean")]
        lines.append(f"| {seed} | {fractions[0]:.1%} | {fractions[1]:.1%} | {fractions[2]:.1%} |")
    all_zero_h1 = [seed for seed in protocol["seeds"] if heads[str(seed)]["h1_mean"]["zero_fraction"] == 1]
    all_zero_joint = [seed for seed in protocol["seeds"] if all(heads[str(seed)][key]["zero_fraction"] == 1
                       for key in ("h1_mean", "h2_mean", "alpha_mean"))]
    lines += ["", f"The h1 head is zero on every validation example for seeds {all_zero_h1}. "
              f"Both slopes and alpha are all zero for seeds {all_zero_joint}. "
              "This is a material limitation for interpreting the learned pharmacological parameters, even when aggregate "
              "prediction error improves. The diagnosis was checked using the saved trained weights, and its response "
              "predictions exactly matched the previously saved validation predictions.", "",
              "In the released response equation, a zero Hill slope removes that drug's concentration dependence "
              "from the SynBa component; dose-dependent bias heads can still contribute. A zero alpha suppresses the joint "
              "interaction term. These are observed parameter-output diagnostics, not a controlled causal explanation of "
              "performance differences. ReLU activation makes zero outputs possible. Model changes to address this "
              "would constitute a separate experiment and were not made here.", "",
              "## What the results support", "",
              "Longer training modestly improves the matched seed-42 result, but initialization sensitivity remains large. "
              "The seed-43 run is worse than the simple mean-surface baseline. The current evidence supports a working "
              "small-data reproduction, not robust pharmacological interpretation. A useful next controlled experiment "
              "would investigate the zero-valued parameter heads (initialization and positive-output parameterization) "
              "while preserving this run as the released-code reference. Such a modified model should be labeled and "
              "compared separately using validation data.", "",
              "## Protocol and limits", "",
              "- Fixed source archive, subset, and released drug-pair-disjoint split: 3,000 training / 1,000 validation / 1,000 test.",
              "- Training seeds 42, 43, and 44 were fixed before the runs. All three results are reported.",
              "- Model widths, source model code, MSE, AdamW, and learning-rate schedule were retained. The schedule reaches "
              "half its initial rate at epoch 30 and stays there. The epoch budget is 100; early stopping patience is eight.",
              "- Every checkpoint minimizes validation MSE within its seed. Selection hashes were frozen before evaluating tests.",
              "- Three seeds quantify training randomness only. They do not quantify variability from resampling datasets or "
              "establish a confidence interval, significance claim, clinical validity, or replication of the paper's full-data scores.",
              "- This experiment evaluates unseen drug pairs among known drugs and cell lines; unseen-drug/cell and O'Neil "
              "benchmarks, synergy scores, and published baseline methods were not reproduced.",
              "- Test results from the original setup were already visible. This follow-up retained its predeclared configuration "
              "and used validation for checkpoint/model selection; it is not a new untouched benchmark dataset.", "",
              f"Total measured training time: {table.training_seconds.sum():.1f} seconds, excluding process startup, "
              "installation, diagnostic evaluation, and figure generation.", "",
              "## Reproducibility and verification", "",
              "The analysis independently recalculated RMSE and centered-vector/rank correlations from saved numeric predictions. "
              "It checked dose arrays, original row IDs, all validation/test measurements, data checksums, frozen checkpoint hashes, "
              "minimum-validation epoch selection, and exact reproduction of validation predictions after checkpoint reload.", "",
              "- [Machine-readable summary](summary.json) and [per-seed table](seed_results.csv).",
              "- [Experiment protocol](../protocol.json), [frozen model selections](../selection.json), and [completion record](../completed.json).",
              "- [Validation example identities, dose grids, and numeric values](validation_examples.json).",
              "- [Validation parameter-head diagnostics](parameter_diagnostics.json).",
              "- Each seed directory contains its checkpoint, logs, history, environment lock, validation predictions, and final test evaluation.",
              "- PNG and SVG versions of each figure are in `figures/`.", ""]
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({"summary": summary, "selected_seed": int(selected_seed), "report": str(directory / "RESULTS.md")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/extended_100")
    args = parser.parse_args()
    analyze(args.experiment)
