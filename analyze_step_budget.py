"""Audit and report the validation-only optimizer-update budget experiment."""
import argparse
import gc
from pathlib import Path

from train_small import dump_json, predict, seed_everything
import numpy as np
import pandas as pd
import torch

from ablation_model import build_model
from analyze_extended import assert_metrics, independent_metrics
from prepare_small_data import sha256
from run_step_budget import (ROOT, SEEDS, SIZES, VARIANTS, load_protocol,
                             make_loaders, read_json, validate_file, verify_run_files)

LABELS = {"positive_swap": "Enhanced", "small_swap": "Compact"}
COLORS = {"positive_swap": "#2166a5", "small_swap": "#b36b13"}
MEASURES = ("rmse", "combination_rmse", "pearson", "spearman", "updates",
            "best_update", "examples_seen", "training_seconds", "peak_cuda_memory_mib")


def summarize(table):
    expected = {(method, size, name) for method in ("epoch_budget", "update_budget")
                for size in SIZES for name in VARIANTS}
    found, rows = set(), []
    for key, frame in table.groupby(["method", "training_matrices", "variant"], sort=False):
        found.add(key)
        if set(frame.seed) != set(SEEDS) or len(frame) != len(SEEDS):
            raise ValueError("Every condition must contain the three distinct frozen seeds")
        if frame.parameters.nunique() != 1:
            raise ValueError("Architecture changed between seeds")
        row = dict(zip(("method", "training_matrices", "variant"), key))
        row["parameters"] = int(frame.parameters.iloc[0])
        row["limit_reached"] = int((frame.stop_reason.isin(["epoch_limit", "update_limit"])).sum())
        for field in MEASURES:
            row[f"{field}_mean"] = float(frame[field].mean())
            row[f"{field}_sd"] = float(frame[field].std(ddof=1))
        rows.append(row)
    if found != expected:
        raise ValueError("Missing frozen model/size/budget conditions")
    return pd.DataFrame(rows).sort_values(["training_matrices", "variant", "method"]).reset_index(drop=True)


def paired_changes(table):
    keys = ["training_matrices", "variant", "seed"]
    old = table.loc[table.method == "epoch_budget"]
    new = table.loc[table.method == "update_budget"]
    paired = new.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    if len(paired) != 18:
        raise ValueError("Expected 18 complete paired-seed comparisons")
    for field in ("rmse", "combination_rmse", "updates", "training_seconds"):
        paired[f"{field}_change"] = paired[f"{field}_new"] - paired[f"{field}_old"]
    rows = []
    for (size, name), frame in paired.groupby(["training_matrices", "variant"], sort=True):
        row = {"training_matrices": size, "variant": name}
        for field in ("rmse", "combination_rmse", "updates", "training_seconds"):
            row[f"{field}_change_mean"] = float(frame[f"{field}_change"].mean())
            row[f"{field}_change_sd"] = float(frame[f"{field}_change"].std(ddof=1))
        row["relative_rmse_change_percent"] = float(100 * (frame.rmse_new.mean() / frame.rmse_old.mean() - 1))
        rows.append(row)
    return paired, pd.DataFrame(rows)


def verify_history(folder, record, settings):
    history = pd.read_csv(folder / "history.csv")
    interval, limit = settings["validation_interval"], settings["max_updates"]
    final = int(record["updates_completed"])
    np.testing.assert_array_equal(history["update"], np.arange(interval, final + 1, interval))
    if final > limit or final < interval:
        raise ValueError("Invalid actual update count")
    batches = int(record["batches_per_epoch"])
    size = int(record["training_matrices"])
    np.testing.assert_array_equal(history.epoch, (history["update"] - 1) // batches + 1)
    np.testing.assert_array_equal(history.batch_in_epoch, (history["update"] - 1) % batches + 1)
    np.testing.assert_allclose(history.epoch_equivalent, history["update"] / batches, rtol=0, atol=1e-12)
    exposure = history["update"] // batches * size + history["update"] % batches * settings["batch_size"]
    np.testing.assert_array_equal(history.examples_seen, exposure)
    rates = settings["optimizer"]["lr"] * (1 - .5 * np.minimum(np.arange(len(history)), 30) / 30)
    np.testing.assert_allclose(history.lr, rates, rtol=0, atol=1e-17)
    best, stale = float("inf"), 0
    for row in history.itertuples(index=False):
        improved = row.val_rmse < best
        if improved:
            best, stale = row.val_rmse, 0
        else:
            stale += 1
        if bool(row.improved) != improved or row.stale_evaluations != stale:
            raise ValueError("Checkpoint improvement/stopping counters disagree with validation")
        if stale >= settings["patience_evaluations"] and row.update != final:
            raise ValueError("Training continued beyond its stopping point")
    minimum = history.loc[history.val_rmse.idxmin()]
    if int(minimum["update"]) != record["best_update"]:
        raise ValueError("Checkpoint does not minimize scheduled validation RMSE")
    np.testing.assert_allclose(minimum.val_rmse ** 2, record["best_validation_mse"], rtol=1e-12)
    if int(exposure.iloc[-1]) != record["examples_seen"] or int(minimum.examples_seen) != record["best_examples_seen"]:
        raise ValueError("Recorded example exposure differs from independently counted batches")
    if record["stop_reason"] == "early_stopping":
        if stale != settings["patience_evaluations"] or final - record["best_update"] != settings["patience_updates"]:
            raise ValueError("Patience did not use the frozen update units")
    elif record["stop_reason"] != "update_limit" or final != limit or stale >= settings["patience_evaluations"]:
        raise ValueError("Stop reason does not match the history")
    return history


def verify_prediction(record, arrays, *, new):
    folder = Path(record["folder"])
    verify_run_files(record)
    rows = torch.where(arrays["split"] == 2)[0]
    with np.load(folder / "val_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_row"], arrays["source_row"][rows].numpy())
        np.testing.assert_array_equal(saved["observed"], arrays["response"][rows].numpy())
        if saved["predicted"].shape != (1000, 4, 4):
            raise ValueError("Validation predictions are not 1000 complete dose matrices")
        if new:
            for key in ("dose1", "dose2"):
                np.testing.assert_array_equal(saved[key], arrays[key][rows].numpy())
        observed, predicted = saved["observed"].copy(), saved["predicted"].copy()
    full = independent_metrics(observed, predicted)
    combination = independent_metrics(observed[:, 1:, 1:], predicted[:, 1:, 1:])
    assert_metrics(full, record["validation"]["all_16"] if new else record["validation"])
    if new:
        assert_metrics(combination, record["validation"]["combination_9"])
    return {**full, "combination_rmse": combination["rmse"]}, predicted


def verify_checkpoint(record, protocol, arrays):
    seed_everything(record["seed"], 8)
    folder = Path(record["folder"])
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    for field in ("seed", "dataset_sha256", "protocol_sha256", "best_update"):
        if checkpoint[field] != record[field]:
            raise ValueError(f"Checkpoint provenance differs on {field}")
    if checkpoint["variant"]["id"] != record["variant"]:
        raise ValueError("Checkpoint model variant differs")
    if checkpoint["training_source_rows"] != protocol["nested_subsets"][str(record["training_matrices"])]["source_rows"]:
        raise ValueError("Checkpoint training rows differ from the frozen nested subset")
    model, _ = build_model(checkpoint["model_config"], checkpoint["variant"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to("cuda").eval()
    if sum(parameter.numel() for parameter in model.parameters()) != record["parameters"]:
        raise ValueError("Parameter count differs from the saved model")
    _, predicted = predict(model, make_loaders(arrays, record["seed"])["val"], torch.device("cuda"))
    with np.load(folder / "val_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(predicted, saved["predicted"])
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def audit(output, reload_checkpoints):
    protocol, arrays, metadata = load_protocol(output)
    completion, selection = read_json(output / "completed.json"), read_json(output / "selection.json")
    validate_file(output / "selection.json", completion["selection_sha256"])
    if completion["runs"] != selection["runs"] or completion["new_training_jobs"] != 18:
        raise ValueError("Completion does not freeze every new fit")
    expected = {(size, name, seed) for size in SIZES for name in VARIANTS for seed in SEEDS}
    identity = lambda record: (record["training_matrices"], record["variant"], record["seed"])
    for records in (protocol["baseline_runs"], completion["runs"]):
        if len(records) != 18 or {identity(record) for record in records} != expected:
            raise ValueError("Duplicate or missing fit identities")
    if list(output.glob("**/*holdout*")) or list(output.glob("**/test_predictions.npz")):
        raise ValueError("This validation-only experiment unexpectedly contains holdout inference")
    histories, records, reloads = {}, [], []
    settings = protocol["hyperparameters"]
    for current in completion["runs"]:
        size, name, seed = identity(current)
        folder = Path(current["folder"])
        if read_json(folder / "metrics.json") != current:
            raise ValueError("Per-fit metrics differ from the frozen selection")
        current_metrics, prediction = verify_prediction(current, arrays, new=True)
        history = verify_history(folder, current, settings)
        histories[("update_budget", size, name, seed)] = history
        baseline = next(record for record in protocol["baseline_runs"] if identity(record) == identity(current))
        old_metrics, old_prediction = verify_prediction(baseline, arrays, new=False)
        old_history = pd.read_csv(Path(baseline["folder"]) / "history.csv")
        minimum = old_history.loc[old_history.val_rmse.idxmin()]
        if baseline["best_epoch"] != int(minimum.epoch) or len(old_history) != baseline["epochs_completed"]:
            raise ValueError("Earlier budget history/checkpoint mismatch")
        np.testing.assert_allclose(minimum.val_rmse, old_metrics["rmse"], rtol=1e-12)
        histories[("epoch_budget", size, name, seed)] = old_history
        if size == 3000:
            if current["control_3000_exactly_reproduced"] is not True:
                raise ValueError("The unchanged 3000-matrix control was not verified")
            np.testing.assert_array_equal(prediction, old_prediction)
            np.testing.assert_allclose(history[["train_rmse", "val_rmse", "lr"]],
                old_history[["train_rmse", "val_rmse", "lr"]], rtol=0, atol=1e-10)
        else:
            if current["control_3000_exactly_reproduced"] is not None:
                raise ValueError("A changed lower-data fit was marked as an unchanged control")
        for row, method, computed in ((current, "update_budget", current_metrics),
                                     (baseline, "epoch_budget", old_metrics)):
            new = method == "update_budget"
            batches = (size + settings["batch_size"] - 1) // settings["batch_size"]
            records.append({"method": method, "training_matrices": size, "variant": name, "seed": seed,
                "parameters": row["parameters"], **computed,
                "updates": row["updates_completed"] if new else row["epochs_completed"] * batches,
                "best_update": row["best_update"] if new else row["best_epoch"] * batches,
                "examples_seen": row["examples_seen"] if new else row["epochs_completed"] * size,
                "training_seconds": row["training_seconds"], "peak_cuda_memory_mib": row["peak_cuda_memory_mib"],
                "stop_reason": row["stop_reason"]})
        if reload_checkpoints:
            verify_checkpoint(current, protocol, arrays)
            reloads.append({"training_matrices": size, "variant": name, "seed": seed, "validation_reload_exact": True})
            print(f"VERIFIED n={size} {name} seed={seed}: exact CUDA validation reload", flush=True)
    table = pd.DataFrame(records).sort_values(["training_matrices", "variant", "seed", "method"]).reset_index(drop=True)
    total_updates = int(table.loc[table.method == "update_budget", "updates"].sum())
    if total_updates != completion["total_optimizer_updates"]:
        raise ValueError("Total update accounting differs")
    np.testing.assert_allclose(table.loc[table.method == "update_budget", "training_seconds"].sum(),
                               completion["new_training_seconds"], rtol=1e-12)
    val_rows = torch.where(arrays["split"] == 2)[0]
    for key in ("dose1", "dose2"):
        dose = arrays[key][val_rows].numpy()
        if not (dose[:, 0] == 0).all() or not (dose[:, 1:] > 0).all():
            raise ValueError("Combination-9 must have positive doses on both axes")
    for size in SIZES:
        rows = protocol["nested_subsets"][str(size)]["array_rows"]
        np.testing.assert_array_equal(arrays["source_row"][rows].numpy(),
                                      protocol["nested_subsets"][str(size)]["source_rows"])
        if not (metadata.iloc[rows].split == 1).all():
            raise ValueError("Non-training row in a fitted subset")
    return protocol, completion, selection, table, histories, reloads


def figures(directory, summary, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e5e7eb", "grid.alpha": .7, "figure.facecolor": "white"})
    destination = directory / "figures"
    destination.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(destination / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(destination / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    for ax, field, title in zip(axes, ("rmse", "combination_rmse"), ("All 16 dose entries", "9 combination entries")):
        for name in VARIANTS:
            for method, linestyle, marker in (("epoch_budget", "--", "o"), ("update_budget", "-", "s")):
                data = summary.loc[(summary.variant == name) & (summary.method == method)].sort_values("training_matrices")
                ax.errorbar(data.training_matrices, data[f"{field}_mean"], yerr=data[f"{field}_sd"],
                    color=COLORS[name], linestyle=linestyle, marker=marker, capsize=4, linewidth=1.7,
                    label=f"{LABELS[name]} · {'100 epochs' if method == 'epoch_budget' else '2,400 updates'}")
        ax.set_title(title)
        ax.set_xticks(SIZES, ["1,000", "2,000", "3,000"])
        ax.set_xlabel("Training matrices")
        ax.set_ylabel("Validation RMSE (response units)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .055))
    fig.suptitle("Validation performance with equal update limits", fontsize=15, x=.07, ha="left")
    fig.text(.07, .012, "Mean ± sample SD of seeds 42/43/44; one fixed nested split. At 3,000 matrices the training rule is unchanged.", fontsize=9)
    fig.tight_layout(rect=(0, .2, 1, .93))
    save(fig, "validation_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
    for ax, field, title, ylabel in zip(axes, ("updates", "best_update", "training_seconds"),
            ("Realized optimizer updates", "Validation-selected checkpoint", "Training time per seed"),
            ("Updates completed", "Best update", "Seconds (includes validation and saving)")):
        for name in VARIANTS:
            data = summary.loc[(summary.variant == name) & (summary.method == "update_budget")].sort_values("training_matrices")
            ax.errorbar(data.training_matrices, data[f"{field}_mean"], yerr=data[f"{field}_sd"],
                color=COLORS[name], marker="s", capsize=4, label=LABELS[name])
        if field != "training_seconds":
            ax.axhline(2400, color="#7b818a", linestyle=":", label="2,400-update limit")
            ax.set_ylim(0, 2650)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_xlabel("Training matrices")
        ax.set_ylabel(ylabel)
        ax.set_xticks(SIZES, ["1,000", "2,000", "3,000"])
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Compute used under the shared maximum budget", fontsize=15, x=.055, ha="left")
    fig.text(.055, .01, "Mean ± sample SD of three seeds. Early stopping can reduce realized updates. Wall times are descriptive laptop measurements.", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    save(fig, "compute_budget")

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), sharex=True, sharey=True)
    seed_colors = {42: "#2166a5", 43: "#ad6916", 44: "#754a98"}
    for index, name in enumerate(VARIANTS):
        for column, size in enumerate(SIZES):
            ax = axes[index, column]
            for seed in SEEDS:
                current = histories[("update_budget", size, name, seed)]
                prior = histories[("epoch_budget", size, name, seed)]
                batches = (size + 127) // 128
                ax.plot(prior.epoch * batches, prior.val_rmse, color=seed_colors[seed], linestyle="--", linewidth=1.1, alpha=.7)
                ax.plot(current["update"], current.val_rmse, color=seed_colors[seed], linewidth=1.4)
                minimum = current.loc[current.val_rmse.idxmin()]
                ax.scatter(minimum["update"], minimum.val_rmse, color=seed_colors[seed], marker="*", s=60, zorder=3)
            ax.axvline(720, color="#a6aab0", linestyle=":", linewidth=1)
            ax.set_title(f"{LABELS[name]} · {size:,} matrices")
            ax.set_xlim(0, 2450)
            ax.set_xticks([0, 800, 1600, 2400], ["0", "800", "1,600", "2,400"])
            if column == 0:
                ax.set_ylabel("Validation RMSE (response units)")
            if index == 1:
                ax.set_xlabel("Optimizer updates")
    legend = [Line2D([0], [0], color=seed_colors[seed], label=f"Seed {seed}") for seed in SEEDS]
    legend += [Line2D([0], [0], color="#555b65", linestyle="--", label="Earlier epoch budget"),
               Line2D([0], [0], color="#555b65", label="Shared update budget")]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .035))
    fig.suptitle("Validation trajectories on matched training subsets", fontsize=15, x=.065, ha="left")
    fig.text(.065, .006, "Stars: new validation-selected checkpoints. Vertical dots: new learning-rate decay finishes at update 720. Three-thousand-row controls overlap.", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .95))
    save(fig, "training_trajectories")


def report(directory, protocol, completion, selection, table, summary, changes, reloads):
    def group(method, size, name):
        return summary.loc[(summary.method == method) & (summary.training_matrices == size) & (summary.variant == name)].iloc[0]

    def mean_sd(row, field):
        return f"{row[f'{field}_mean']:.3f} ± {row[f'{field}_sd']:.3f}"

    winner = selection["selected_configuration"]
    compact = min((item for item in selection["ranking"] if item["variant"] == "small_swap"),
                  key=lambda item: item["validation_rmse_mean"])
    new = table.loc[table.method == "update_budget"]
    lines = ["# Equal optimizer-update budget: validation results", "",
        f"All **18 new fits** are complete: two fixed models × 1,000/2,000/3,000 matrices × seeds 42/43/44. "
        f"The shared maximum is 2,400 optimizer updates; {int(completion['total_optimizer_updates']):,} updates "
        f"were actually used, with {completion['new_training_seconds']/60:.1f} minutes of summed training time.", "",
        f"The validation accuracy winner is **{LABELS[winner['variant']]} with {winner['training_matrices']:,} matrices**, "
        f"RMSE **{winner['validation_rmse_mean']:.3f} ± {winner['validation_rmse_sd']:.3f}**. "
        f"The best compact configuration uses **{compact['training_matrices']:,} matrices**, "
        f"RMSE **{compact['validation_rmse_mean']:.3f} ± {compact['validation_rmse_sd']:.3f}**. "
        "Every ± value is sample SD across three seeds on one fixed split, not a confidence interval.", "",
        "## Matched accuracy comparison", "",
        "Negative changes favor the update-budget run. The earlier runs had a maximum of 100 epochs, "
        "corresponding to 800/1,600/2,400 updates at the three sizes. Combination entries exclude the zero-dose row and column.", "",
        "| Model | Training matrices | Earlier all-16 RMSE | New all-16 RMSE | Paired change | Earlier combination-9 RMSE | New combination-9 RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for size in SIZES:
        for name in VARIANTS:
            old, current = group("epoch_budget", size, name), group("update_budget", size, name)
            change = changes.loc[(changes.training_matrices == size) & (changes.variant == name)].iloc[0]
            lines.append(f"| {LABELS[name]} | {size:,} | {mean_sd(old, 'rmse')} | {mean_sd(current, 'rmse')} | "
                         f"{change.rmse_change_mean:+.3f} ± {change.rmse_change_sd:.3f} | "
                         f"{mean_sd(old, 'combination_rmse')} | {mean_sd(current, 'combination_rmse')} |")
    lines += ["", "![Validation comparison](figures/validation_comparison.png)", "", "## Compute and stopping", "",
        "| Model | Matrices | Parameters | Mean updates completed | Mean best update | Fits at the limit | Mean training seconds | Peak allocated CUDA MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for size in SIZES:
        for name in VARIANTS:
            current = group("update_budget", size, name)
            lines.append(f"| {LABELS[name]} | {size:,} | {int(current.parameters):,} | {current.updates_mean:.0f} | "
                         f"{current.best_update_mean:.0f} | {int(current.limit_reached)}/3 | "
                         f"{current.training_seconds_mean:.1f} | {current.peak_cuda_memory_mib_mean:.1f} |")
    lines += ["", "![Compute](figures/compute_budget.png)", "",
        f"{int((new.stop_reason == 'update_limit').sum())} of 18 fits reached 2,400 updates; "
        f"{int((new.stop_reason == 'early_stopping').sum())} stopped earlier after eight failed validation checks. "
        "Stopping is based on all-16 validation error. Combination-only error is reported at that same checkpoint. "
        "The maximum epochs equivalent are 300/150/100; a 2,000-matrix run can stop halfway through an epoch.", "",
        "Times include optimizer work, validation and checkpoint saving. The earlier times and these times were measured "
        "in different sessions, with some different diagnostic overhead and uncontrolled laptop thermal/load conditions. "
        "Use update counts and parameter counts for the controlled compute comparison; treat wall times as descriptive.", "",
        "## What this resolves", ""]
    for name in VARIANTS:
        one, three = group("update_budget", 1000, name), group("update_budget", 3000, name)
        gap = one.rmse_mean - three.rmse_mean
        change = changes.loc[(changes.training_matrices == 1000) & (changes.variant == name)].iloc[0]
        lines.append(f"For {LABELS[name].lower()}, the 1,000-matrix validation RMSE changes by "
            f"{change.rmse_change_mean:+.3f} ({change.relative_rmse_change_percent:+.1f}%) under the update-based rule. "
            f"Its remaining mean gap to the 3,000-matrix setting is {gap:+.3f} response units.")
        lines.append("")
        two = group("update_budget", 2000, name)
        change_two = changes.loc[(changes.training_matrices == 2000) & (changes.variant == name)].iloc[0]
        lines.append(f"At 2,000 matrices, {LABELS[name].lower()} validation RMSE changes by "
            f"{change_two.rmse_change_mean:+.3f} ({change_two.relative_rmse_change_percent:+.1f}%). "
            f"Using one third fewer training matrices leaves a mean gap of "
            f"{two.rmse_mean - three.rmse_mean:+.3f} response units to its 3,000-matrix setting.")
        lines.append("")
    lines += ["At 1,000 and 2,000 matrices, the intervention combines a larger update limit, learning-rate decay "
        "ending at update 720, validation every 24 updates and patience measured as 192 updates. It estimates the effect "
        "of that combined timing rule; it does not isolate extra training duration from each timing change. "
        "Both architecture packages and augmentation remain fixed.", "",
        "All six 3,000-matrix controls exactly reproduced their earlier saved predictions on CUDA, "
        "and training/validation/lr trajectories match to 1e-10 absolute tolerance. Their unchanged results are "
        "a reproducibility check of the new runner, not six additional independent experimental replications.", "",
        "![Trajectories](figures/training_trajectories.png)", "", "## Validation and provenance", "",
        f"The protocol was frozen before training (SHA-256 `{sha256(directory.parent / 'protocol.json')}`). "
        "It fixes source hashes, all 18 earlier checkpoint/prediction/history identities, exact nested training row IDs, "
        "seeds, model definitions, optimizer schedule, endpoints, ranking rule and stopping rule. "
        "Every one of the six configuration groups is complete; the ranking averages all three seeds.", "",
        "Independent verification recomputed all-16 and combination-9 RMSE, Pearson and Spearman for all 36 earlier/new "
        "validation prediction files. Validation source IDs, observed values and available dose arrays match the prepared data. "
        "The dose audit confirms that the interior nine entries have positive doses on both axes. "
        "Batch arithmetic independently confirms actual update counts, examples seen, scheduler boundaries and strict patience.", "",
        ("All 18 new checkpoints were rebuilt from their saved variant/configuration and reproduced their stored validation "
         "predictions exactly on CUDA; checkpoint seed/data/protocol identities and training rows were checked."
         if len(reloads) == 18 else "Checkpoint reload verification was not requested for this analysis run."), "",
        "One wording caveat in the frozen protocol: 'do not load ... holdout' is stricter than the shared "
        "data-loader behavior. The runner reads the existing 5,000-row prepared container, which includes unused "
        "original test rows, to construct its train/validation loaders. Only split 1 enters training and split 2 "
        "enters checkpoint selection. No original test response is evaluated or used for any decision; neither "
        "original test predictions nor the newer inspected holdout dataset/predictions are used here. "
        "No new held-out evaluation was performed.", "",
        "## Scope and next decision", "",
        "These are validation results from repeated development on a single fixed, pair-disjoint split. "
        "The three seeds describe initialization/training variation, not sampling uncertainty across new splits. "
        "The subsets use one shared nested ordering; correlated pairs can share drugs and cell lines. "
        "The study does not establish a minimum sufficient dataset size, generalization accuracy, parameter identifiability "
        "or an isolated activation/initialization effect.", "",
        f"Carry **{LABELS[winner['variant']].lower()} / {winner['training_matrices']:,} matrices** forward as the "
        f"validation accuracy choice, and **compact / {compact['training_matrices']:,} matrices** as the resource candidate. "
        "Before any further tuning, reserve a new evaluation set and freeze the final comparison. The already inspected "
        "holdouts cannot provide an untouched final confirmation after adaptive development. Independent-source data "
        "would provide a stronger next check.", "",
        "## Inspectable artifacts", "",
        "- `../protocol.json`, `../protocol.sha256`: frozen design and source/data identities.",
        "- `../selection.json`, `../completed.json`: all validation-selected checkpoints and completion accounting.",
        "- `validation_per_seed.csv`, `validation_summary.csv`: earlier/new errors and compute for all seeds.",
        "- `paired_changes_per_seed.csv`, `paired_changes_summary.csv`: matched-seed differences.",
        "- `verification.json`: audit/reload results and analysis source fingerprints.",
        "- `test_results.txt`: saved full correctness-suite output for this experiment, when present.",
        "- `figures/`: PNG and SVG figures; per-fit histories/predictions/checkpoints are under `../training/`.", ""]
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/step_budget_20261001")
    parser.add_argument("--verify-checkpoints", action="store_true", help="Reload every new checkpoint on CUDA")
    args = parser.parse_args()
    output = args.experiment.resolve()
    protocol, completion, selection, table, histories, reloads = audit(output, args.verify_checkpoints)
    summary = summarize(table)
    paired, changes = paired_changes(table)
    independent_ranking = summary.loc[summary.method == "update_budget"].sort_values(
        ["rmse_mean", "rmse_sd", "parameters", "training_matrices", "variant"]).reset_index(drop=True)
    for index, frozen in enumerate(selection["ranking"]):
        actual = independent_ranking.iloc[index]
        if (actual.variant, int(actual.training_matrices)) != (frozen["variant"], frozen["training_matrices"]):
            raise ValueError("Frozen ranking does not follow the predeclared validation rule")
        np.testing.assert_allclose([actual.rmse_mean, actual.rmse_sd],
            [frozen["validation_rmse_mean"], frozen["validation_rmse_sd"]], rtol=1e-12)
    if selection["selected_configuration"] != selection["ranking"][0]:
        raise ValueError("The selected configuration is not the top ranked mean validation error")
    directory = output / "analysis"
    directory.mkdir(exist_ok=True)
    for filename, frame in (("validation_per_seed.csv", table), ("validation_summary.csv", summary),
            ("paired_changes_per_seed.csv", paired), ("paired_changes_summary.csv", changes)):
        frame.to_csv(directory / filename, index=False)
    dump_json(directory / "summary.json", {"selected_configuration": selection["selected_configuration"],
        "summary": summary.to_dict(orient="records"), "paired_changes": changes.to_dict(orient="records"),
        "new_training_jobs": 18, "total_optimizer_updates": completion["total_optimizer_updates"],
        "training_seconds": completion["new_training_seconds"]})
    dump_json(directory / "verification.json", {"protocol_sha256": sha256(output / "protocol.json"),
        "selection_sha256": sha256(output / "selection.json"), "conditions": 36, "new_fits": 18,
        "exact_3000_control_reproductions": 6, "independent_metric_checks": 36,
        "update_exposure_scheduler_patience_audit": True, "validation_ranking_verified": True,
        "no_new_holdout_inference": True, "checkpoint_reloads": reloads,
        "correctness_suite_log_sha256": (sha256(directory / "test_results.txt")
                                          if (directory / "test_results.txt").exists() else None),
        "analysis_source_sha256": {name: sha256(ROOT / name) for name in
            ("analyze_step_budget.py", "analyze_extended.py", "tests/test_step_budget.py",
             "tests/test_step_budget_analysis.py")}})
    figures(directory, summary, histories)
    report(directory, protocol, completion, selection, table, summary, changes, reloads)
    print(summary[["method", "training_matrices", "variant", "rmse_mean", "combination_rmse_mean", "updates_mean"]].to_string(index=False))
    print(f"SAVED verified report and figures: {directory}")


if __name__ == "__main__":
    main()
