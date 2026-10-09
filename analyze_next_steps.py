"""Independently verify and summarize the frozen confirmation/learning studies."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_extended import assert_metrics, independent_metrics
from next_steps_statistics import paired_cluster_bootstrap
from run_next_steps import (FINALISTS, ROOT, SEEDS, SIZES, load_protocol, pair_set,
                            read_json, validate_file)
from prepare_small_data import sha256, sorted_pair_columns
from train_small import dump_json

LABELS = {"reference": "Reference", "positive_swap": "Positive + augmentation",
          "small_swap": "Compact + augmentation"}
COLORS = {"reference": "#60646c", "positive_swap": "#2166a5", "small_swap": "#be7a12"}
MARKERS = {"reference": "o", "positive_swap": "s", "small_swap": "^"}
MEASURES = ("rmse", "pearson", "spearman", "combination_rmse")


def verified_prediction(folder, recorded, held):
    path = folder / "holdout_predictions.npz"
    validate_file(path, recorded["predictions_sha256"])
    with np.load(path, allow_pickle=False) as saved:
        for key, original in (("source_row", "source_row"), ("dose1", "dose1"),
                              ("dose2", "dose2"), ("observed", "response")):
            np.testing.assert_array_equal(saved[key], held[original].numpy())
        observed, predicted = saved["observed"].copy(), saved["predicted"].copy()
    if predicted.shape != observed.shape:
        raise ValueError("Prediction shape differs from the complete holdout matrices")
    actual = independent_metrics(observed, predicted)
    interior = independent_metrics(observed[:, 1:, 1:], predicted[:, 1:, 1:])
    assert_metrics(actual, recorded["all_16"])
    assert_metrics(interior, recorded["combination_9"])
    return observed, predicted, {**actual, "combination_rmse": interior["rmse"]}


def aggregate(table, include_size=False):
    groups = ["training_matrices", "variant"] if include_size else ["variant"]
    expected = {(size, name) for size in SIZES for name in FINALISTS} if include_size else set(FINALISTS)
    rows, found = [], set()
    for identity, frame in table.groupby(groups, sort=False):
        name = identity[-1] if isinstance(identity, tuple) else identity
        size = identity[0] if include_size else 3000
        group_key = (size, name) if include_size else name
        found.add(group_key)
        if set(frame.seed) != set(SEEDS) or len(frame) != len(SEEDS):
            raise ValueError("Every summary must contain each of the three fixed seeds once")
        row = {"variant": name, "label": LABELS[name], "training_matrices": size,
               "parameters": int(frame.parameters.iloc[0])}
        fields = MEASURES + (("validation_rmse", "training_seconds") if include_size else ())
        for field in fields:
            row[f"{field}_mean"] = float(frame[field].mean())
            row[f"{field}_sd"] = float(frame[field].std(ddof=1))
        rows.append(row)
    if found != expected:
        raise ValueError("Missing frozen model/size groups")
    return pd.DataFrame(rows).sort_values(["training_matrices", "variant"]).reset_index(drop=True)


def verify_confirmation(output, protocol, held):
    completion = read_json(output / "confirmation/completed.json")
    if len(completion["runs"]) != 9:
        raise ValueError("Confirmation must contain all nine frozen inference jobs")
    predictions, records = {}, []
    for item in completion["runs"]:
        name, seed = item["variant"], item["seed"]
        folder = output / "confirmation" / name / f"seed_{seed}"
        frozen = next(row for row in protocol["frozen_checkpoints"]
                      if row["variant"] == name and row["seed"] == seed)
        validate_file(Path(frozen["folder"]) / "best.pt", frozen["checkpoint_sha256"])
        if item["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
            raise ValueError("Confirmation checkpoint differs from its frozen identity")
        recorded = read_json(folder / "metrics.json")
        if recorded != item:
            raise ValueError("Completion and per-run metrics disagree")
        observed, predicted, actual = verified_prediction(folder, item, held)
        if (name, seed) in predictions:
            raise ValueError("Duplicate frozen model/seed")
        predictions[(name, seed)] = predicted
        old = read_json(Path(frozen["folder"]) / "metrics.json")
        records.append({"variant": name, "seed": seed, "parameters": old["parameters"], **actual})
    table = pd.DataFrame(records)
    summary = aggregate(table)
    stack = np.stack([np.stack([predictions[(name, seed)] for seed in SEEDS]) for name in FINALISTS])
    return table, summary, observed, stack


def verify_learning(output, protocol, held):
    completion = read_json(output / "learning/completed.json")
    selection = read_json(output / "learning/selection.json")
    if len(completion["runs"]) != 36 or len(selection["runs"]) != 36:
        raise ValueError("Learning curve must contain 36 fits (27 new, 9 reused)")
    records = []
    for report in completion["runs"]:
        name, seed, size = report["variant"], report["seed"], report["training_matrices"]
        frozen = next(row for row in selection["runs"] if
                      (row["variant"], row["seed"], row["training_matrices"]) == (name, seed, size))
        if {key: value for key, value in report.items() if key != "holdout"} != frozen:
            raise ValueError("Completed run differs from the frozen learning checkpoint selection")
        folder = Path(report["folder"])
        validate_file(folder / "best.pt", report["checkpoint_sha256"])
        validate_file(folder / "val_predictions.npz", report["validation_predictions_sha256"])
        if not report["reused"]:
            validate_file(folder / "history.csv", report["history_sha256"])
        history = pd.read_csv(folder / "history.csv")
        minimum = history.loc[history.val_rmse.idxmin()]
        if int(minimum.epoch) != report["best_epoch"] or len(history) != report["epochs_completed"]:
            raise ValueError("Best epoch is not the minimum-validation checkpoint")
        np.testing.assert_allclose(minimum.val_rmse, report["validation"]["rmse"], rtol=1e-10)
        with np.load(folder / "val_predictions.npz", allow_pickle=False) as validation:
            assert_metrics(independent_metrics(validation["observed"], validation["predicted"]), report["validation"])
        destination = (output / "confirmation" / name / f"seed_{seed}") if report["reused"] else folder
        _, _, actual = verified_prediction(destination, report["holdout"], held)
        records.append({"variant": name, "seed": seed, "training_matrices": size,
            "parameters": report["parameters"], "validation_rmse": report["validation"]["rmse"],
            "training_seconds": report["training_seconds"], "epochs_completed": report["epochs_completed"],
            "best_epoch": report["best_epoch"], "stop_reason": report["stop_reason"],
            "reused": report["reused"], **actual})
    table = pd.DataFrame(records)
    return table, aggregate(table, include_size=True)


def figures(directory, confirmation, bootstrap, learning=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e6e8", "grid.alpha": .7})
    destination = directory / "figures"
    destination.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(destination / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(destination / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    for index, field in enumerate(("rmse", "combination_rmse")):
        ax = axes[index]
        for x, name in enumerate(FINALISTS):
            row = confirmation.loc[confirmation.variant == name].iloc[0]
            ax.errorbar(x, row[f"{field}_mean"], yerr=row[f"{field}_sd"],
                marker=MARKERS[name], color=COLORS[name], capsize=5, markersize=8, linestyle="none")
            ax.annotate(f"{row[f'{field}_mean']:.2f}", (x, row[f"{field}_mean"]),
                        xytext=(0, 15), textcoords="offset points", ha="center")
        ax.set_xticks(range(3), ["Reference", "Positive +\naugmentation", "Compact +\naugmentation"])
        ax.set_ylim(0, 30)
        ax.set_ylabel("RMSE (response units)")
        ax.set_title("All 16 dose entries" if index == 0 else "Nine nonzero-dose entries")
    fig.suptitle("Fresh same-source holdout: 1,000 matrices / 323 new drug pairs\nMean ± sample SD across three fixed training seeds", fontsize=12)
    save(fig, "fresh_confirmation")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    for index, key in enumerate(("all_16", "combination_9")):
        for y, comparison in enumerate(bootstrap[key]["comparisons"]):
            low, high = comparison["percentile_95_ci"]
            point = comparison["mean_seed_rmse_difference"]
            axes[index].hlines(y, low, high, color="#2166a5", linewidth=2)
            axes[index].plot(point, y, marker="o", color="#2166a5")
            axes[index].text(high + .1, y, f"{point:+.2f} [{low:+.2f}, {high:+.2f}]", va="center", fontsize=9)
        axes[index].axvline(0, color="#60646c", linestyle="--")
        axes[index].set_yticks(range(3), ["Positive − reference", "Compact − reference", "Compact − positive"])
        axes[index].set_xlim(-11, 6)
        axes[index].set_xlabel("Difference in mean seed RMSE (response units)")
        axes[index].set_title("All 16 entries" if index == 0 else "Nine combination entries")
    fig.suptitle("Paired differences: 95% drug-pair cluster bootstrap intervals\n5,000 matched resamples; negative differences favor the first model", fontsize=12)
    save(fig, "paired_confirmation")
    if learning is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained")
        for name in FINALISTS:
            rows = learning.loc[learning.variant == name].sort_values("training_matrices")
            for ax, field in zip(axes, ("validation_rmse", "rmse")):
                ax.errorbar(rows.training_matrices, rows[f"{field}_mean"], yerr=rows[f"{field}_sd"],
                    color=COLORS[name], marker=MARKERS[name], capsize=4, label=LABELS[name], linewidth=1.8)
        for ax, title in zip(axes, ("Fixed validation set", "Fresh confirmation set")):
            ax.set_title(title)
            ax.set_xticks(SIZES)
            ax.set_xlabel("Training matrices (one fixed nested subset)")
            ax.set_ylabel("RMSE (response units)")
            ax.set_ylim(0, 32)
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle("Data efficiency: fixed configurations, seeds and validation rules\nMean ± sample SD across three seeds; 3,000-matrix fits reused", fontsize=12)
        save(fig, "learning_curves")


def markdown_table(summary, size=False):
    if size:
        lines = ["| Training matrices | Configuration | Validation RMSE ± SD | Fresh RMSE ± SD | Combination RMSE ± SD |",
                 "|---:|---|---:|---:|---:|"]
        for count in SIZES:
            for name in FINALISTS:
                row = summary.loc[(summary.training_matrices == count) & (summary.variant == name)].iloc[0]
                lines.append(f"| {count} | {LABELS[name]} | {row.validation_rmse_mean:.3f} ± {row.validation_rmse_sd:.3f} | {row.rmse_mean:.3f} ± {row.rmse_sd:.3f} | {row.combination_rmse_mean:.3f} ± {row.combination_rmse_sd:.3f} |")
    else:
        lines = ["| Configuration | Parameters (M) | Fresh RMSE ± SD | Pearson ± SD | Spearman ± SD | Combination RMSE ± SD |",
                 "|---|---:|---:|---:|---:|---:|"]
        for name in FINALISTS:
            row = summary.loc[summary.variant == name].iloc[0]
            lines.append(f"| {LABELS[name]} | {row.parameters/1e6:.2f} | {row.rmse_mean:.3f} ± {row.rmse_sd:.3f} | {row.pearson_mean:.3f} ± {row.pearson_sd:.3f} | {row.spearman_mean:.3f} ± {row.spearman_sd:.3f} | {row.combination_rmse_mean:.3f} ± {row.combination_rmse_sd:.3f} |")
    return "\n".join(lines)


def analyze(output):
    protocol, _, metadata, held = load_protocol(output)
    directory = output / "analysis"
    directory.mkdir(exist_ok=True)
    confirmation, summary, truth, stack = verify_confirmation(output, protocol, held)
    confirmation.to_csv(directory / "confirmation_per_seed.csv", index=False)
    summary.to_csv(directory / "confirmation_summary.csv", index=False)
    frame = pd.read_csv(output / "holdout/metadata.csv")
    _, groups = np.unique(sorted_pair_columns(frame).astype(str), axis=0, return_inverse=True)
    bootstrap = {}
    samples = {}
    for name, observed, predicted in (("all_16", truth, stack),
            ("combination_9", truth[:, 1:, 1:], stack[:, :, :, 1:, 1:])):
        bootstrap[name], samples[name] = paired_cluster_bootstrap(observed, predicted, groups,
            list(FINALISTS), protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
    dump_json(directory / "paired_bootstrap.json", bootstrap)
    np.savez_compressed(directory / "bootstrap_draws.npz", **samples)
    learning_table, learning_summary = None, None
    if (output / "learning/completed.json").exists():
        learning_table, learning_summary = verify_learning(output, protocol, held)
        learning_table.to_csv(directory / "learning_per_seed.csv", index=False)
        learning_summary.to_csv(directory / "learning_summary.csv", index=False)
    figures(directory, summary, bootstrap, learning_summary)
    first = bootstrap["all_16"]["comparisons"][0]
    low, high = first["percentile_95_ci"]
    relative_low, relative_high = first["relative_reduction_percentile_95_ci"]
    lines = ["# DeepSynBa: fresh confirmation and data efficiency", "",
        f"The frozen positive-output-plus-augmentation model reduces mean seed RMSE by **{first['relative_reduction_percent']:.1f}%** on 1,000 fresh matrices from {len(pair_set(frame))} previously unused drug pairs. The paired difference against the reference is **{first['mean_seed_rmse_difference']:+.3f}** response units (95% pair-cluster percentile interval **[{low:+.3f}, {high:+.3f}]**).",
        "", "## Fresh holdout", "",
        "Nine existing fits (three configurations × seeds 42/43/44) were frozen by checkpoint hashes before new response access. All nine reproduced their saved validation predictions exactly on CUDA. No retraining, configuration changes, favorable-seed selection or inference averaging was used for this confirmation.",
        "", markdown_table(summary), "",
        "All ± values above are sample SD across fixed training seeds, not confidence intervals. RMSE and correlations pool the 16,000 dose entries; interior RMSE pools 9,000 nonzero-dose entries. Responses retain the source scale, with untreated reference 100; dose vectors are micromolar.",
        "", "![Fresh confirmation](figures/fresh_confirmation.png)", "",
        f"The candidate pool contains {protocol['eligible_pool']['matrices']:,} matrices / {protocol['eligible_pool']['pairs']} pairs in the released test partition. All {protocol['previously_seen_pairs']:,} unordered pairs in the previous train/validation/test subset were excluded, including reversed drug order. The selected sample covers {protocol['holdout']['drugs']} drugs and {protocol['holdout']['cells']} cells. Pair overlap and row overlap with the previous 5,000 matrices are both zero.",
        "", "## Paired sampling uncertainty", "",
        "| Endpoint | Comparison | Difference in mean seed RMSE | Marginal 95% interval |",
        "|---|---|---:|---:|"]
    for key, description in (("all_16", "All 16 entries"), ("combination_9", "Combination 9")):
        for comparison in bootstrap[key]["comparisons"]:
            a, b = comparison["percentile_95_ci"]
            lines.append(f"| {description} | {LABELS[comparison['model']]} − {LABELS[comparison['baseline']]} | {comparison['mean_seed_rmse_difference']:+.3f} | [{a:+.3f}, {b:+.3f}] |")
    lines += ["", f"The primary relative reduction is {first['relative_reduction_percent']:.1f}% with a marginal 95% percentile interval [{relative_low:.1f}%, {relative_high:.1f}%]. Intervals use 5,000 paired resamples of {len(pair_set(frame))} drug-pair clusters, retaining all selected cells and dose entries within each pair. Each resample uses the same cluster multiplicities for every model and seed. The statistic is the mean of seed-specific RMSEs, not an ensemble RMSE and not the square root of an average over seeds.",
        "", "Intervals condition on these fixed fits and assume independent pair clusters. They do not include training-subset uncertainty, seed-population uncertainty or all dependence from shared drugs/cells. Intervals are marginal, with no familywise correction; the compact/full comparison is descriptive. This is additional same-source unseen-pair evidence; external generalization, synergy accuracy, and biological parameter validity remain untested.",
        "", "![Paired confirmation](figures/paired_confirmation.png)", "", "## Nested data-efficiency study", ""]
    if learning_summary is None:
        lines += ["Training is still in progress. The fixed protocol specifies 27 new fits and reuses the nine frozen 3,000-matrix fits. Learning-curve outcomes have not yet been summarized."]
    else:
        completion = read_json(output / "learning/completed.json")
        capped = sum(not row["reused"] and row["stop_reason"] == "epoch_limit" for row in completion["runs"])
        lines += ["All 27 new fits are complete, with nine existing 3,000-matrix fits reused: 36 data-size/configuration/seed conditions. One nested random ordering of the original training subset defines 500, 1,000, 2,000 and 3,000 complete matrices. The same 1,000 validation matrices, AdamW settings, linear schedule, augmentation, batch size 128, 100-epoch cap, and patience eight apply throughout. Smaller samples therefore also receive fewer gradient steps under the common epoch cap.",
            "", markdown_table(learning_summary, size=True), "",
            "![Learning curves](figures/learning_curves.png)", "",
            f"The curve describes this one nested subset, three seeds, and fixed compute protocol. {capped} of 27 new fits reached the 100-epoch cap; longer training could change the smaller-data results. It does not establish a population-wide minimum sufficient data size or a tuned best configuration. The fresh holdout was not used for early stopping or configuration/seed selection. All 27 new checkpoints were selected on validation and frozen together before their holdout inference.",
            "", "| Training matrices | Distinct pairs | Drugs | Cells | Validation drugs absent from training |",
            "|---:|---:|---:|---:|---|"]
        for size in SIZES:
            subset = protocol["nested_subsets"][str(size)]
            cov = subset["coverage"]
            lines.append(f"| {size} | {cov['pairs']} | {cov['drugs']} | {cov['cells']} | {', '.join(subset['validation_drugs_absent_from_training']) or 'None'} |")
        lines += ["", "All sizes cover all 60 cell lines. The 500-matrix training set omits Pemetrexed Disodium, which is present in validation and the fresh holdout; that point therefore mixes an unseen-drug case with unseen pairs. Larger sizes cover all 101 drugs. The smallest subset has fewer distinct pairs as well as fewer matrices.",
            "", f"Measured new training time totals {completion['new_training_seconds']/60:.1f} minutes, including epoch validation/checkpoint writing and excluding installation, preparation and final inference. Timings on a shared laptop are descriptive. The 3,000-matrix timings come from the previous experiment and should not be treated as a controlled speed comparison. Per-run epochs, caps, best epochs, timings and full metrics are in `learning_per_seed.csv`."]
    lines += ["", "## Verification and reproducibility", "",
        "The analyzer independently recomputed RMSE by vector norms, Pearson by centered dot products, and Spearman by rank correlations. It checked numeric targets, concentrations, row identities, prediction hashes, all frozen checkpoint identities, all seed groups, and (when complete) each history's minimum-validation epoch and saved validation outputs. The software tests cover whole-pair exclusion, nested sampling, hash tampering, unequal cluster sizes, paired resampling, mean-seed versus ensemble RMSE, and invalid numeric inputs.",
        "", "The previous project and its results are read-only. This worktree restores its workflow files and shape fixes, preserving their saved source fingerprints. New artifacts live only under this experiment directory. No packages, external data or cloud logging were added.",
        "", "From this worktree, using an available CUDA Python environment:", "", "```powershell",
        "python run_next_steps.py --stage prepare --previous-root E:\\PROJECTS\\UNI\\DeepSynBa --output runs/next_repeat",
        "python run_next_steps.py --stage confirm --output runs/next_repeat",
        "python run_next_steps.py --stage learning --output runs/next_repeat",
        "python analyze_next_steps.py --experiment runs/next_repeat",
        "python verify_next_steps.py --experiment runs/next_repeat",
        "python -m unittest discover -s tests -v", "```", "",
        "Preparation rejects an existing output directory. Learning may skip fully completed, hash-verified jobs after an interruption; it does not resume optimizer state from a partial job. Variant checkpoints require `ablation_model.build_model`, including their recorded activation treatment. Holdout inference is a separate audited path; the original training-data checksum guard is preserved.",
        "", "Artifacts: `../protocol.json`, `../protocol.sha256`, `../holdout/metadata.csv`, `../holdout/manifest.json`, per-seed numeric prediction files, `confirmation_per_seed.csv`, `confirmation_summary.csv`, `paired_bootstrap.json`, `bootstrap_draws.npz`, and (when complete) `learning_per_seed.csv` / `learning_summary.csv`. All figures are exported in PNG and SVG.",
        "", "Method references: [cluster bootstrap research](https://pmc.ncbi.nlm.nih.gov/articles/PMC5965657/), [paired percentile bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).", ""]
    if (directory / "source_verification.json").exists():
        verification = read_json(directory / "source_verification.json")
        if verification["protocol_sha256"] != sha256(output / "protocol.json"):
            raise ValueError("Source audit belongs to a different experiment")
        for field in ("all_response_values_exact", "all_micromolar_doses_exact", "all_drug_features_exact",
                      "all_cell_features_exact", "bootstrap_arithmetic_matches_explicit_cluster_expansion"):
            if verification[field] is not True:
                raise ValueError("The independent source audit did not pass")
        lines += ["## Source audit", "",
            f"**Ready within reviewed scope:** all {verification['holdout_source_rows_checked']:,} holdout identities, response matrices, dose vectors and both drug/cell features match the checksum-verified source archive exactly. An independent check expanded {verification['bootstrap_draws_independently_expanded_per_endpoint']} actual bootstrap draws for each endpoint and matched the optimized calculation. See `source_verification.json`. This confirms the recorded inputs and arithmetic within this protocol; the scientific limits above still apply.", ""]
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    dump_json(directory / "summary.json", {"confirmation": summary.to_dict(orient="records"),
        "paired_bootstrap": bootstrap, "learning_complete": learning_summary is not None,
        "learning": learning_summary.to_dict(orient="records") if learning_summary is not None else None,
        "protocol_sha256": sha256(output / "protocol.json"),
        "analysis_source_sha256": {name: sha256(ROOT / name) for name in ("analyze_next_steps.py", "next_steps_statistics.py")}})
    print(summary[["label", "rmse_mean", "rmse_sd", "combination_rmse_mean"]].to_string(index=False))
    print(f"Primary paired RMSE difference: {first['mean_seed_rmse_difference']:+.3f} [{low:+.3f}, {high:+.3f}]")
    if learning_summary is not None:
        print(learning_summary[["training_matrices", "label", "validation_rmse_mean", "rmse_mean", "rmse_sd"]].to_string(index=False))
    print(f"Verified results: {directory / 'RESULTS.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/next_steps_20261001")
    analyze(parser.parse_args().experiment.resolve())
