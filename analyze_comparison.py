"""Independently verify the small-data comparison and export scientific figures."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from analyze_extended import (BLUE, ORANGE, GREY, RESIDUAL_MAP, assert_metrics,
                              choose_cases, independent_metrics, read_json,
                              save_figure, write_json)
import matplotlib.pyplot as plt
from ablation_model import POSITIVE_KEYS
from prepare_small_data import ROOT, SPLITS, sha256

MEASURES = ["validation_rmse", "test_rmse", "test_pearson", "test_spearman", "combination_rmse",
            "validation_combination_rmse", "swap_rmse", "bias_rms", "training_seconds",
            "peak_cuda_memory_mb", "h1_near_zero_pct", "h2_near_zero_pct", "alpha_near_zero_pct"]


def aggregate_results(table, variants):
    rows = []
    for variant in variants:
        subset = table[table.variant == variant["id"]]
        if len(subset) != 3 or set(subset.seed) != {42, 43, 44}:
            raise AssertionError("Every configuration must contain exactly the three predeclared seeds")
        row = {"variant": variant["id"], "label": variant["label"],
               "parameters": int(subset.parameters.iloc[0]), "seeds": len(subset)}
        for key in MEASURES:
            values = subset[key].to_numpy(dtype=np.float64)
            row[key + "_mean"] = float(values.sum() / len(values))
            row[key + "_sd"] = float(np.sqrt(np.sum((values - values.mean()) ** 2) / (len(values) - 1)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["validation_rmse_mean", "validation_rmse_sd", "parameters", "variant"], kind="stable").reset_index(drop=True)


def factorial_contrasts(table, variants):
    lookup = table.set_index(["variant", "seed"])
    cells = variants[:8]
    rows, details = [], []
    for flag, label in (("positive", "Positive-output package"), ("small", "Smaller model"),
                        ("swap", "Drug-order augmentation")):
        others = [name for name in ("positive", "small", "swap") if name != flag]
        pairs = []
        for control in cells:
            if control[flag]:
                continue
            treatment = next(row for row in cells if row[flag] and
                             all(row[key] == control[key] for key in others))
            pairs.append((control["id"], treatment["id"]))
        rows.append(contrast_record(label, pairs, lookup, details))
    for name, label in (("positive_small_swap_bias_l2", "Bias penalty on all-three"),
                        ("positive_small_swap_no_bias", "Remove bias on all-three")):
        rows.append(contrast_record(label, [("positive_small_swap", name)], lookup, details))
    return rows, details


def contrast_record(label, pairs, lookup, details):
    record = {"comparison": label, "matched_backgrounds": len(pairs)}
    for metric in ("validation_rmse", "test_rmse"):
        seed_deltas = []
        for seed in (42, 43, 44):
            deltas = []
            for control, treatment in pairs:
                delta = float(lookup.loc[(treatment, seed), metric] - lookup.loc[(control, seed), metric])
                deltas.append(delta)
                details.append({"comparison": label, "metric": metric, "seed": seed,
                                "control": control, "treatment": treatment, "delta": delta})
            seed_deltas.append(float(np.mean(deltas)))
        record[metric] = {"mean_delta": float(np.mean(seed_deltas)),
                          "sample_sd": float(np.std(seed_deltas, ddof=1)), "seed_deltas": seed_deltas}
    return record


def load_verified_results(experiment, protocol, selection, source):
    rows, histories, file_hashes = [], {}, {}
    labels = {row["id"]: row for row in protocol["variants"]}
    if len(selection["checkpoints"]) != 30:
        raise AssertionError("Expected 30 frozen checkpoint selections")
    frozen_time = datetime.fromisoformat(selection["frozen_utc"])
    for item in selection["checkpoints"]:
        name, seed = item["variant"], item["seed"]
        folder = experiment / name / f"seed_{seed}"
        training, testing = read_json(folder / "metrics.json"), read_json(folder / "test_metrics.json")
        if training["variant"] != labels[name] or training["seed"] != seed or training["test_evaluated"]:
            raise AssertionError("Training protocol or deferred-test status changed")
        if training["dataset_sha256"] != protocol["dataset_sha256"]:
            raise AssertionError("Dataset mismatch")
        if datetime.fromisoformat(testing["evaluated_utc"]) <= frozen_time:
            raise AssertionError("Test inference preceded the frozen selection")
        expected_hashes = {"best.pt": item["checkpoint_sha256"],
                           "val_predictions.npz": item["validation_predictions_sha256"],
                           "test_predictions.npz": testing["test_predictions_sha256"]}
        for filename, expected in expected_hashes.items():
            actual = sha256(folder / filename)
            if actual != expected:
                raise AssertionError(f"Changed frozen artifact: {folder / filename}")
            file_hashes[str((folder / filename).relative_to(experiment))] = actual
        if not testing["validation_reload_exact"]:
            raise AssertionError("Checkpoint validation reload was not verified")
        history = pd.read_csv(folder / "history.csv")
        histories[(name, seed)] = history
        best = history.loc[history.val_rmse.idxmin()]
        if int(best.epoch) != item["epoch"] or int(best.epoch) != training["best_epoch"]:
            raise AssertionError("Checkpoint is not the minimum-validation epoch")
        computed = {}
        for split, recorded in (("val", training["validation"]), ("test", testing["test"])):
            indices = np.flatnonzero(source["split"] == SPLITS[split])
            with np.load(folder / f"{split}_predictions.npz", allow_pickle=False) as saved:
                if saved["predicted"].shape != (1000, 4, 4):
                    raise AssertionError("Wrong prediction grain")
                for key in ("source_row", "dose1", "dose2"):
                    np.testing.assert_array_equal(saved[key], source[key][indices])
                np.testing.assert_array_equal(saved["observed"], source["response"][indices])
                full = independent_metrics(saved["observed"], saved["predicted"])
                interior = independent_metrics(saved["observed"][:, 1:, 1:], saved["predicted"][:, 1:, 1:])
                assert_metrics(full, recorded["all_16"])
                assert_metrics(interior, recorded["combination_9"])
                swap = independent_metrics(saved["predicted"], saved["swapped_transposed"])["rmse"]
                bias = float(np.linalg.norm(saved["bias"].astype(np.float64).ravel()) / np.sqrt(saved["bias"].size))
                np.testing.assert_allclose([swap, bias], [recorded["swap_rmse"], recorded["bias_rms"]], atol=1e-9, rtol=1e-9)
                fractions = []
                for key in POSITIVE_KEYS:
                    values = saved[key].ravel()
                    stats = recorded["parameters"][key]
                    np.testing.assert_allclose([values.min(), values.max(), values.mean(), (values == 0).mean(), (values < .01).mean()],
                                               [stats[k] for k in ("min", "max", "mean", "zero_fraction", "below_0_01_fraction")])
                    fractions.append(float((values < 0.01).mean() * 100))
                if not labels[name]["bias"]:
                    np.testing.assert_array_equal(saved["bias"], 0)
                if name == "reference":
                    previous = ROOT / f"runs/extended_100/seed_{seed}/evaluation/{split}_predictions.npz"
                    with np.load(previous, allow_pickle=False) as original:
                        np.testing.assert_array_equal(saved["predicted"], original["predicted"])
                computed[split] = dict(full=full, interior=interior, swap=swap, bias=bias, fractions=fractions)
        val, test = computed["val"], computed["test"]
        np.testing.assert_allclose(val["full"]["rmse"], [best.val_rmse, item["validation_rmse"]], rtol=1e-10, atol=1e-10)
        rows.append({"variant": name, "seed": seed, "best_epoch": training["best_epoch"],
                     "epochs_completed": training["epochs_completed"], "stop_reason": training["stop_reason"],
                     "parameters": training["parameters"], "train_eval_rmse": training["train_eval"]["rmse"],
                     "validation_rmse": val["full"]["rmse"], "test_rmse": test["full"]["rmse"],
                     "validation_combination_rmse": val["interior"]["rmse"],
                     "test_pearson": test["full"]["pearson"], "test_spearman": test["full"]["spearman"],
                     "combination_rmse": test["interior"]["rmse"], "swap_rmse": val["swap"], "bias_rms": val["bias"],
                     "h1_near_zero_pct": val["fractions"][0], "h2_near_zero_pct": val["fractions"][1],
                     "alpha_near_zero_pct": val["fractions"][2], "training_seconds": training["training_seconds"],
                     "peak_cuda_memory_mb": training["peak_cuda_memory_mb"],
                     **{f"{key}_gradient_norm_at_best_epoch": float(best[f"{key}_gradient_norm"]) for key in POSITIVE_KEYS}})
    return pd.DataFrame(rows), histories, file_hashes


def comparison_plot(directory, summary, table, baselines):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True, sharex=True)
    positions = np.arange(len(summary))
    for ax, key, title, split in zip(axes, ("validation_rmse", "test_rmse"),
                                    ("Validation (selection metric)", "Test (exploratory confirmation)"), ("val", "test")):
        for y, row in enumerate(summary.itertuples()):
            group = table[table.variant == row.variant].sort_values("seed")
            for offset, value, marker in zip((-.15, 0, .15), group[key], ("o", "s", "^")):
                ax.scatter(value, y + offset, color=BLUE, marker=marker, s=26, zorder=4)
            mean, sd = getattr(row, key + "_mean"), getattr(row, key + "_sd")
            ax.errorbar(mean, y, xerr=sd, fmt="D", color=ORANGE, markersize=5, capsize=4, zorder=5)
            ax.text(.98, y, f"{mean:.2f} ± {sd:.2f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.axvline(baselines[split]["rmse"], color=GREY, linestyle=":", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("RMSE (original response units; lower is better)")
        ax.set_xlim(0, max(table[key].max(), baselines[split]["rmse"]) * 1.27)
        ax.grid(axis="x")
    axes[0].set_yticks(positions, summary.label)
    axes[0].invert_yaxis()
    fig.suptitle("Ten model configurations on the same small dataset", fontsize=16, x=.03, ha="left")
    fig.text(.03, .02, "Validation-ordered rows. Circles/squares/triangles: seeds 42/43/44; orange: mean ± sample SD, not a confidence interval.\n"
             "Dotted line: training-mean surface baseline. 1,000 validation and 1,000 test matrices; 16 dose entries per matrix.", color=GREY, fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95))
    save_figure(fig, directory, "configuration_comparison")


def effects_plot(directory, contrasts):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, sharex=True)
    for ax, key, title in zip(axes, ("validation_rmse", "test_rmse"), ("Validation", "Test (exploratory)")):
        for y, row in enumerate(contrasts):
            value = row[key]
            ax.scatter(value["seed_deltas"], np.array([y-.1, y, y+.1]), color=BLUE, s=27, zorder=4)
            ax.errorbar(value["mean_delta"], y, xerr=value["sample_sd"], fmt="D", capsize=5,
                        color=ORANGE, markersize=6, zorder=5)
        ax.axvline(0, color=GREY, linewidth=1.2)
        ax.grid(axis="x")
        ax.set_title(title)
        ax.set_xlabel("RMSE change: with − without (negative = improvement)")
    axes[0].set_yticks(range(len(contrasts)), [row["comparison"] for row in contrasts])
    axes[0].invert_yaxis()
    fig.suptitle("Matched effects of the proposed changes", fontsize=16, x=.03, ha="left")
    fig.text(.03, .015, "First three: each seed's effect is averaged equally across four factorial backgrounds. Bias effects use the all-three background only.\n"
             "Dots: three seed-level effects. Orange intervals: mean ± sample SD, not confidence intervals or significance tests.", fontsize=9, color=GREY)
    fig.tight_layout(rect=(0, .12, 1, .94))
    save_figure(fig, directory, "matched_effects")


def diagnostic_plot(directory, summary, table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    for y, row in enumerate(summary.itertuples()):
        values = table[table.variant == row.variant].sort_values("seed").swap_rmse
        axes[0].scatter(values, [y-.12, y, y+.12], color=BLUE, s=28)
        axes[0].errorbar(row.swap_rmse_mean, y, xerr=row.swap_rmse_sd, fmt="D", capsize=4, color=ORANGE, markersize=5)
    axes[0].set_yticks(range(len(summary)), summary.label)
    axes[0].set_ylim(len(summary)-.5, -.5)
    axes[0].set_xlim(left=0)
    axes[0].grid(axis="x")
    axes[0].set_title("Drug-order inconsistency on validation")
    axes[0].set_xlabel("RMSE between original and swapped-back predictions\n(lower is more consistent; means ± seed SD)")
    values = summary[[name + "_near_zero_pct_mean" for name in ("h1", "h2", "alpha")]].to_numpy()
    im = axes[1].imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    axes[1].set_xticks(range(3), ["Hill slope h1", "Hill slope h2", "Interaction α"])
    axes[1].set_yticks([])
    axes[1].set_title("Near-zero parameter outputs (<0.01)")
    for y in range(len(summary)):
        for x in range(3):
            axes[1].text(x, y, f"{values[y, x]:.1f}%", ha="center", va="center",
                         color="white" if values[y, x] > 60 else "#20272C")
    fig.colorbar(im, ax=axes[1], orientation="horizontal",
                 label="Validation outputs below 0.01 (%)\nMean across three training seeds", shrink=.9)
    fig.suptitle("Diagnostics beyond prediction accuracy", fontsize=16)
    save_figure(fig, directory, "parameter_and_symmetry_diagnostics")


def curves_plot(directory, variants, histories):
    fig, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True, sharey=True)
    styles = [(42, BLUE, "-"), (43, ORANGE, "--"), (44, "#6B7338", ":")]
    for ax, variant in zip(axes.flat, variants):
        for seed, color, style in styles:
            history = histories[(variant["id"], seed)]
            ax.plot(history.epoch, history.val_rmse, color=color, linestyle=style, label=f"Seed {seed}")
            best = history.loc[history.val_rmse.idxmin()]
            ax.scatter(best.epoch, best.val_rmse, s=25, color=color, zorder=4)
        ax.set_title(variant["label"])
        ax.set_xlim(1, 100)
        ax.set_xticks([1, 25, 50, 75, 100])
        ax.grid(axis="y")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation RMSE\n(response units)")
    for ax in axes[-1]:
        ax.set_xlabel("Training epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .014))
    fig.suptitle("Validation learning curves for every configuration", fontsize=17, y=.995)
    fig.text(.06, .004, "Identical axes and 100-epoch cap; traces end at early stopping. Dots mark each selected checkpoint.", color=GREY, fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .98))
    save_figure(fig, directory, "validation_learning_curves")


def matrix_plot(directory, experiment, selection, metadata):
    with np.load(experiment / "reference/seed_44/val_predictions.npz", allow_pickle=False) as raw:
        reference = {key: raw[key] for key in raw.files}
    name, seed = selection["selected_variant"], selection["selected_seed_for_examples"]
    with np.load(experiment / name / f"seed_{seed}/val_predictions.npz", allow_pickle=False) as raw:
        chosen = {key: raw[key] for key in raw.files}
    error = np.sqrt(np.mean((reference["predicted"].astype(np.float64) - reference["observed"])**2, axis=(1, 2)))
    choices = choose_cases(error)
    indices = [index for _, index in choices]
    responses = np.concatenate([reference["observed"][indices].ravel(), reference["predicted"][indices].ravel(), chosen["predicted"][indices].ravel()])
    residuals = chosen["predicted"] - chosen["observed"]
    limit = float(np.max(np.abs(residuals[indices])))
    fig = plt.figure(figsize=(16, 17), layout="constrained")
    grid = fig.add_gridspec(8, 4, height_ratios=[.22, 1] * 4)
    response_axes, residual_axes, records = [], [], []
    for y, (quantile, index) in enumerate(choices):
        row = metadata.iloc[index]
        new_error = independent_metrics(chosen["observed"][index], chosen["predicted"][index])["rmse"]
        title_ax = fig.add_subplot(grid[y*2, :])
        title_ax.set_axis_off()
        title_ax.text(.01, .4, textwrap.fill(f"Reference error rank {quantile:.0%}: {row.drug1} + {row.drug2} · {row.cell} · matrix RMSE {error[index]:.2f} → {new_error:.2f}", 140), va="center", fontsize=11)
        for x, (title, values) in enumerate((("Measured", chosen["observed"][index]),
                                            ("Reference · seed 44", reference["predicted"][index]),
                                            (f"Selected variant · seed {seed}", chosen["predicted"][index]),
                                            ("Selected − measured", residuals[index]))):
            ax = fig.add_subplot(grid[y*2+1, x])
            signed = x == 3
            low, high = (-limit, limit) if signed else (float(responses.min()), float(responses.max()))
            im = ax.imshow(values, origin="lower", cmap=RESIDUAL_MAP if signed else "cividis", vmin=low, vmax=high)
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(4), [f"{value:.2g}" for value in chosen["dose2"][index]])
            ax.set_yticks(range(4), [f"{value:.2g}" for value in chosen["dose1"][index]])
            ax.set_xlabel("Drug 2 (µM)")
            ax.set_ylabel("Drug 1 (µM)")
            for a in range(4):
                for b in range(4):
                    fraction = (values[a, b] - low) / max(high - low, 1e-9)
                    white = (fraction < .2 or fraction > .8) if signed else fraction < .45
                    ax.text(b, a, f"{values[a,b]:.1f}", ha="center", va="center", fontsize=9, color="white" if white else "#20272C")
            if signed:
                residual_axes.append(ax)
                residual_im = im
            else:
                response_axes.append(ax)
                response_im = im
        records.append({"reference_error_rank": quantile, "source_row": int(row.source_row),
                        "drug1": row.drug1, "drug2": row.drug2, "cell": row.cell,
                        "reference_matrix_rmse": float(error[index]), "selected_matrix_rmse": new_error,
                        "observed": chosen["observed"][index].tolist(),
                        "reference_predicted": reference["predicted"][index].tolist(),
                        "selected_predicted": chosen["predicted"][index].tolist(),
                        "dose1_micromolar": chosen["dose1"][index].tolist(),
                        "dose2_micromolar": chosen["dose2"][index].tolist()})
    fig.colorbar(response_im, ax=response_axes, orientation="horizontal", shrink=.7, label="Response: shared scale, no clipping")
    fig.colorbar(residual_im, ax=residual_axes, orientation="horizontal", shrink=.9, label="Residual: symmetric scale")
    fig.suptitle(f"Fixed validation examples · selected variant: {name}\nExamples chosen from reference errors, not selected to favor the new model; cell spacing is not linear dose spacing", fontsize=13)
    save_figure(fig, directory, "validation_matrix_comparison")
    return records


def report_text(summary, table, contrasts, selection, completion, baselines):
    best = summary.iloc[0]
    reference = summary[summary.variant == "reference"].iloc[0]
    reduction = (1 - best.test_rmse_mean / reference.test_rmse_mean) * 100
    best_compact = summary[summary.parameters < 4_000_000].iloc[0]
    compact_reduction = (1 - best_compact.parameters / reference.parameters) * 100
    rows = ["# DeepSynBa small-data enhancement comparison", "",
            f"Ten configurations × three seeds = **30 completed runs** on the same 5,000-example dataset. "
            f"The configuration selected by mean validation RMSE is **{best.label}** (`{best.variant}`). "
            f"Its test RMSE is **{best.test_rmse_mean:.3f} ± {best.test_rmse_sd:.3f}**, versus "
            f"**{reference.test_rmse_mean:.3f} ± {reference.test_rmse_sd:.3f}** for the reference "
            f"({reduction:+.1f}% relative RMSE reduction). All ± values are sample SD across three training seeds, not confidence intervals.", "",
            "## Accuracy and model size", "",
            "Rows are ordered by validation performance, not by test results. Full metrics pool 16,000 entries from 1,000 matrices per split; combination-only metrics pool the 9,000 nonzero-dose interior entries.", "",
            "| Configuration | Parameters (M) | Validation RMSE | Test RMSE | Test Pearson | Test Spearman | Combination-only RMSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        rows.append(f"| {row.label} | {row.parameters/1e6:.2f} | {row.validation_rmse_mean:.3f} ± {row.validation_rmse_sd:.3f} | "
                    f"{row.test_rmse_mean:.3f} ± {row.test_rmse_sd:.3f} | {row.test_pearson_mean:.3f} ± {row.test_pearson_sd:.3f} | "
                    f"{row.test_spearman_mean:.3f} ± {row.test_spearman_sd:.3f} | {row.combination_rmse_mean:.3f} ± {row.combination_rmse_sd:.3f} |")
    rows += ["", f"The training-mean surface baseline has validation RMSE {baselines['val']['rmse']:.3f} and test RMSE {baselines['test']['rmse']:.3f}.", "",
             f"**Accuracy choice:** {best.label}, selected using mean validation error. "
             f"**Compact alternative:** {best_compact.label}, the lowest-validation-error configuration among models under 4 million parameters. "
             f"It uses {best_compact.parameters/1e6:.2f} million parameters ({compact_reduction:.1f}% fewer than the reference), "
             f"with validation RMSE {best_compact.validation_rmse_mean:.3f} and test RMSE {best_compact.test_rmse_mean:.3f} ± {best_compact.test_rmse_sd:.3f}. "
             "The compact alternative is a descriptive size/accuracy tradeoff, not a second confirmatory winner.", "",
             "![Configuration comparison](figures/configuration_comparison.png)", "",
             "## Which changes helped?", "",
             "These matched contrasts compare the same seed and other factor settings. The first three average each seed's effect over four matched factorial backgrounds. The bias comparisons are conditional on the all-three variant only. Negative RMSE changes indicate improvement. Three seeds are not enough to establish statistical significance.", "",
             "| Change | Validation RMSE change | Test RMSE change | Backgrounds |", "| --- | ---: | ---: | ---: |"]
    for row in contrasts:
        val, test = row["validation_rmse"], row["test_rmse"]
        rows.append(f"| {row['comparison']} | {val['mean_delta']:+.3f} ± {val['sample_sd']:.3f} | {test['mean_delta']:+.3f} ± {test['sample_sd']:.3f} | {row['matched_backgrounds']} |")
    rows += ["", "![Matched effects](figures/matched_effects.png)", "",
             "Main effects are averages, not promises for every combination. The individual matched differences are retained in `matched_differences.csv`; interactions and differing convergence can affect them.", "",
             "## Parameter and drug-order diagnostics", "",
             "The positive-output package is Softplus + 0.0001 for h1, h2 and alpha, together with near-one initialization (final weights scaled by 0.01). Activation and initialization were changed together; this experiment does not separate their effects. Exact positivity alone is not sufficient: the table also checks outputs below 0.01.", "",
             "| Configuration | Swap inconsistency RMSE | h1 <0.01 | h2 <0.01 | alpha <0.01 | Bias RMS |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        rows.append(f"| {row.label} | {row.swap_rmse_mean:.3f} ± {row.swap_rmse_sd:.3f} | {row.h1_near_zero_pct_mean:.1f}% | {row.h2_near_zero_pct_mean:.1f}% | {row.alpha_near_zero_pct_mean:.1f}% | {row.bias_rms_mean:.3f} |")
    rows += ["", "All diagnostics in this table use validation data. Swap inconsistency compares f(A,B) with the transposed f(B,A), with both dose axes swapped. Random augmentation encourages consistency but does not enforce exact symmetry; inference predictions are not averaged. Bias RMS measures the magnitude of the additive bias component in response units, not its causal contribution to accuracy.", "",
             "![Parameter and symmetry diagnostics](figures/parameter_and_symmetry_diagnostics.png)", "",
             "Training logs also retain the mean final-layer gradient norm and fraction of zero-gradient batches for each positive parameter at every epoch, before clipping. The unused sigma output was left unchanged and is excluded from these diagnostics.", "",
             "## Convergence and computation", "",
             "| Configuration | Epochs run (42 / 43 / 44) | Best epochs | Mean training seconds | Mean peak allocated GPU MB |", "| --- | --- | --- | ---: | ---: |"]
    for row in summary.itertuples():
        group = table[table.variant == row.variant].sort_values("seed")
        rows.append(f"| {row.label} | {' / '.join(str(x) for x in group.epochs_completed)} | {' / '.join(str(x) for x in group.best_epoch)} | {row.training_seconds_mean:.1f} | {row.peak_cuda_memory_mb_mean:.0f} |")
    rows += ["", f"Total measured training time: {completion['total_training_seconds']/60:.1f} minutes. Timings include per-epoch validation and checkpoint writing, but exclude final diagnostic/test evaluation and plotting. GPU allocation is PyTorch's peak allocation, not total device usage. Timing is descriptive on this laptop, not a controlled hardware benchmark.", "",
             "![All validation learning curves](figures/validation_learning_curves.png)", "",
             "## Same-example prediction comparison", "",
             f"Selected configuration: `{best.variant}`. Illustrative checkpoint seed {selection['selected_seed_for_examples']} is chosen by validation error within that configuration. The same four validation matrices used in the earlier reference diagnostic are retained: reference seed-44 error ranks 10%, 50%, 90%, and 99%. These are error-stratified diagnostics, not a random or representative sample. Inspect both improvements and failures.", "",
             "![Validation matrix comparison](figures/validation_matrix_comparison.png)", "",
             "## Protocol, limitations, and verification", "",
             "- The source model files and original results were preserved. All 30 models were trained anew. The three new reference histories and validation/test predictions match the earlier reference runs numerically.",
             "- Fixed drug-pair-disjoint split: 3,000 training / 1,000 validation / 1,000 test matrices, with the same source data and row identities. No added experiments or relabeled targets.",
             "- Same optimizer (AdamW, learning rate 0.0001, weight decay 0.00001), 128 batch size, gradient clipping 5, 100-epoch cap, patience 8. The original linear schedule reaches half rate at epoch 30 and remains there.",
             "- Smaller model: gene_channel 128, emb_size 512. Drug-order augmentation swaps each training example with probability 0.5, including drugs, dose axes, and target transposition; it does not double independent sample size.",
             "- Bias regularization uses a predeclared coefficient 0.1 on mean(bias²); the no-bias variant removes both bias networks. These two conditions were tested only on the all-three model. No regularization-coefficient search was performed.",
             "- All variants have the same epoch budget, not equal wall time. Early stopping can interact with architecture. Size changes also change the random-number allocation to weights, so matched seeds are not identical initial functions.",
             "- Configuration ranking uses mean validation RMSE across all three seeds, not the best individual seed or test score. All selections and checkpoint hashes were frozen before any new test evaluation. Every result is reported.",
             "- The old test results were already inspected; new test scores are exploratory comparisons on a reused test set. A fresh pair-disjoint holdout is needed before claiming confirmatory improvement. Selecting among ten configurations also risks validation overfitting.",
             "- Standard deviations quantify initialization/training randomness on this one small subset, not dataset-sampling uncertainty. Lower predictive error or nonzero heads does not validate the learned pharmacological parameters, synergy predictions, or clinical use.",
             "- Analysis independently recomputed RMSE, Pearson and Spearman correlations, combination-only error, symmetry error, bias magnitude, and near-zero fractions. It checked all row IDs, responses, dose arrays, frozen hashes, best epochs, and checkpoint reload equality.", "",
             "## Artifacts", "",
             "- [Per-seed results](per_seed_results.csv), [configuration summary](configuration_summary.csv), [machine-readable summary](summary.json).",
             "- [Matched differences](matched_differences.csv), [factorial contrasts](factorial_contrasts.json), [fixed prediction examples](validation_examples.json).",
             "- [Predeclared protocol](../protocol.json), [validation selections frozen before tests](../selection.json), [completion record](../completed.json).",
             f"- [Illustrative selected checkpoint](../{best.variant}/seed_{selection['selected_seed_for_examples']}/best.pt). Variant checkpoints must be loaded with `ablation_model.build_model`, not the original runner.",
             "- Each run contains history.csv, metrics.json, test_metrics.json, best.pt, and numeric validation/test predictions including swapped-back predictions and parameter outputs. Figures are available in PNG and SVG.", ""]
    return "\n".join(rows)


def analyze(experiment):
    experiment = Path(experiment).resolve()
    protocol, selection, completion = (read_json(experiment / name) for name in ("protocol.json", "selection.json", "completed.json"))
    if completion["status"] != "complete" or completion["runs"] != 30:
        raise AssertionError("Suite is incomplete")
    if sha256(ROOT / "data/small/dataset.npz") != protocol["dataset_sha256"]:
        raise AssertionError("Source dataset changed")
    for name, expected in protocol["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise AssertionError(f"Experiment source changed: {name}")
    with np.load(ROOT / "data/small/dataset.npz", allow_pickle=False) as raw:
        source = {key: raw[key] for key in raw.files}
    table, histories, hashes = load_verified_results(experiment, protocol, selection, source)
    summary = aggregate_results(table, protocol["variants"])
    if summary.variant.tolist() != [row["variant"] for row in selection["ranking"]]:
        raise AssertionError("Independent validation ranking disagrees with frozen selection")
    if summary.variant.iloc[0] != selection["selected_variant"]:
        raise AssertionError("Selected variant mismatch")
    mean_surface = source["response"][source["split"] == SPLITS["train"]].mean(axis=0)
    baselines = {split: independent_metrics(source["response"][source["split"] == SPLITS[split]],
                                           np.broadcast_to(mean_surface, (1000, 4, 4))) for split in ("val", "test")}
    contrasts, details = factorial_contrasts(table, protocol["variants"])
    output = experiment / "analysis"
    output.mkdir(exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    table.to_csv(output / "per_seed_results.csv", index=False)
    summary.to_csv(output / "configuration_summary.csv", index=False)
    pd.DataFrame(details).to_csv(output / "matched_differences.csv", index=False)
    write_json(output / "factorial_contrasts.json", contrasts)
    comparison_plot(figures, summary, table, baselines)
    effects_plot(figures, contrasts)
    diagnostic_plot(figures, summary, table)
    curves_plot(figures, protocol["variants"], histories)
    metadata = pd.read_csv(ROOT / "data/small/metadata.csv")
    metadata = metadata[metadata.split == SPLITS["val"]].reset_index(drop=True)
    np.testing.assert_array_equal(metadata.source_row, source["source_row"][source["split"] == SPLITS["val"]])
    examples = matrix_plot(figures, experiment, selection, metadata)
    write_json(output / "validation_examples.json", examples)
    write_json(output / "summary.json", {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "analysis_source_sha256": sha256(ROOT / "analyze_comparison.py"),
               "selected_variant": selection["selected_variant"], "configuration_summary": summary.to_dict("records"),
               "baselines": baselines, "contrasts": contrasts, "verified_artifact_sha256": hashes,
               "verification": {"all_30_runs_checked": True, "numeric_metrics_independently_recomputed": True,
                                "source_rows_and_doses_exact": True, "all_reference_predictions_exact": True,
                                "ranking_frozen_before_test": True, "all_checkpoint_hashes_unchanged": True},
               "uncertainty": "Sample SD across three seeds, not a confidence interval; exploratory reused test set"})
    report = report_text(summary, table, contrasts, selection, completion, baselines)
    comparisons = "; ".join(f"{row['reference_matrix_rmse']:.2f} → {row['selected_matrix_rmse']:.2f}" for row in examples)
    report = report.replace("![Validation matrix comparison](figures/validation_matrix_comparison.png)",
                            "![Validation matrix comparison](figures/validation_matrix_comparison.png)\n\n"
                            f"Reference → selected-model matrix RMSE for the four fixed examples: {comparisons}. "
                            "The aggregate gain does not eliminate difficult failures: the high-error and extreme-error examples "
                            "still have large residuals, and the initially low-error example becomes slightly worse. "
                            "These examples illustrate limitations rather than estimate their population frequency.")
    (output / "RESULTS.md").write_text(report, encoding="utf-8")
    print(summary[["label", "validation_rmse_mean", "test_rmse_mean", "test_rmse_sd", "parameters"]].to_string(index=False))
    print(f"REPORT {output / 'RESULTS.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/comparison_small")
    analyze(parser.parse_args().experiment)
