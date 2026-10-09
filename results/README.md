# DeepSynBa experiment records

These records document the local reproduction and follow-up experiments completed
between 29 September and 8 October 2026. They contain reported numeric results,
per-seed CSVs, analysis summaries, protocols and figures. Model checkpoints,
full prediction arrays, generated datasets and the Python environment remain local
and are excluded from Git. The workflow scripts and guides describe how to produce
those artifacts.

| Study | Evidence | Main finding |
| --- | --- | --- |
| Three-seed reference | [Results](extended_100/RESULTS.md) | Test RMSE 21.327 ± 2.095; zero-valued parameter heads were observed. |
| Ten configurations and three seeds | [Results](comparison_small/RESULTS.md), [per-seed CSV](comparison_small/per_seed_results.csv) | Validation selected positive outputs plus drug-order augmentation. |
| Fresh pair confirmation and nested training sizes | [Results](next_steps_20261001/RESULTS.md), [confirmation CSV](next_steps_20261001/confirmation_summary.csv) | Fresh holdout RMSE 14.443 ± 0.367 versus 21.357 ± 2.247 for the local reference. |
| Equal optimizer-update budget | [Results](step_budget_20261001/RESULTS.md), [validation CSV](step_budget_20261001/validation_summary.csv) | At 2,000 training matrices, enhanced validation RMSE was 14.343 ± 0.178. |
| Activation and initialization factorial | [Results](activation_initialization_20261008/RESULTS.md), [validation CSV](activation_initialization_20261008/validation_summary.csv) | Near-one calibration recovered most of the gain; activation and initialization were studied separately. |

All ± values above are sample standard deviations over seeds 42, 43 and 44.
The fresh holdout contained 1,000 matrices from 323 previously unused unordered
drug pairs. Its paired cluster bootstrap interval conditions on the frozen fitted
models. Later optimization studies evaluated validation data only. These results
do not establish performance on a new experimental source, unseen drugs/cell
lines, or the original paper's full-data benchmark.

The original model and processed data are credited to
[the DeepSynBa authors](https://github.com/hikuru/DeepSynBa), whose repository states
CC BY-NC-SA 4.0 terms.
