# Reproduce DeepSynBa with 5,000 examples

This is my small-data research workflow using the external
[official DeepSynBa implementation](https://github.com/hikuru/DeepSynBa)
at commit `43c16961da835c8545842d564a8efdfec58bdd2f`. Run `setup_reference.py`
once to fetch and verify the required model, configuration, and data files.
The workflow runs the released model on a reproducible NCI-ALMANAC subset.
It is a scaled-down reproduction of the implementation, not a replication of
the paper's full-data results or all benchmark comparisons.

The recorded experiments used an NVIDIA RTX 3070 Laptop GPU (8 GB), an Intel
i7-12650H, approximately 32 GB RAM, and Python 3.11. The setup below creates
the project environment in `.venv`.

## Enhancement comparison

The separate enhancement workflow compares ten configurations on the same subset:
the full 2×2×2 combinations of positive-output repair, smaller widths, and drug-order
augmentation, plus bias-penalized and no-bias versions of the combined model.
Each configuration uses seeds 42, 43, and 44, a 100-epoch cap, and patience eight.
The reference model is prepared locally; earlier result records are preserved.

Run the complete comparison in a **new** output directory:

```powershell
.\.venv\Scripts\python.exe compare_variants.py --output runs/comparison_repeat
.\.venv\Scripts\python.exe analyze_comparison.py --experiment runs/comparison_repeat
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This fixed protocol requires the CUDA environment above and `requirements-analysis.txt`.
It freezes the configuration ranking by mean validation RMSE and all checkpoint hashes
before evaluating any test results. Test comparisons are exploratory because the same
test subset was inspected in earlier experiments. It refuses to overwrite an existing
experiment directory and does not resume interrupted training from optimizer state.

Results are saved in each experiment's `analysis/RESULTS.md`, with per-seed CSVs,
configuration summaries, matched factorial contrasts, parameter/gradient diagnostics,
and PNG/SVG figures. The local comparison output is `runs/comparison_small`.

All 30 local runs are complete: [full comparison and plots](results/comparison_small/RESULTS.md).
The validation-selected configuration is **positive outputs + drug-order augmentation**
at the original widths, with test RMSE **14.605 ± 0.390** versus **21.327 ± 2.095**
for the reference (31.5% reduction). The best compact configuration by validation is
**smaller + augmentation**, with 3.50 million parameters and test RMSE **15.161 ± 0.082**.
These are sample SDs across seeds, not confidence intervals; the existing test set was
reused, so the comparison is exploratory. The selected illustrative checkpoint is
`runs/comparison_small/positive_swap/seed_42/best.pt`.

**Load variant checkpoints with their variant-aware constructor, not `train_small.py`.**
The original runner does not reconstruct the changed output activations. For example:

```python
import torch
from ablation_model import build_model

checkpoint = torch.load("runs/comparison_repeat/positive/seed_42/best.pt",
                        map_location="cpu", weights_only=True)
model, _ = build_model(checkpoint["model_config"], checkpoint["variant"])
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
# Supply the same five input tensors used by SmallDataset.
```

The positive-output treatment changes both activation and initialization; it is not
an activation-only comparison. Bias ablations are conditional on the all-three model,
not a complete four-factor experiment. Treat these models as DeepSynBa variants,
not an exact reproduction of the released architecture.

## Completed local runs

The subsequent three-seed experiment with a 100-epoch limit is complete:
[extended results and plots](results/extended_100/RESULTS.md).
Mean test RMSE was **21.327 ± 2.095** (sample SD, seeds 42/43/44).
The validation-selected model is seed 44, epoch 50, with test RMSE **20.022**;
its checkpoint is `runs/extended_100/seed_44/best.pt`.
All runs stopped early. Parameter-head diagnostics reveal zero Hill slopes,
so inspect the limitations in the extended results before interpreting the model.

The table below preserves the original 30-epoch setup runs for comparison.

The default 5,000-example experiment has already completed on both CPU and GPU.
Both ran for 30 epochs and selected epoch 28 by validation MSE.

| Run | Training time | Test RMSE | Test Pearson | Test Spearman |
| --- | ---: | ---: | ---: | ---: |
| RTX 3070, `runs/small_gpu` | 35.7 seconds | 20.536 | 0.626 | 0.463 |
| CPU, `runs/small_cpu` | 190.8 seconds | 20.713 | 0.625 | 0.464 |
| Training-set mean-surface baseline | — | 22.885 | 0.477 | 0.439 |

These metrics pool the 16,000 dose-response entries in the 1,000-example test
split. The GPU model's RMSE on the nine nonzero-dose combinations is 23.008,
versus 26.377 for the baseline. Peak CUDA tensor allocation was 791.5 MiB;
this does not include all driver/context memory. Timings exclude installation
and data preparation. The CPU and GPU runs used different package environments,
both recorded alongside their outputs, so they are not a controlled speed benchmark.

The trained GPU model is `runs/small_gpu/best.pt`; detailed metrics are in
`runs/small_gpu/metrics.json`. To evaluate this existing model:

```powershell
.\.venv\Scripts\python.exe train_small.py --device cuda --checkpoint runs/small_gpu/best.pt
```

The project environment passed its dependency check and all eight correctness
tests before the GPU training run. Reloading the GPU checkpoint reproduced
all 1,000 saved test predictions exactly (`runs/verify_gpu_checkpoint`).
This confirms a runnable small-data workflow;
it does not establish the paper's reported full-data accuracy.

## Run in PowerShell

From this repository directory, after environment and subset preparation:

```powershell
.\.venv\Scripts\python.exe train_small.py
```

The default trains for up to 30 epochs, stopping after eight epochs without
validation improvement. It selects the best checkpoint using validation MSE,
then evaluates the test split. Every invocation creates a new `runs` folder.

For a one-epoch check or a longer run:

```powershell
.\.venv\Scripts\python.exe train_small.py --epochs 1
.\.venv\Scripts\python.exe train_small.py --epochs 100
```

Use `--device cuda` to require GPU execution or `--device cpu` to use the CPU.
The default automatically selects an available GPU. `config.small.yaml`
controls batch size, epoch limit, early stopping, and seed. Relative paths
in that file resolve from its directory.

## Initial setup on another machine

Use Python 3.11. These commands leave global packages unchanged.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe setup_reference.py
.\.venv\Scripts\python.exe prepare_small_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe train_small.py
```

The CUDA PyTorch wheel is about 2.45 GB. The NVIDIA driver must support its
CUDA runtime; a separate CUDA toolkit and MolFormer download are not needed.
For CPU only, use `https://download.pytorch.org/whl/cpu` in the PyTorch install
command. See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/).

## Repeat the three-seed experiment

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe run_extended.py --output runs/extended_repeat
.\.venv\Scripts\python.exe analyze_extended.py --experiment runs/extended_repeat
```

The suite fixes training seeds 42/43/44, keeps the existing prepared subset,
uses a 100-epoch limit with patience eight, and records the protocol before
training. It defers test evaluation until every validation-selected checkpoint
is frozen. The analyzer compares against the original 30-epoch GPU run in
`runs/small_gpu` and saves an experiment summary plus PNG/SVG research figures.
Use a fresh output directory for another run; existing experiments are preserved.

For individual runs, `train_small.py --seed 43 --epochs 100 --defer-test`
sets the training seed without changing the subset and saves training/validation
results only. Omitting `--defer-test` also evaluates the test set. Saved checkpoints
can be evaluated later with `--checkpoint`. Validation predictions and metadata
are now saved alongside the existing artifacts.

## Dataset size and split

The released metadata contains 274,516 complete drug-pair/cell-line matrices.
The default subset has 3,000 training, 1,000 validation, and 1,000 test matrices:
1.82% of the released examples. Each example retains its complete 4 x 4 matrix,
including zero-dose and monotherapy entries. Sampling is uniform without
replacement within each original split, with fixed seeds. Individual dose
points are never split between partitions.

The original split assigns each unordered drug pair to one partition across
all cell lines. The subset preserves those assignments; preparation checks
both original and selected data for pair overlap and duplicate experiments.
The default subset covers all 101 drugs and all 60 cell lines in each split.
This run evaluates new drug combinations, not new drugs or cell lines.

The 19 MB source archive is read once during preparation. Only selected matrices,
concentrations, and the required precomputed embeddings/expression vectors are
saved in `data/small`. Training loads only those reduced arrays. Concentrations
are converted from molar to micromolar, matching the original loader. Responses
retain their original scale (untreated reference 100) and are not clipped.
The original model fills missing gene expression with zero; the subset applies
the same treatment to 82 missing values for SF-539 and records this in its manifest.

For an even smaller dataset, use a new directory:

```powershell
.\.venv\Scripts\python.exe prepare_small_data.py --train 600 --val 200 --test 200 --output data/tiny
.\.venv\Scripts\python.exe train_small.py --data-dir data/tiny
```

To increase the default later:

```powershell
.\.venv\Scripts\python.exe prepare_small_data.py --train 12000 --val 4000 --test 4000 --output data/20k
.\.venv\Scripts\python.exe train_small.py --data-dir data/20k --epochs 100
```

Preparation refuses to overwrite a nonempty subset directory. The source
archive checksum is pinned before its pickle is deserialized. The generated
subset uses numeric arrays without pickle objects. Its manifest records any
drugs/cells absent from training; extremely small subsets can lose coverage.

## Fidelity to the released code

- The released `DeepSynBa` class, encoder/head widths (256/2048), fixed MolFormer
  features, output heads, and response equation are retained: 40,944,401 parameters.
- MSE, AdamW, learning rate `1e-4`, weight decay `1e-5`, the released linear
  learning-rate schedule, batch size 128, and float32 arithmetic are retained.
- A plain PyTorch runner replaces the machine-specific Lightning/W&B launch and
  evaluation scripts. Logs/checkpoints stay local; no W&B account is needed.
- Dataset size and epoch budget are reduced. Early stopping and deterministic
  seeds are added. The final partial training batch is used unless it has one
  example (BatchNorm requires two); the original loader dropped all partial batches.
- The upstream unrestricted `squeeze()` calls are fixed to preserve the batch
  dimension in single-example evaluation, without changing model parameters.
- PyTorch 2.5.1 and Python 3.11-compatible packages replace the upstream pins.
  Each run records all installed versions.

The implementation includes a `sigma_mean` head unused in deterministic
response computation and two dose-dependent bias heads. These are retained
for fidelity to the released code. This is therefore not an independent exact
reimplementation of the journal paper's eight-parameter architecture.
O'Neil evaluation, unseen-drug/cell experiments, synergy-score calculations,
and baseline model retraining are not included in this small run.

## Saved results

Each run produces:

- `best.pt`: validation-selected weights, model config, seed, and subset hash.
  This supports inference; optimizer state for resuming training is not saved.
- `history.csv`: training/validation RMSE and timing by epoch.
- `metrics.json`: test RMSE, Pearson, and Spearman pooled over all 16 entries;
  separate metrics on the nine nonzero-dose combinations; and a simple mean
  response surface baseline fitted only on the training set.
- `test_predictions.npz` and `test_metadata.csv`: measurements, predictions,
  dose grids in micromolar, and matching example identities.
- `environment.json`, `requirements-lock.txt`, `resolved_config.yaml`, and
  `data_manifest.json`: software/hardware, code hashes, config, and provenance.
  `data/small/metadata.csv` records original row IDs for every selected example.

The all-entry metrics follow the pooling convention in the released evaluator.
The nine-entry metrics separately measure the combination interior. A small
subset, one seed, and fewer epochs produce different results from the paper.
Use validation for tuning and reserve test results for the final comparison.
Deterministic settings target repeatability on the same hardware/software;
exact equality across platforms or package versions is not guaranteed.

To evaluate saved weights without retraining, replace the example path:

```powershell
.\.venv\Scripts\python.exe train_small.py --checkpoint runs/small_YYYYMMDD_HHMMSS/best.pt
```

The subset checksum must match the checkpoint. Supply `--data-dir` if the
checkpoint was trained on a different subset directory.

Paper: [DeepSynBa: actionable drug combination prediction with complete dose-response profiles](https://academic.oup.com/bioinformatics/article/42/7/btag416/8712837).
Retain upstream attribution and the repository's stated CC BY-NC-SA 4.0 terms.
