# DeepSynBa research on small data and optimization

Research and experiments by **Kami80** on reproducing DeepSynBa with limited
training data, diagnosing parameter-head failures, and evaluating improvements.
This repository contains my experimental code, protocols, result tables, figures,
and report. The original DeepSynBa model is an external reference dependency from
[the authors' repository](https://github.com/hikuru/DeepSynBa).

[Two-page research report](reports/DeepSynBa_Two_Page_Report.pdf) ·
[Results and evidence](results/README.md) ·
[Completed research stages](PROGRESS.md)

## My contributions

- Built a deterministic 5,000-matrix NCI-ALMANAC subset with pair-disjoint
  training, validation, and test partitions and source-data checks.
- Added a local CPU/GPU training workflow, validation-based checkpoint selection,
  three-seed analysis, and diagnostics for zero Hill slopes and drug-order effects.
- Compared ten configurations across three seeds, including positive parameter
  outputs, smaller networks, drug-order augmentation, and bias controls.
- Evaluated frozen models on fresh drug pairs and studied nested training subsets
  of 500, 1,000, 2,000, and 3,000 matrices.
- Tested equal optimizer-update budgets and separated output activation from
  initialization in a controlled factorial experiment.

## Selected results

The following models used 3,000 training matrices and the same fresh holdout of
1,000 matrices from 323 previously unused unordered drug pairs. Lower RMSE is
better. Values are means ± sample SD across seeds 42, 43, and 44.

| Local model | Parameters | Fresh holdout RMSE |
| --- | ---: | ---: |
| Reference implementation | 40.94 million | 21.357 ± 2.247 |
| Positive outputs + drug-order augmentation | 40.94 million | **14.443 ± 0.367** |
| Compact model + drug-order augmentation | 3.50 million | 15.019 ± 0.011 |

The enhanced model reduced mean RMSE by **32.4%** relative to the local reference.
The paired drug-pair bootstrap 95% interval was **29.0–35.8%**, conditional on the
frozen fitted models. The compact model used **91.5% fewer parameters**.
[Full confirmation results](results/next_steps_20261001/RESULTS.md).

With an equal update budget and 2,000 training matrices, enhanced validation RMSE
was **14.343 ± 0.178**. The activation/initialization study found that calibrated
ReLU also reached **14.408 ± 0.091**: most of the original ReLU gap was recovered
by initialization calibration. This does not establish equivalence between the
activations. [Budget results](results/step_budget_20261001/RESULTS.md) ·
[Factorial results](results/activation_initialization_20261008/RESULTS.md).

These are local small-data results, not a reproduction of the paper's full-data
benchmark scores. The initial test set was reused during exploratory development;
the fresh confirmation uses previously unexamined pairs from the same source.
Later optimization studies report validation results only. Generalization to a
new experimental source, unseen drugs/cell lines, and biological parameter
validity remains to be established.

## Research records

| Stage | Experiment guide | Published evidence |
| --- | --- | --- |
| Small-data reproduction and diagnostics | [REPRODUCE_SMALL.md](REPRODUCE_SMALL.md) | [Three-seed reference](results/extended_100/RESULTS.md) |
| Ten-configuration comparison | [REPRODUCE_SMALL.md](REPRODUCE_SMALL.md#enhancement-comparison) | [Comparison results](results/comparison_small/RESULTS.md) |
| Fresh confirmation and learning curves | [NEXT_STEPS.md](NEXT_STEPS.md) | [Confirmation results](results/next_steps_20261001/RESULTS.md) |
| Equal optimizer-update budget | [STEP_BUDGET.md](STEP_BUDGET.md) | [Budget results](results/step_budget_20261001/RESULTS.md) |
| Activation and initialization | [ACTIVATION_INITIALIZATION.md](ACTIVATION_INITIALIZATION.md) | [Factorial results](results/activation_initialization_20261008/RESULTS.md) |

The `results/` directory contains numeric summaries, per-seed CSVs, frozen
protocols, analysis figures, and verification records. Large model checkpoints,
full prediction arrays, prepared datasets, and the Python environment remain in
the local run archive. Some recorded paths and fingerprints identify that
historical archive; published records are retained without changing their values.

## Run my experiments

Use Python 3.11. The example installs CPU PyTorch; the reproduction guide also
contains the CUDA setup used for the recorded experiments.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe setup_reference.py
.\.venv\Scripts\python.exe prepare_small_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe train_small.py --device cpu --output runs/reproduction_repeat
```

`setup_reference.py` fetches the pinned original model/configuration and processed
data, verifies every required file's SHA-256, and applies the batch-dimension
correction needed for single-example evaluation. These reference files are local
ignored dependencies, separate from the research files published here. Matching
existing files are preserved; differing files are rejected to protect local work.
The script also supports `--source-zip` for an offline copy of the pinned source.

To repeat the ten-configuration experiment in a new output directory:

```powershell
.\.venv\Scripts\python.exe compare_variants.py --output runs/comparison_repeat
.\.venv\Scripts\python.exe analyze_comparison.py --experiment runs/comparison_repeat
```

Follow-up studies use earlier run artifacts as described in their guides.
Recorded source hashes may differ after documentation and repository updates;
a new experiment should freeze its own code, data, protocol, and checkpoints.

## Original work and attribution

DeepSynBa was introduced by Arslan, Aktukmak, Ferreira, Liò, Gao, and Taştan in
*Bioinformatics* (2026), [doi:10.1093/bioinformatics/btag416](https://doi.org/10.1093/bioinformatics/btag416).
Credit for the original model and processed dataset belongs to those authors.
See [ATTRIBUTION.md](ATTRIBUTION.md) for the pinned source, licensing reference,
and the boundary between the reference implementation and my research additions.
