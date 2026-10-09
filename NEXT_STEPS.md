# Fresh confirmation and data efficiency

The subsequent equal-update experiment is also complete; see
[STEP_BUDGET.md](STEP_BUDGET.md) for its 18 new fits and validation-only results.
The holdout results below belong to the preceding frozen experiment.

## Completed results — 1 October 2026

All nine frozen-model confirmation jobs and all 27 new learning-curve training
jobs are complete. The curve includes 36 conditions after reusing nine previous
3,000-matrix fits. All 30 correctness tests passed; numeric results were
independently recomputed and all fresh inputs checked against the source.

On the fresh holdout, positive outputs plus augmentation achieved RMSE
**14.443 ± 0.367**, against reference **21.357 ± 2.247** and compact plus
augmentation **15.019 ± 0.011**. These ± values are sample SD across three seeds.
The primary relative RMSE reduction is **32.4%**, with a marginal paired
drug-pair bootstrap 95% interval of **29.0–35.8%**, conditional on the fixed fits.

| Training matrices | Reference RMSE | Positive + augmentation RMSE | Compact + augmentation RMSE |
|---:|---:|---:|---:|
| 500 | 25.003 ± 0.133 | 23.872 ± 0.119 | 25.401 ± 0.471 |
| 1,000 | 24.529 ± 0.755 | 18.253 ± 0.219 | 21.816 ± 0.341 |
| 2,000 | 20.374 ± 4.166 | 16.100 ± 0.794 | 16.831 ± 0.388 |
| 3,000 | 21.357 ± 2.247 | 14.443 ± 0.367 | 15.019 ± 0.011 |

For the frozen enhanced model, using 2,000 instead of 3,000 training matrices
saves one third of the training data with higher observed error. The
3,000-matrix configuration retains the best mean validation performance.
Seven new fits reached the 100-epoch cap, so these curves do not establish a
minimum sufficient dataset size. Further tuning needs validation and another
reserved evaluation set; this holdout has now been examined.

## Protocol and artifacts

This study continues the completed small-data reproduction and ten-variant
comparison. It reuses their frozen checkpoints, model shape fixes, and
5,000-matrix subset with matching source fingerprints. The full checkpoint
archive remains local; numeric summaries and protocols are published in `results/`.

The new experiment is `runs/next_steps_20261001`. Its immutable `protocol.json`
freezes the three previously selected configurations, seeds 42/43/44, all nine
existing checkpoint hashes, fresh holdout row IDs, nested training subsets,
endpoints, bootstrap method, and learning-curve hyperparameters.

The fresh holdout has 1,000 complete matrices from 323 drug pairs. It is sampled
from the original test partition after excluding every unordered pair present
in any earlier train, validation, or test subset. The candidate pool is 18,822
matrices / 343 pairs. There is zero previous-pair or previous-row overlap.

The data-efficiency study uses one fixed nested ordering of the original
training subset: 500, 1,000, 2,000 and 3,000 complete matrices. It trains 27 new
models and reuses the nine existing 3,000-matrix fits. All new checkpoints are
selected on the same validation set and frozen before holdout inference. The
500-matrix training subset omits Pemetrexed Disodium; the other sizes cover all
101 drugs, and all sizes cover all 60 cells.

Results, numeric predictions, paired intervals, per-seed CSVs and PNG/SVG figures
are saved in [the experiment analysis](results/next_steps_20261001/RESULTS.md).
Intervals resample complete drug-pair clusters, preserving all their selected
cell matrices and doses. They condition on fixed models and do not represent
all dependence across pairs sharing drugs or cells. Seed SD and bootstrap
intervals describe different sources of variability.

## Run a new experiment

Use the CUDA Python environment described in `REPRODUCE_SMALL.md`. Run from
this repository with the required earlier run archive available:

```powershell
$pythonExecutable = '.\.venv\Scripts\python.exe'
& $pythonExecutable run_next_steps.py --stage prepare --previous-root . --output runs/next_repeat
& $pythonExecutable run_next_steps.py --stage confirm --output runs/next_repeat
& $pythonExecutable run_next_steps.py --stage learning --output runs/next_repeat
& $pythonExecutable analyze_next_steps.py --experiment runs/next_repeat
& $pythonExecutable verify_next_steps.py --experiment runs/next_repeat
& $pythonExecutable -m unittest discover -s tests -v
```

`--previous-root` identifies the prior experiment files, not a destination for
new outputs. The example expects the prior archive beneath this repository.
For another machine, copy the prior workflow and
`runs/comparison_small` with its protocols, histories, validation predictions,
metrics and nine finalist checkpoints. Install the documented Python/CUDA
requirements and preserve their source hashes. Use `setup_reference.py` to fetch
the checksum-verified external `data.zip`, and prepare `data/small` with
`prepare_small_data.py`. The compact published CSVs do not replace the full
checkpoint and prediction archive needed to rerun these follow-up stages.

Preparation refuses an existing output directory. Learning can skip fully
completed jobs after verifying their hashes; partial optimizer state is not
resumed. To repeat from scratch, use a new directory. The runner enforces the
frozen code/data identities. The analyzer can be rerun during training to show
confirmation results and again after completion to verify the full curves.

This is same-source unseen-pair confirmation and a fixed-protocol learning
curve. Further model tuning should use validation and reserve another final
holdout or independent source. Latent parameter validity and external-data
generalization remain separate experiments.
