# Activation and initialization — completed 8 October 2026

This study separates the activation change from the output calibration bundled
in the earlier enhanced model. All **12 new fits** completed: four conditions
with seeds 42, 43 and 44, using the exact earlier 2,000 training matrices and
1,000 validation matrices. Training performed 21,984 optimizer updates and
27.9 minutes of summed timed training. All fits stopped before the 2,400 cap.

| Activation / initialization | Validation all-16 RMSE | Combination-9 RMSE |
|---|---:|---:|
| ReLU / original | 20.378 ± 2.132 | 22.499 ± 2.064 |
| Softplus / original | 14.480 ± 0.200 | 17.270 ± 0.237 |
| ReLU / near-one | 14.408 ± 0.091 | 17.189 ± 0.100 |
| Softplus / near-one | **14.343 ± 0.178** | **17.126 ± 0.185** |

All ± values are sample SD over three seeds, not confidence intervals. The
winner is selected by mean all-16 validation RMSE across every seed, rather
than selecting the best individual seed. Combination metrics use the same
checkpoint and exclude the zero-dose row and column.

The improvement cannot be attributed to Softplus alone. Its paired change is
−5.898 ± 2.123 RMSE with original initialization, compared with −0.065 ± 0.090
after near-one calibration. Calibration alone changes ReLU error by
−5.970 ± 2.042, compared with −6.036 when both changes are applied. The paired
interaction is +5.833 ± 2.209. Either intervention recovers most of the
observed reduction on this split; the smaller calibrated activation difference
does not establish equivalence or universal superiority.

At their selected checkpoints, all three original-ReLU seeds have at least
one head that is zero for every validation matrix. Calibrated ReLU avoids
entirely zero heads on these matrices but still has individual zero outputs;
for example, seed 43 has 14.1% zero alpha outputs. Softplus is positive by
construction. These observations support an optimization explanation without
establishing the complete training mechanism or biological parameter validity.

## Fixed protocol and evidence

- Full 40,944,401-parameter architecture; only the final activation and
  initialization of the h1, h2 and alpha heads change.
- Original initialization retains seeded Kaiming weights and zero biases.
  Near-one calibration multiplies these same final weights by 0.01 and uses
  bias 1 for ReLU or `log(expm1(1 - 1e-4))` for Softplus + 1e-4. Other seeded
  parameters and the random generator state remain identical.
- Calibration is a package of weight scaling and activation-specific bias;
  this experiment does not isolate scale from bias.
- Additive bias, 50% drug/dose swapping with target transposition, AdamW,
  batch size 128, float32, norm clipping at 5 and the preceding update-based
  learning-rate schedule stay fixed.
- Validation checks every 24 updates, an eight-check / 192-update patience,
  and a 2,400-update maximum. Early stopping changes realized exposure, so
  this is a comparison under a common training rule.
- The dedicated input archive contains only 2,000 training and 1,000
  validation matrices. Preparation verified these against the selected rows
  in the old shared archive; subsequent fitting/evaluation uses this dedicated
  archive and performs no holdout inference.
- Protocol, data, source identities and training selections froze before
  fitting. Softplus / near-one repeats the earlier enhanced 2,000-matrix cell.

All **44 integrated preflight tests** passed. All 12 initial models and all
12 selected checkpoints reproduce their stored outputs exactly on CUDA.
The three Softplus / near-one controls also reproduce the previous validation
predictions and training, validation, learning-rate and stopping trajectories.
Independent calculations check metrics, head summaries, histories, paired
contrasts and the configuration ranking; a coefficient-matrix calculation
separately checks the factorial arithmetic. Both scientific figures were
visually inspected after checking their plotted data.

Full results, all seven paired contrasts for both RMSE measures, per-seed
diagnostics and PNG/SVG figures are in
[the numeric analysis](results/activation_initialization_20261008/RESULTS.md).
The frozen protocol, source/data hashes, checkpoints, outputs, histories,
selection, completion and verification records are retained in
`runs/activation_initialization_20261008/`.

## Repeat in a fresh directory

Run from this repository using the documented CUDA environment and the
earlier run archive required by this study:

```powershell
$pythonExecutable = '.\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $pythonExecutable run_activation_initialization.py --stage prepare --output runs/factorial_repeat
& $pythonExecutable run_activation_initialization.py --stage train --output runs/factorial_repeat
& $pythonExecutable analyze_activation_initialization.py --experiment runs/factorial_repeat --verify-checkpoints
& $pythonExecutable -m unittest discover -s tests -v
```

Use a fresh output directory. A complete fit can be reused after its frozen
file/protocol checks; an interrupted fit cannot resume partial optimizer state.

These are validation results from one reused split. The two previously examined
holdouts remain development evidence. Independently tuned competing methods,
multiple splits, a reserved final evaluation, another experimental source and
unseen-drug/cell evaluation remain necessary for broader research claims.
Runtime includes validation and checkpoint I/O and is descriptive.
