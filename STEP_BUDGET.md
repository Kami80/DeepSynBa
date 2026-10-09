# Equal optimizer-update budget

This experiment follows the completed confirmation and learning curves in
[NEXT_STEPS.md](NEXT_STEPS.md). It asks how the reduced-data models change when
their maximum optimizer updates, learning-rate timing, validation cadence and
stopping patience use the same units.

## Completed results — 1 October 2026

All 18 new fits are complete. They used **37,464 actual updates** and **24.0
minutes** of summed training time. All 37 tests passed; all 18 new checkpoints
reproduced their saved validation predictions exactly on CUDA. All six
3,000-matrix controls matched their earlier predictions and training trajectories.

| Training matrices | Earlier enhanced RMSE | New enhanced RMSE | Earlier compact RMSE | New compact RMSE |
|---:|---:|---:|---:|---:|
| 1,000 | 17.831 ± 0.241 | **16.095 ± 0.105** | 21.152 ± 0.366 | **16.396 ± 0.247** |
| 2,000 | 15.402 ± 0.660 | **14.343 ± 0.178** | 16.252 ± 0.312 | **14.892 ± 0.101** |
| 3,000 | 14.048 ± 0.329 | **14.048 ± 0.329** | 14.454 ± 0.064 | **14.454 ± 0.064** |

These are validation all-16-entry RMSEs, with sample SD across three seeds.
At 1,000 matrices, mean error decreased by 9.7% for the enhanced model and 22.5%
for the compact model. The enhanced 2,000-matrix alternative uses one third less
training data and has 0.294 higher mean validation RMSE than its 3,000-matrix
setting. The accuracy winner remains enhanced / 3,000 matrices; the compact
3,000-matrix configuration remains the best mean result within the compact model.

Combination-only error also improved: at 1,000 matrices, enhanced changed from
20.923 to 19.046, and compact from 24.468 to 19.424. At 2,000 matrices, the
corresponding changes were 18.226 to 17.126 and 19.196 to 17.735. Seven compact
fits reached the update limit; all nine enhanced fits and two compact fits
stopped earlier. These results do not establish that the smaller datasets have
equivalent generalization accuracy.

The experiment directory is `runs/step_budget_20261001`. The
[full numeric report](results/step_budget_20261001/RESULTS.md) includes
paired changes, compute accounting, independent audit results, limitations and
three PNG/SVG figures. No new holdout inference was performed.

## Fixed design

- Models: `positive_swap` (enhanced, 40,944,401 parameters) and `small_swap`
  (compact, 3,499,409 parameters), with the earlier architectures and augmentation.
- Training data: the exact earlier nested 1,000/2,000/3,000-matrix subsets.
- Seeds: 42, 43, 44 for every model and size; 18 new fits from scratch.
- Maximum budget: 2,400 updates, equivalent to at most 300/150/100 epochs.
- Validation: the same 1,000 matrices, evaluated every 24 updates. Checkpoints
  minimize all-16-entry validation MSE. Combination-only results use that same
  checkpoint and exclude the zero-dose row and column.
- Early stopping: eight checks without strict improvement, or 192 updates.
- AdamW: learning rate 1e-4, weight decay 1e-5, float32, batch size 128,
  gradient clipping at norm 5. LinearLR decreases to half the starting rate by
  update 720 and stays there. It steps after each validation check.
- No holdout inference or selection. Report all seeds and conditions; rank
  configurations by mean validation RMSE. Seed SD is not a confidence interval.

All settings, training-row identities, earlier artifact hashes and source
fingerprints were frozen before training. The 3,000-matrix rule is identical to
the earlier experiment: all six controls must reproduce their earlier histories
and predictions before the lower-data comparisons can be accepted.

At smaller sizes, this changes the maximum budget together with scheduler,
evaluation and stopping timing. It estimates the combined training-rule effect,
not an isolated duration effect. Realized updates can differ through early
stopping. Wall times include validation and saving and are descriptive laptop
measurements; earlier/new sessions are not a controlled timing benchmark.

## Reproduce in a new directory

Use the existing environment; no dependency installation or data download is
needed on this machine. Run from the worktree:

```powershell
$pythonExecutable = 'E:\PROJECTS\UNI\DeepSynBa\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path (Get-Location).Path 'runs/step_budget_repeat/matplotlib'
& $pythonExecutable run_step_budget.py --stage prepare --baseline runs/next_steps_20261001 --output runs/step_budget_repeat
& $pythonExecutable run_step_budget.py --stage train --output runs/step_budget_repeat
& $pythonExecutable analyze_step_budget.py --experiment runs/step_budget_repeat --verify-checkpoints
& $pythonExecutable -m unittest discover -s tests -v
```

Preparation refuses an existing directory. Training can reuse a fully completed
job after verifying its frozen identities; partial optimizer state is not
resumed. Use a new directory to repeat from scratch. The baseline experiment's
selection and all 18 referenced histories, checkpoints and validation prediction
files must remain available, including its original 3,000-matrix files under
`E:\PROJECTS\UNI\DeepSynBa\runs\comparison_small`.

Inspect `analysis/validation_per_seed.csv`, `validation_summary.csv`,
`paired_changes_per_seed.csv` and `paired_changes_summary.csv` for numeric
comparisons. `analysis/verification.json` records independent arithmetic,
source/batch/stopping audits and CUDA checkpoint reload checks. PNG/SVG figures
are under `analysis/figures/`; histories and checkpoints are under `training/`.

The shared prepared container includes unused original test rows; only its
training and validation partitions enter the fits or selection. The separate
fresh holdout from the preceding experiment is not evaluated here. Another
reserved evaluation set or independent-source data is needed for final
confirmation after adaptive development.
