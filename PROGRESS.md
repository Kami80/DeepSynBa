# DeepSynBa continuation — updated 8 October 2026

The project originally requested a small-data local reproduction, followed by
enhancement comparisons and a LaTeX/PDF history report. Those earlier artifacts
are in `E:\PROJECTS\UNI\DeepSynBa`; this worktree started from clean upstream
commit `43c1696`, so their code and reduced dataset were restored here before
continuing. The model experiments left the previous project unchanged. The
subsequent report request updated its LaTeX source and main PDF, as recorded below.

The report's first two proposed follow-ups are now complete:

1. Fresh same-source confirmation: three frozen configurations × seeds 42/43/44,
   on 1,000 previously unexamined matrices from 323 pairs. Excludes all 3,144
   pairs present in the previous train/validation/test subset.
2. Nested data-efficiency study: 500/1,000/2,000/3,000 training matrices with the
   same validation data and fixed settings. 27 new fits plus nine reused fits,
   all validation-selected and frozen before new holdout inference.

Results and reproducibility are in [NEXT_STEPS.md](NEXT_STEPS.md) and
[the full numeric analysis](runs/next_steps_20261001/analysis/RESULTS.md).
All 30 tests pass. All 1,000 holdout examples match the source archive exactly,
including input features and dose units; paired bootstrap arithmetic has an
independent expansion check. Every new checkpoint reproduces its stored
validation predictions exactly on CUDA.

The enhanced model's fresh RMSE is 14.443 ± 0.367 versus reference 21.357 ± 2.247;
the paired relative reduction is 32.4% (marginal cluster interval 29.0–35.8%).
With 500/1,000/2,000 training matrices its mean fresh RMSE is 23.872/18.253/16.100.
The compact model has 3.50M parameters and fresh RMSE 15.019 ± 0.011 at 3,000.
All ± values are sample SD across three seeds, not confidence intervals.

## Equal optimizer-update budget — completed

The subsequently authorized budget experiment is complete: 18 new fits for the
enhanced and compact models at 1,000/2,000/3,000 matrices and seeds 42/43/44.
Each has a 2,400-update maximum, validation every 24 updates, patience of 192
updates and the same learning-rate schedule in update units. Actual updates
total 37,464; summed training time is 24.0 minutes. The exact earlier subsets,
architectures and augmentation were preserved.

At 1,000 matrices, mean validation RMSE improved from 17.831 to 16.095 for the
enhanced model and from 21.152 to 16.396 for compact. At 2,000, the changes are
15.402 to 14.343 and 16.252 to 14.892. The enhanced 2,000-matrix mean is 0.294
above its 3,000-matrix mean while using one third less training data. Enhanced /
3,000 remains the validation accuracy winner at 14.048 ± 0.329; compact /
3,000 is 14.454 ± 0.064. These ± values are seed SD, not confidence intervals.

All 37 correctness tests passed. All 18 new checkpoints reproduce their saved
validation predictions exactly on CUDA; all six unchanged 3,000-matrix controls
also reproduce their earlier predictions and learning trajectories. Independent
audits check all 36 earlier/new validation files, doses/source identities,
example exposure, scheduler timing, stopping patience and configuration ranking.
The three scientific figures were visually checked.

Results and repeat commands are in [STEP_BUDGET.md](STEP_BUDGET.md) and
[the budget-study report](runs/step_budget_20261001/analysis/RESULTS.md).
No new holdout inference was performed in the budget study. Its protocol's
literal 'do not load' wording is stricter than reading the shared prepared
container, which includes unused original test rows; the report records that
caveat. Training and selection use only the training/validation partitions.

Future investigations remain separate: another reserved evaluation set or
independent source, separate weight-scale/bias interventions, and exact
drug-order symmetry. Activation versus calibration was subsequently completed
on 8 October, as recorded below. Seven compact fits still reached the 2,400-update cap.
The 500-matrix subset from the preceding study lacks one drug. Its fresh
holdout has been inspected and must not be represented as untouched after
adaptive development. The requested model experiment stages are complete.

## Report revision 1.1 — completed 1 October 2026

Version 1.1 of the project report now includes fresh confirmation, nested learning
curves, the 18-fit update-budget study, comparison tables, vector plots, and a
complete per-seed appendix. The editable source remains at
`E:\PROJECTS\UNI\DeepSynBa\output\pdf\DeepSynBa_Project_Report.tex`.
The 34-page PDF compiled successfully with bundled Tectonic; all new summary
tables and plot coordinates match the audited CSVs, all 18 appendix rows were
checked, and every rendered page passed visual review. The native editor/compiler
tools were unavailable for this request.

The main PDF at `E:\PROJECTS\UNI\DeepSynBa\DeepSynBa_Project_Report.pdf` is updated.
The older `output\pdf\DeepSynBa_Project_Report.pdf` is locked by another
application, so the identical revised copy is saved beside it as
`output\pdf\DeepSynBa_Project_Report_v1.1.pdf`. Original version 1.0 source/PDF
backups, the revised local source/PDF, build log, evidence fingerprints, and PDF
verification records are preserved in this worktree's `output/pdf/` directory.

## Activation versus initialization — completed 8 October 2026

All 12 new fits completed: ReLU/Softplus × original/near-one calibration ×
seeds 42/43/44, at the exact previous 2,000-matrix training subset. They performed
21,984 optimizer updates and 27.9 minutes of summed timed training. All fits
stopped early. The dedicated archive contains only training/validation data;
this study performs no holdout inference.

Mean validation RMSE is 20.378 ± 2.132 for ReLU/original, 14.480 ± 0.200 for
Softplus/original, 14.408 ± 0.091 for ReLU/near-one and 14.343 ± 0.178 for
Softplus/near-one. Softplus's paired change is −5.898 with original initialization
and −0.065 after calibration. Calibration explains most of the original-ReLU
gap; this does not establish equivalence or split/source robustness.

All 44 integrated preflight tests passed. Every initial model and selected
checkpoint reproduced its saved CUDA outputs exactly; all three repeated
enhancement controls also matched the preceding predictions and trajectories.
Independent metrics, contrast arithmetic, histories, selection and head
diagnostics passed audit. Results and repeat commands are in
[ACTIVATION_INITIALIZATION.md](ACTIVATION_INITIALIZATION.md).

## Report revision 1.2 — completed 8 October 2026

The report now contains the complete activation/calibration protocol, all four
condition means, both RMSE measures, paired conditional/main/interaction
contrasts, a vector comparison figure, initial/selected head diagnostics and
all 12 per-seed records. The executive summary, chronology, conclusions,
remaining research steps and repeat commands also reflect the completed study.

The final 38-page PDF compiled with bundled Tectonic. Independent report checks
verified all new summary/contrast/diagnostic tables, four plot-coordinate
tables and 12 appendix rows, plus 18 evidence hashes. Both previous 30-fit and
18-fit appendices were preserved. All 38 pages were rendered and visually
reviewed; the revised summary, protocol, plot/effects, head diagnostics,
appendix and artifact map were also inspected individually. There were no
margin outliers, overfull boxes, missing glyphs or unresolved references.

All three final PDF copies have the same verified SHA-256:
`348ac9b0e35b8e98e971c9a480929f83135807bd45446c714ebff4ddc250b9f7`.
The final source SHA-256 is
`552207303b42187ee1902b675a8b4af2c018b692021733e99cbafe839527874b`.

- `E:\PROJECTS\UNI\DeepSynBa\DeepSynBa_Project_Report.pdf`
- `E:\PROJECTS\UNI\DeepSynBa\output\pdf\DeepSynBa_Project_Report.pdf`
- `E:\PROJECTS\UNI\DeepSynBa\output\pdf\DeepSynBa_Project_Report_v1.2.pdf`

The previously locked secondary PDF was writable during this revision and is
now updated. The editable source remains in the original `output\pdf` location,
with an identical source/PDF staged in this worktree. Version 1.0 and 1.1
backups, build logs, numeric/visual QA records and report manifests remain in
`output/pdf/`. The native editor/compiler tools returned `Invalid app tool
request`; the successful compilation and verification used the bundled tools.
