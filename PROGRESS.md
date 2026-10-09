# Completed research stages

This repository records my small-data reproduction and controlled follow-up
experiments on DeepSynBa, completed between 29 September and 8 October 2026.
The original model and processed data are external dependencies credited in
[ATTRIBUTION.md](ATTRIBUTION.md). The summaries below link to the published
experiment evidence; full checkpoints and prediction arrays remain in the local
run archive.

## Small-data reproduction and diagnostics

I prepared a deterministic 5,000-matrix subset: 3,000 training matrices, 1,000
validation matrices, and 1,000 test matrices, with unordered drug pairs kept
separate between partitions. The CPU and GPU setup runs completed, followed by
three reference fits with a 100-epoch cap. Their original-test RMSE was
**21.327 ± 2.095**. Diagnostics identified zero Hill-slope outputs and motivated
controlled changes to the parameter heads.

[Guide](REPRODUCE_SMALL.md) · [Reference results](results/extended_100/RESULTS.md).

## Ten-configuration comparison

I compared positive-output repair, smaller network widths, drug-order augmentation,
and bias controls across ten configurations and three seeds (30 fits). Validation
selected positive outputs plus augmentation at the original widths. Original-test
RMSE was **14.605 ± 0.390**, versus **21.327 ± 2.095** for the reference.
The compact selected configuration used 3.50 million parameters and reached
**15.161 ± 0.082**. These test comparisons are exploratory because the original
test subset had already been inspected. Positive-output repair changes both
activation and initialization, motivating the later factorial study.

[Comparison results](results/comparison_small/RESULTS.md).

## Fresh confirmation and nested training sizes — 1 October

I evaluated nine frozen fits on 1,000 previously unexamined matrices from 323
unordered drug pairs. The holdout excluded all pairs present in the earlier
5,000-matrix subset. Enhanced RMSE was **14.443 ± 0.367**, versus reference
**21.357 ± 2.247** and compact **15.019 ± 0.011**. The enhanced model's relative
RMSE reduction was **32.4%**, with a paired drug-pair bootstrap 95% interval of
**29.0–35.8%**, conditional on the frozen fitted models.

The learning-curve study added 27 fits at 500/1,000/2,000 training matrices and
reused nine 3,000-matrix fits. Enhanced mean fresh RMSE was
23.872/18.253/16.100/14.443 at 500/1,000/2,000/3,000 matrices. Seven new fits
reached the epoch cap; the 500-matrix subset omitted one drug. The holdout is
same-source evidence and has now been inspected.

[Guide](NEXT_STEPS.md) · [Results](results/next_steps_20261001/RESULTS.md).

## Equal optimizer-update budget — 1 October

I trained 18 further models: enhanced and compact configurations, three training
sizes, and three seeds, with a shared maximum of 2,400 updates. Validation was
checked every 24 updates, with stopping patience and scheduling defined in update
units. At 2,000 training matrices, enhanced validation RMSE improved to
**14.343 ± 0.178**, 0.294 above the 3,000-matrix mean while using one third less
training data. This study performed no new holdout inference. Its results document
the shared prepared-container caveat and the remaining update-cap limitations.

[Guide](STEP_BUDGET.md) · [Results](results/step_budget_20261001/RESULTS.md).

## Activation and initialization — 8 October

I completed 12 fits at the fixed 2,000-matrix training subset: ReLU/Softplus ×
original/near-one initialization × three seeds. Validation RMSE was:

| Activation | Original initialization | Near-one calibration |
| --- | ---: | ---: |
| ReLU | 20.378 ± 2.132 | 14.408 ± 0.091 |
| Softplus | 14.480 ± 0.200 | 14.343 ± 0.178 |

Calibration recovered most of the original-ReLU gap. The paired Softplus change
was −5.898 with original initialization and −0.065 after calibration. This does
not establish equivalence or isolate weight scaling from bias initialization.
The dedicated prepared archive contains training and validation data only, and
no holdout inference was performed.

[Guide](ACTIVATION_INITIALIZATION.md) ·
[Results](results/activation_initialization_20261008/RESULTS.md).

All ± values above are sample SD across seeds 42, 43, and 44. Future work includes
another reserved evaluation source, weight-scale versus bias interventions, exact
drug-order symmetry, and validation of predicted biological parameters.

[Two-page report for Professor Taştan](reports/DeepSynBa_Two_Page_Report.pdf).
