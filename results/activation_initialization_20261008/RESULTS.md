# Activation and initialization — completed

Validation-only 2×2 factorial: the same 2,000 training matrices, 1,000 validation matrices, full architecture, augmentation and optimizer-update schedule; seeds 42/43/44.

| Activation / initialization | All-16 RMSE | Combination-9 RMSE | Cap fits |
|---|---:|---:|---:|
| ReLU / original | 20.378 ± 2.132 | 22.499 ± 2.064 | 0/3 |
| Softplus / original | 14.480 ± 0.200 | 17.270 ± 0.237 | 0/3 |
| ReLU / near-one | 14.408 ± 0.091 | 17.189 ± 0.100 | 0/3 |
| Softplus / near-one | 14.343 ± 0.178 | 17.126 ± 0.185 | 0/3 |

All ± values are sample SD over three paired seeds, not confidence intervals. Negative conditional/main effects favor Softplus or near-one calibration. Interaction measures how the activation effect changes with initialization.

| All-16 paired contrast | Mean change | Seed SD |
|---|---:|---:|
| activation_original | -5.898 | 2.123 |
| activation_near_one | -0.065 | 0.090 |
| initialization_relu | -5.970 | 2.042 |
| initialization_softplus | -0.137 | 0.252 |
| interaction | +5.833 | 2.209 |
| average_activation | -2.982 | 1.018 |
| average_initialization | -3.054 | 0.947 |

Validation-ranked cell: **softplus_near_one**. All three Softplus/near-one controls exactly reproduce the preceding 2,000-matrix enhanced fits.

The improvement cannot be attributed to Softplus alone. Its paired RMSE change is -5.898 with original initialization, compared with -0.065 after near-one calibration. Calibration alone changes ReLU RMSE by -5.970; changing both yields -6.036. Either intervention recovers most of the observed reduction on this split. The smaller calibrated activation difference does not establish equivalence. 3/3 original-ReLU seeds have at least one entirely zero head on validation at their selected checkpoints; this observation supports an optimization explanation but does not establish the complete training mechanism.

The calibration factor combines 0.01 weight scaling and an activation-specific bias targeting output one. It does not isolate bias from scale. Early stopping changes realized updates; all cells share the same maximum/checkpoint/stopping rule. Initial and selected h1/h2/alpha outputs are recorded, with exact zeros and below-0.01 fractions. Softplus positivity is built into the function and is not proof of biological parameter recovery.

This experiment describes algorithmic changes under a fixed adaptive validation protocol; it supplies neither a new holdout claim, a generalization/significance interval nor split/source robustness. Both previously inspected holdouts remain development evidence. Independent-source evaluation, strong external baselines, multiple splits and drug/cell generalization remain future research.

Actual optimizer updates: 21,984. Summed timed training: 27.9 minutes. Histories, predictions, diagnostics and protocol/data hashes audited; exact initial and selected CUDA reloads: 12/12.

Evidence: `protocol.json`, `selection.json`, `completed.json`, training files and independent CSVs/verification JSON in this experiment. Runtime is descriptive; no controlled speed claim.
