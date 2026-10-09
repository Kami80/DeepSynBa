"""Explicit experimental variants; the released model files remain unchanged."""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from deepsynba import DeepSynBa

POSITIVE_KEYS = ("h1_mean", "h2_mean", "alpha_mean")
EPSILON = 1e-4


def variant_catalog():
    """Eight factorial cells plus two bias ablations of the combined variant."""
    cells = [
        ("reference", "Reference", False, False, False),
        ("positive", "Positive outputs", True, False, False),
        ("small", "Smaller model", False, True, False),
        ("swap", "Drug-order augmentation", False, False, True),
        ("positive_small", "Positive + smaller", True, True, False),
        ("positive_swap", "Positive + augmentation", True, False, True),
        ("small_swap", "Smaller + augmentation", False, True, True),
        ("positive_small_swap", "All three", True, True, True),
    ]
    variants = [dict(id=name, label=label, positive=positive, small=small,
                     swap=swap, bias=True, bias_l2=0.0)
                for name, label, positive, small, swap in cells]
    variants += [dict(variants[-1], id="positive_small_swap_bias_l2",
                      label="All three + bias penalty", bias_l2=0.1),
                 dict(variants[-1], id="positive_small_swap_no_bias",
                      label="All three, no bias", bias=False)]
    return variants


class PositiveOutput(nn.Module):
    def forward(self, value):
        return F.softplus(value) + EPSILON


def build_model(base_config, variant):
    config = deepcopy(base_config)
    if variant["small"]:
        config["model_params"].update(gene_channel=128, emb_size=512)
    model = DeepSynBa(config)
    if variant["positive"]:
        for key in POSITIVE_KEYS:
            layers = model.predictors[key].prediction_head
            if not isinstance(layers[-1], nn.ReLU) or not isinstance(layers[-2], nn.Linear):
                raise ValueError(f"Unexpected released positive head structure: {key}")
            layers[-1] = PositiveOutput()
            # Scale the existing seeded weights, introducing no extra RNG draws.
            # Near-one output initialization is part of the positive-output package.
            with torch.no_grad():
                layers[-2].weight.mul_(0.01)
                layers[-2].bias.fill_(math.log(math.expm1(1.0 - EPSILON)))
    if not variant["bias"]:
        model.bias = False
        del model.bias_predictor1
        del model.bias_predictor2
    return model, config


def swap_examples(inputs, target, mask):
    """Swap features AND physical dose axes; never mutate the source dataset."""
    drug1, drug2, cell, dose1, dose2 = inputs
    mask = mask.to(device=target.device, dtype=torch.bool)
    if mask.shape != (len(target),):
        raise ValueError("Swap mask must have one boolean per matrix")
    select = lambda first, second: torch.where(
        mask.reshape((-1,) + (1,) * (first.ndim - 1)), second, first)
    return (select(drug1, drug2), select(drug2, drug1), cell,
            select(dose1, dose2), select(dose2, dose1)), select(target, target.transpose(1, 2))


def training_objective(outputs, target, coefficient):
    mse = F.mse_loss(outputs["dose_response"], target)
    penalty = outputs["bias"].square().mean() if coefficient else mse.new_zeros(())
    return mse + coefficient * penalty if coefficient else mse, mse, penalty
