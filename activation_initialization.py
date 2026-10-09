"""Separate the activation and calibration components of the positive-head package."""
from copy import deepcopy
import math

from ablation_model import EPSILON, POSITIVE_KEYS, PositiveOutput
from deepsynba import DeepSynBa
from torch import nn
import numpy as np
import torch


CONDITIONS = (
    {"id": "relu_original", "activation": "relu", "initialization": "original"},
    {"id": "softplus_original", "activation": "softplus", "initialization": "original"},
    {"id": "relu_near_one", "activation": "relu", "initialization": "near_one"},
    {"id": "softplus_near_one", "activation": "softplus", "initialization": "near_one"},
)
SEEDS = (42, 43, 44)


def build_factorial_model(base_config, condition):
    activation, initialization = condition["activation"], condition["initialization"]
    if activation not in ("relu", "softplus") or initialization not in ("original", "near_one"):
        raise ValueError("Unknown activation/initialization condition")
    if condition["id"] != f"{activation}_{initialization}":
        raise ValueError("Condition identity differs from its factors")
    config = deepcopy(base_config)
    model = DeepSynBa(config)
    for key in POSITIVE_KEYS:
        layers = model.predictors[key].prediction_head
        if not isinstance(layers[-1], nn.ReLU) or not isinstance(layers[-2], nn.Linear):
            raise ValueError(f"Unexpected released head structure: {key}")
        if activation == "softplus":
            layers[-1] = PositiveOutput()
        if initialization == "near_one":
            # Same seeded weights for all cells; no new random draws. Calibrate
            # the output to one using the appropriate inverse activation.
            bias = 1.0 if activation == "relu" else math.log(math.expm1(1.0 - EPSILON))
            with torch.no_grad():
                layers[-2].weight.mul_(0.01)
                layers[-2].bias.fill_(bias)
    return model, config


@torch.no_grad()
def collect_outputs(model, loader, device):
    """Record predictions and the three affected heads without changing RNG or weights."""
    model.eval()
    values = {key: [] for key in (*POSITIVE_KEYS, "dose_response")}
    observed = []
    for batch in loader:
        *inputs, target = [value.to(device, non_blocking=True) for value in batch]
        output = model(*inputs)
        for key in values:
            values[key].append(output[key].cpu().numpy())
        observed.append(target.cpu().numpy())
    return np.concatenate(observed), {key: np.concatenate(parts) for key, parts in values.items()}


def head_summary(outputs):
    result = {}
    for key in POSITIVE_KEYS:
        value = np.asarray(outputs[key], dtype=np.float64).ravel()
        if not len(value) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Invalid positive-head outputs")
        result[key] = {"min": float(value.min()), "median": float(np.median(value)),
            "mean": float(value.mean()), "max": float(value.max()),
            "exact_zero_fraction": float(np.mean(value == 0)),
            "below_0_01_fraction": float(np.mean(value < 0.01)), "examples": int(value.size)}
    return result
