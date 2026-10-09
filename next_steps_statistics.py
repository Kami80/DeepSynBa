"""Paired drug-pair cluster bootstrap for fixed, matched model seeds."""
import numpy as np


def cluster_sums(truth, predictions, groups):
    truth = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    groups = np.asarray(groups)
    if truth.ndim != 3 or predictions.ndim != 5 or predictions.shape[2:] != truth.shape:
        raise ValueError("Expected truth [matrix,row,column] and predictions [model,seed,matrix,row,column]")
    if groups.shape != (len(truth),) or len(truth) < 1:
        raise ValueError("One nonempty cluster label per matrix is required")
    if not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("Non-finite bootstrap input")
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    errors = ((predictions - truth) ** 2).sum(axis=(-1, -2))
    sums = np.stack([np.stack([np.bincount(inverse, weights=seed, minlength=len(counts))
                              for seed in model]) for model in errors])
    return sums, counts, int(np.prod(truth.shape[1:]))


def resampled_mean_rmse(sums, counts, dose_points, draws):
    """Each draw lists cluster IDs; retain each cluster's unequal matrix count."""
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != len(counts):
        raise ValueError("Draw exactly the number of observed clusters each time")
    if np.any(draws < 0) or np.any(draws >= len(counts)):
        raise ValueError("Invalid sampled cluster ID")
    weights = np.stack([np.bincount(row, minlength=len(counts)) for row in draws])
    numerator = np.einsum("msg,bg->bms", sums, weights)
    denominator = (weights @ counts)[:, None, None] * dose_points
    return np.sqrt(numerator / denominator).mean(axis=-1)


def paired_cluster_bootstrap(truth, predictions, groups, labels, replicates=5000, seed=20261003):
    if replicates < 1 or len(labels) != 3 or len(labels) != len(predictions) or len(set(labels)) != len(labels):
        raise ValueError("Positive bootstrap count and three unique matching model labels required")
    sums, counts, points = cluster_sums(truth, predictions, groups)
    observed = np.sqrt(sums.sum(axis=-1) / (counts.sum() * points)).mean(axis=-1)
    rng = np.random.default_rng(seed)
    pieces = []
    for start in range(0, replicates, 250):
        draws = rng.integers(len(counts), size=(min(250, replicates - start), len(counts)))
        pieces.append(resampled_mean_rmse(sums, counts, points, draws))
    samples = np.concatenate(pieces)
    comparisons = []
    # Two frozen comparisons to reference, and the descriptive compact/full gap.
    for first, second in ((1, 0), (2, 0), (2, 1)):
        delta = samples[:, first] - samples[:, second]
        bounds = np.quantile(delta, [0.025, 0.975])
        reduction = 100 * (1 - samples[:, first] / samples[:, second])
        comparison = {"model": labels[first], "baseline": labels[second],
            "mean_seed_rmse_difference": float(observed[first] - observed[second]),
            "percentile_95_ci": bounds.tolist(),
            "relative_reduction_percent": float(100 * (1 - observed[first] / observed[second])),
            "relative_reduction_percentile_95_ci": np.quantile(reduction, [0.025, 0.975]).tolist()}
        comparisons.append(comparison)
    return {"clusters": len(counts), "matrices": int(counts.sum()), "replicates": replicates,
            "seed": seed, "comparisons": comparisons}, samples
