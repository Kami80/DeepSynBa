"""Audit holdout identities, units/features and bootstrap arithmetic from source."""
import argparse
import io
import pickle
import zipfile

import numpy as np
import pandas as pd
import torch

from dataset import NciAlmanacDataset
from next_steps_statistics import cluster_sums, resampled_mean_rmse
from prepare_small_data import ARCHIVE_SHA256, sha256
from run_next_steps import FINALISTS, ROOT, SEEDS, load_protocol, read_json, validate_file
from train_small import dump_json


def verify(output):
    protocol, _, _, held = load_protocol(output)
    validate_file(ROOT / "data.zip", ARCHIVE_SHA256)
    metadata = pd.read_csv(output / "holdout/metadata.csv")
    with zipfile.ZipFile(ROOT / "data.zip") as bundle:
        raw_metadata = pd.read_csv(bundle.open("data/nci_synergy_metadata.csv"))
        responses = pickle.loads(bundle.read("data/nci_synergy_dose_response.pickle"))
        embeddings = torch.load(io.BytesIO(bundle.read("data/features/NCI-ALMANAC_drug_molformer_embeddings.pt")),
                                map_location="cpu", weights_only=True)
        expression = pd.read_csv(bundle.open("data/features/NCI-60_landmark_gex.csv"))
    for index, row in enumerate(metadata.itertuples(index=False)):
        original = raw_metadata.iloc[row.source_row]
        for key in ("drug1", "drug2", "cell", "split"):
            if getattr(row, key) != original[key]:
                raise AssertionError(f"Source row identity differs: {row.source_row} {key}")
        raw = responses[f"{row.drug1}_{row.drug2}_{row.cell}"]
        np.testing.assert_array_equal(held["response"][index].numpy(), np.asarray(raw["matrix"], dtype=np.float32))
        for stored, source in (("dose1", "x1_dose"), ("dose2", "x2_dose")):
            np.testing.assert_array_equal(held[stored][index].numpy(), np.asarray(raw[source], dtype=np.float32) * 1e6)
        for side in ("drug1", "drug2"):
            feature = held["drug_features"][held[f"{side}_index"][index]].numpy()
            np.testing.assert_array_equal(feature, embeddings[getattr(row, side).strip()].numpy())
        gene = expression[NciAlmanacDataset.convert_cell_name(row.cell)].to_numpy(dtype=np.float32)
        gene[np.isnan(gene)] = 0
        np.testing.assert_array_equal(held["cell_features"][held["cell_index"][index]].numpy().ravel(), gene)
    # Independently expand 25 actual bootstrap draws, preserving each repeated
    # cluster's complete matrices, and compare to sufficient-statistic arithmetic.
    pairs = np.sort(metadata[["drug1", "drug2"]].apply(lambda column: column.str.strip()).to_numpy(), axis=1)
    _, groups = np.unique(pairs.astype(str), axis=0, return_inverse=True)
    predicted = np.stack([np.stack([
        np.load(output / "confirmation" / name / f"seed_{seed}/holdout_predictions.npz",
                allow_pickle=False)["predicted"] for seed in SEEDS]) for name in FINALISTS])
    truth = held["response"].numpy().astype(np.float64)
    error = predicted.astype(np.float64) - truth
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(groups.max() + 1, size=(25, groups.max() + 1))
    for interior in (False, True):
        observed = truth[:, 1:, 1:] if interior else truth
        forecasts = predicted[:, :, :, 1:, 1:] if interior else predicted
        sums, counts, points = cluster_sums(observed, forecasts, groups)
        fast = resampled_mean_rmse(sums, counts, points, draws)
        errors = error[:, :, :, 1:, 1:] if interior else error
        for replicate, sampled in enumerate(draws):
            rows = np.concatenate([np.flatnonzero(groups == cluster) for cluster in sampled])
            expanded = np.sqrt(np.mean(errors[:, :, rows] ** 2, axis=(2, 3, 4))).mean(axis=1)
            np.testing.assert_allclose(fast[replicate], expanded, rtol=1e-12, atol=1e-12)
    result = {"holdout_source_rows_checked": len(metadata),
        "all_response_values_exact": True, "all_micromolar_doses_exact": True,
        "all_drug_features_exact": True, "all_cell_features_exact": True,
        "bootstrap_draws_independently_expanded_per_endpoint": len(draws),
        "bootstrap_arithmetic_matches_explicit_cluster_expansion": True,
        "protocol_sha256": sha256(output / "protocol.json"),
        "verification_source_sha256": sha256(ROOT / "verify_next_steps.py")}
    directory = output / "analysis"
    directory.mkdir(exist_ok=True)
    dump_json(directory / "source_verification.json", result)
    print(result)


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/next_steps_20261001")
    verify(parser.parse_args().experiment.resolve())
