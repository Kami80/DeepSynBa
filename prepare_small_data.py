"""Build a reproducible subset of the released NCI-ALMANAC data.

The original split holds out unordered drug pairs across all cell lines.
Sample complete matrices within that split; never split individual dose points.
"""
import argparse
import hashlib
import io
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

from dataset import NciAlmanacDataset

ROOT = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "43c16961da835c8545842d564a8efdfec58bdd2f"
ARCHIVE_SHA256 = "342c43266b85109a3bc11fc3b4ceffe6f4b34cb9f0be89f003314cf202af6e33"
SPLITS = {"train": 1, "val": 2, "test": 3}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_split(frame):
    """Reject reversed-pair leakage and repeated pair/cell experiments."""
    checked = frame.copy()
    pairs = sorted_pair_columns(checked)
    checked["pair_a"], checked["pair_b"] = pairs[:, 0], pairs[:, 1]
    groups = checked.groupby(["pair_a", "pair_b"])["split"].nunique()
    overlaps = int((groups > 1).sum())
    duplicates = int(checked.duplicated(["pair_a", "pair_b", "cell"]).sum())
    if overlaps or duplicates:
        raise ValueError(f"Invalid split: {overlaps} overlapping pairs; {duplicates} duplicate experiments")
    if not checked["split"].isin(SPLITS.values()).all():
        raise ValueError("Expected original split values 1, 2, and 3")
    return {"pairs_crossing_splits": overlaps, "duplicate_experiments": duplicates}


def sorted_pair_columns(frame):
    return np.sort(np.column_stack((frame.drug1.str.strip(), frame.drug2.str.strip())), axis=1)


def sample_split(frame, counts, seed):
    parts = []
    for name, code in SPLITS.items():
        pool = frame.loc[frame.split == code]
        count = counts[name]
        if count < (2 if name == "train" else 1) or count > len(pool):
            raise ValueError(f"Invalid {name} size {count}; available: {len(pool)}")
        parts.append(pool.sample(n=count, random_state=seed + code).sort_index())
    return pd.concat(parts).reset_index(drop=True)


def finite_array(value, shape, label):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {label}: expected finite values with shape {shape}; got {array.shape}")
    return array


def prepare(archive, output, counts, seed):
    archive, output = Path(archive), Path(output)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("data.zip differs from the pinned official release; refusing to unpickle it")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new --output directory")
    with zipfile.ZipFile(archive) as bundle:
        frame = pd.read_csv(bundle.open("data/nci_synergy_metadata.csv"))
        frame.insert(0, "source_row", np.arange(len(frame)))
        full_audit = audit_split(frame)
        selected = sample_split(frame, counts, seed)
        subset_audit = audit_split(selected)
        # Only deserialize the checksum-verified archive from the official repository.
        responses = pickle.loads(bundle.read("data/nci_synergy_dose_response.pickle"))
        embeddings = torch.load(
            io.BytesIO(bundle.read("data/features/NCI-ALMANAC_drug_molformer_embeddings.pt")),
            map_location="cpu", weights_only=True,
        )
        expression = pd.read_csv(bundle.open("data/features/NCI-60_landmark_gex.csv"))

    drugs = sorted(set(selected.drug1.str.strip()) | set(selected.drug2.str.strip()))
    cells = sorted(selected.cell.unique())
    drug_index = {name: i for i, name in enumerate(drugs)}
    cell_index = {name: i for i, name in enumerate(cells)}
    drug_features = np.stack([
        finite_array(embeddings[name].cpu().numpy(), (1, 768), f"embedding {name}")
        for name in drugs
    ])
    missing_expression = {}
    cell_features = []
    for name in cells:
        values = expression[NciAlmanacDataset.convert_cell_name(name)].to_numpy(dtype=np.float32)
        missing_count = int(np.isnan(values).sum())
        if missing_count:
            missing_expression[name] = missing_count
            # Match torch.nan_to_num(gene_expr) in the released model's forward().
            values = np.where(np.isnan(values), 0.0, values)
        cell_features.append(finite_array(values, (976,), f"gene expression {name}")[:, None])
    cell_features = np.stack(cell_features)
    matrices, doses1, doses2 = [], [], []
    for row in selected.itertuples(index=False):
        key = f"{row.drug1}_{row.drug2}_{row.cell}"
        response = responses[key]
        matrices.append(finite_array(response["matrix"], (4, 4), key))
        for name, result in (("x1_dose", doses1), ("x2_dose", doses2)):
            # Released concentrations are molar; the original loader uses micromolar.
            doses = finite_array(response[name], (4,), f"{key} {name}") * 1e6
            if np.any(doses < 0) or np.any(np.diff(doses) < 0):
                raise ValueError(f"Invalid concentration order for {key}")
            result.append(doses)
    arrays = {
        "drug_features": drug_features,
        "cell_features": cell_features,
        "drug1_index": np.array([drug_index[x.strip()] for x in selected.drug1]),
        "drug2_index": np.array([drug_index[x.strip()] for x in selected.drug2]),
        "cell_index": np.array([cell_index[x] for x in selected.cell]),
        "response": np.stack(matrices),
        "dose1": np.stack(doses1),
        "dose2": np.stack(doses2),
        "split": selected.split.to_numpy(dtype=np.int64),
        "source_row": selected.source_row.to_numpy(dtype=np.int64),
    }
    coverage = {}
    training_drugs = set(selected.loc[selected.split == 1, "drug1"].str.strip()) | set(
        selected.loc[selected.split == 1, "drug2"].str.strip())
    training_cells = set(selected.loc[selected.split == 1, "cell"])
    for name, code in SPLITS.items():
        part = selected.loc[selected.split == code]
        split_drugs = set(part.drug1.str.strip()) | set(part.drug2.str.strip())
        split_cells = set(part.cell)
        coverage[name] = {
            "examples": len(part), "drug_pairs": len(np.unique(sorted_pair_columns(part).astype(str), axis=0)),
            "drugs": len(split_drugs), "cell_lines": len(split_cells),
            "drugs_absent_from_training": sorted(split_drugs - training_drugs),
            "cells_absent_from_training": sorted(split_cells - training_cells),
        }
        if split_drugs - training_drugs or split_cells - training_cells:
            print(f"NOTE: {name} includes unseen drugs/cells; inspect manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dataset.npz", **arrays)
    selected.to_csv(output / "metadata.csv", index=False)
    manifest = {
        "source": "https://github.com/hikuru/DeepSynBa",
        "upstream_commit": UPSTREAM_COMMIT,
        "source_archive_sha256": ARCHIVE_SHA256,
        "dataset_sha256": sha256(output / "dataset.npz"),
        "metadata_sha256": sha256(output / "metadata.csv"),
        "seed": seed, "source_examples": len(frame), "selected_examples": len(selected),
        "fraction_used": len(selected) / len(frame),
        "sampling": "uniform without replacement within each released pair-disjoint split",
        "scenario": "new drug combinations; original split labels retained",
        "response_units": "original response scale, untreated reference = 100; no clipping",
        "dose_units": "micromolar (original molar doses multiplied by 1e6)",
        "source_split_audit": full_audit, "subset_split_audit": subset_audit,
        "coverage": coverage, "features": {"drug_dim": 768, "gene_dim": 976},
        "missing_expression_values_filled_with_zero": missing_expression,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved small dataset to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "small")
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.archive, args.output, {name: getattr(args, name) for name in SPLITS}, args.seed)


if __name__ == "__main__":
    main()
