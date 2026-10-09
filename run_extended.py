"""Run a fixed three-seed experiment; freeze validation selections before testing."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from prepare_small_data import ROOT, sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def execute(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", cwd=ROOT)
        for line in process.stdout:
            log.write(line)
            log.flush()
            # Keep progress readable; preserve complete stdout/stderr in the log.
            if line.startswith(("Epoch ", "Early stopping", "Device:", "Examples:", "Traceback")):
                print(line, end="", flush=True)
        if process.wait() != 0:
            raise RuntimeError(f"Command failed. See {log_path}")


def run_suite(output, seeds=(42, 43, 44), epochs=100):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = read_json(ROOT / "data/small/manifest.json")
    protocol = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds), "max_epochs": epochs, "device": "cuda",
        "early_stopping_patience": 8,
        "dataset_sha256": manifest["dataset_sha256"],
        "subset_seed": manifest["seed"],
        "counts": {name: details["examples"] for name, details in manifest["coverage"].items()},
        "source_files": {name: sha256(ROOT / name) for name in
                         ("config.small.yaml", "config.yaml", "train_small.py", "run_extended.py",
                          "deepsynba.py", "blocks/predictor.py", "blocks/drug_encoder.py")},
        "checkpoint_selection": "Minimum validation MSE within each seed; no test evaluation during training",
        "diagnostic_model_selection": "Lowest validation RMSE across all three seeds, before test evaluation",
        "validation_example_selection": "Nearest matrix-RMSE ranks at the 10th, 50th, 90th, and 99th percentiles",
        "seed_summary": "Arithmetic mean and sample standard deviation (ddof=1); not confidence intervals",
    }
    write_json(output / "protocol.json", protocol)
    selections = []
    for seed in seeds:
        print(f"TRAINING seed {seed} with test evaluation deferred", flush=True)
        run_dir = output / f"seed_{seed}"
        execute([sys.executable, str(ROOT / "train_small.py"), "--device", "cuda", "--seed", str(seed),
                 "--epochs", str(epochs), "--defer-test", "--output", str(run_dir)],
                output / f"seed_{seed}_training.log")
        result = read_json(run_dir / "metrics.json")
        recorded_data = read_json(run_dir / "data_manifest.json")
        if result["test_evaluated"] or (run_dir / "test_predictions.npz").exists():
            raise RuntimeError("Test evaluation happened before model selection was frozen")
        if recorded_data["dataset_sha256"] != protocol["dataset_sha256"]:
            raise RuntimeError("Dataset changed between seeds")
        selections.append({"seed": seed, "best_epoch": result["best_epoch"],
                           "validation_rmse": result["validation_all_16_doses"]["rmse"],
                           "checkpoint_sha256": sha256(run_dir / "best.pt")})
    chosen = min(selections, key=lambda row: row["validation_rmse"])
    selection = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                 "diagnostic_seed": chosen["seed"], "selections": selections}
    write_json(output / "selection.json", selection)
    print(f"Selections frozen. Diagnostic model: seed {chosen['seed']}", flush=True)
    for seed in seeds:
        run_dir = output / f"seed_{seed}"
        expected = next(row for row in selections if row["seed"] == seed)
        if sha256(run_dir / "best.pt") != expected["checkpoint_sha256"]:
            raise RuntimeError("Selected checkpoint changed before test evaluation")
        print(f"FINAL TEST EVALUATION seed {seed}", flush=True)
        execute([sys.executable, str(ROOT / "train_small.py"), "--device", "cuda", "--seed", str(seed),
                 "--epochs", str(epochs), "--checkpoint", str(run_dir / "best.pt"),
                 "--output", str(run_dir / "evaluation")], output / f"seed_{seed}_evaluation.log")
    write_json(output / "completed.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                                           "seeds": list(seeds), "status": "all_training_and_tests_completed"})
    print(f"All runs completed: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/extended_100")
    args = parser.parse_args()
    run_suite(args.output)
