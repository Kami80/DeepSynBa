"""Prepare checksum-verified external DeepSynBa dependencies for my research."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent
COMMIT = "43c16961da835c8545842d564a8efdfec58bdd2f"
SOURCE_URL = f"https://codeload.github.com/hikuru/DeepSynBa/zip/{COMMIT}"
MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024
# SHA-256 and byte size of the original, unmodified Git blobs at COMMIT.
REFERENCE_FILES = {
    "blocks/__init__.py": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0),
    "blocks/drug_encoder.py": ("4e5b71c13d12ac4c58944324459828071e52454668257dc4c1d06b169810ba5d", 3655),
    "blocks/predictor.py": ("8ffdd5f8ae3343d06270a045e7e97d6b86016997a3140353126415ed6cb98b49", 3039),
    "config.yaml": ("f122de982fc7ee126b50fba0771bee25540e6eb39af291fe18c89fb2aa168575", 2061),
    "dataset.py": ("5a06ad080d3aab1b810b1ab1e6a1d86e9f2439706dbb14b3b5e0b226ff362d05", 5027),
    "deepsynba.py": ("e00f65593a166042badf387d832bd442ca8b968ac48622ca3c7a7986652bd915", 6564),
    "data.zip": ("342c43266b85109a3bc11fc3b4ceffe6f4b34cb9f0be89f003314cf202af6e33", 18895596),
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def patch_model(name, content):
    """Apply the same single-example shape fix used in the recorded research."""
    if name == "deepsynba.py":
        old = (
            b"        in1 = torch.cat((gene_expr, drug1_features), dim=1).squeeze()\n"
            b"        in2 = torch.cat((gene_expr, drug2_features), dim=1).squeeze()"
        )
        new = (
            b"        # Keep the batch dimension when evaluating a single example.\n"
            b"        in1 = torch.cat((gene_expr, drug1_features), dim=1).squeeze(-1)\n"
            b"        in2 = torch.cat((gene_expr, drug2_features), dim=1).squeeze(-1)"
        )
        if content.count(old) != 1:
            raise ValueError("The pinned model's batch-dimension patch could not be applied")
        return content.replace(old, new).rstrip(b"\r\n") + b"\n"
    if name == "blocks/predictor.py":
        old = b"self.prediction_head(input.squeeze())"
        if content.count(old) != 2:
            raise ValueError("The pinned predictor's batch-dimension patch could not be applied")
        return content.replace(old, b"self.prediction_head(input)")
    return content


def load_reference(archive):
    """Read only allowlisted members; verify everything before installing files."""
    files = {}
    with zipfile.ZipFile(archive) as bundle:
        for name, (expected_hash, expected_size) in REFERENCE_FILES.items():
            members = [member for member in bundle.infolist()
                       if member.filename == name or member.filename.endswith("/" + name)]
            if len(members) != 1 or members[0].file_size != expected_size:
                raise ValueError(f"Missing, ambiguous, or incorrectly sized reference file: {name}")
            content = bundle.read(members[0])
            if digest(content) != expected_hash:
                raise ValueError(f"Reference checksum mismatch: {name}")
            files[name] = patch_model(name, content)
    return files


def equivalent(name, existing, expected):
    # Preserve matching Windows checkouts without changing historical byte hashes.
    if name.endswith((".py", ".yaml")):
        return existing.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n")
    return existing == expected


def install_reference(files, output):
    output = Path(output).resolve()
    targets = {}
    # Check every destination first; never silently replace a locally edited file.
    for name, content in files.items():
        target = (output / name).resolve()
        if not target.is_relative_to(output):
            raise ValueError(f"Reference path escapes the output directory: {name}")
        if target.exists() and not equivalent(name, target.read_bytes(), content):
            raise FileExistsError(f"Preserving differing local file: {target}")
        targets[name] = target
    cache = (output / ".reference-cache").resolve()
    if not cache.is_relative_to(output):
        raise ValueError("Reference cache escapes the output directory")
    for name, target in targets.items():
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(files[name])
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {
        "reference_repository": "https://github.com/hikuru/DeepSynBa",
        "reference_commit": COMMIT,
        "modifications": "Batch axis preserved for single-example evaluation",
        "original_sha256": {name: values[0] for name, values in REFERENCE_FILES.items()},
        "installed_sha256": {name: digest(path.read_bytes()) for name, path in targets.items()},
    }
    (cache / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def download_reference(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "DeepSynBa-research-setup"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as stream:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Reference download exceeded the 80 MB limit")
                stream.write(chunk)
        load_reference(partial)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, help="Offline ZIP of the pinned original source")
    parser.add_argument("--output", type=Path, default=ROOT, help="Local dependency directory")
    args = parser.parse_args()
    output = args.output.resolve()
    archive = args.source_zip
    if archive is None:
        archive = output / ".reference-cache" / f"DeepSynBa-{COMMIT}.zip"
        if not archive.exists():
            print(f"Downloading pinned reference {COMMIT}...", flush=True)
            download_reference(archive)
    install_reference(load_reference(archive), output)
    print(f"Verified {len(REFERENCE_FILES)} reference files; local dependencies ready in {output}")
    print("Next: python prepare_small_data.py")


if __name__ == "__main__":
    main()
