"""Portable integrity checks for the public UAV-VLPA code release.

This check intentionally does not import model code or require restricted
assets. It verifies that shipped Python sources compile and that the release
does not contain common private-path markers or protected binary artefacts.
"""
from __future__ import annotations

import argparse
import compileall
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("configs", "data", "models", "evaluation", "training", "utils", "submission_experiments_20260910")
FORBIDDEN_TEXT = (
    re.compile(r"/home/bingxing2", re.I),
    re.compile(r"scx7f09", re.I),
    re.compile(r"[A-Z]:\\Users\\", re.I),
)
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors",
    ".wav", ".flac", ".mp4",
    ".csv", ".json", ".jsonl", ".zip", ".tar", ".gz",
    ".npy", ".npz", ".h5", ".hdf5", ".parquet", ".geojson", ".gpx",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    for directory in SOURCE_DIRS:
        path = root / directory
        if not path.is_dir():
            failures.append(f"missing required source directory: {path}")
            continue
        if not compileall.compile_dir(path, quiet=1):
            failures.append(f"Python compilation failed: {path}")
        for source in path.rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in FORBIDDEN_TEXT):
                failures.append(f"private host path remains: {source.relative_to(root)}")

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"protected artefact committed: {path.relative_to(root)}")

    if failures:
        print("RELEASE CHECK FAILED", file=sys.stderr)
        print("\n".join(f"- {item}" for item in failures), file=sys.stderr)
        raise SystemExit(1)
    print("RELEASE CHECK PASSED: source compiles and no checked private artefacts were found.")


if __name__ == "__main__":
    main()
