"""Verify an unpacked PADGain-DualWindow release, including its matrices and labels."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from workloads import catalog, target_rows


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    manifest = json.loads((root / "sha256.json").read_text(encoding="utf-8"))
    for name, checksum in manifest.items():
        require(file_hash(root / name) == checksum, f"SHA-256 mismatch: {name}")
    cases = {c["case_id"]: c for c in read_lines(root / "workloads.jsonl")}
    require(len(cases) == 197 and len(catalog()) == len(cases) == len(read_lines(root / "workloads.jsonl")),
            "Catalog size mismatch")
    for spec in catalog():
        case = cases[spec["case_id"]]
        require(all(case[k] == v for k, v in spec.items()), spec["case_id"])
        require(case["pad_pitch_words"] == case["logical_words"] + case["pad_words"], spec["case_id"])
    folds = json.loads((root / "folds.json").read_text(encoding="utf-8"))
    members = []
    for fold in folds:
        for key in fold["test_case_ids"]:
            require(cases[key]["pad_pitch_words"] == fold["held_out_pad_pitch_words"], key)
            members.append(key)
    require(sorted(members) == sorted(cases), "Fold membership mismatch")
    evidence = {e["path"]: e for e in read_lines(root / "evidence_index.jsonl")}
    for path, entry in evidence.items():
        require(manifest.get(path) == entry["sha256"] and (root / path).stat().st_size == entry["bytes"], path)
    measured = read_lines(root / "measured.jsonl")
    reported = read_lines(root / "reported.jsonl")
    ids = [r["observation_id"] for r in measured + reported]
    require(len(set(ids)) == len(ids), "Duplicate observation IDs")
    for row in measured + reported:
        key = row["case_id"]
        spec = cases[key]
        require(all(row[k] == spec[k] for k in ("logical_words", "second_window_words", "pad_words")), key)
        require(row["target_l2_percent"] == spec["l2_percent"] and row["speedup"] > 0, key)
        require((row["base_l2_read_miss_sectors"] is None) == (row["pad_l2_read_miss_sectors"] is None), key)
    for row in measured:
        key = row["observation_id"]
        require(row["rows"] == target_rows(row["l2_bytes"], row["target_l2_percent"]), key)
        require(row["rounds"] == 40 and row["base_verified"] and row["pad_verified"], key)
        require(row["base_pitch_words"] == row["logical_words"] and
                row["pad_pitch_words"] == cases[row["case_id"]]["pad_pitch_words"], key)
        require(row["base_ms"] > 0 and row["pad_ms"] > 0 and
                math.isclose(row["base_ms"] / row["pad_ms"], row["speedup"]), key)
        require(all(p in evidence for p in row["evidence"].values()), key)
        native_plan = read_lines(root / row["evidence"]["native"])[0]
        for arm in ("base", "pad"):
            mean_blocks = cases[row["case_id"]][f"{arm}_mean_128b_blocks_per_row"]
            require(mean_blocks * row["rows"] == native_plan[f"{arm}_tag_count"], key)
    for row in reported:
        require(row["base_ms"] is None and row["pad_ms"] is None and row["label_status"] == "reported",
                row["observation_id"])
    release = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    distinct = {r["case_id"] for r in measured + reported}
    require(release["cases_without_observations"] == len(cases.keys() - distinct) and
            release["native_pairs"] == len(measured) and release["reported_pairs"] == len(reported),
            "Coverage totals mismatch")
    sample = root / "example"
    meta = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    link = json.loads((sample / "source.json").read_text(encoding="utf-8"))
    reference = next(row for row in measured if row["observation_id"] == link["observation_id"])
    require(reference["rows"] == link["rows"] == meta["rows"] and
            reference["case_id"] == link["case_id"] == meta["case_id"], "Example source mismatch")
    base = np.load(sample / meta["base_file"], mmap_mode="r", allow_pickle=False)
    padded = np.load(sample / meta["padded_file"], mmap_mode="r", allow_pickle=False)
    width, rows = meta["logical_words"], meta["rows"]
    require(base.shape == (rows, width) and padded.shape == (rows, width + meta["pad_words"])
            and base.dtype.str == padded.dtype.str == "<u4", "Example NPY header")
    require(np.array_equal(base, padded[:, :width]) and np.all(padded[:, width:] == 0xDEADC0DE),
            "Logical data or tail mismatch")
    for row in (0, rows // 2, rows - 1):
        for col in (0, 7, 24, 31, width - 1):
            expected = ((row + 1) * 2654435761 ^ (col + 1) * 2246822519) & 0xFFFFFFFF
            require(int(base[row, col]) == expected, f"CUDA value mismatch at {row},{col}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    verify(args.directory)
    print("Verified all hashes, case groups, labels and example layouts.")
