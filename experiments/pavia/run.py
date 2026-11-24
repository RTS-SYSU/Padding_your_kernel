"""Freeze an address-only Pavia University experiment, then validate it on CUDA."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
from scipy.io import loadmat


BANDS, PIXELS, SECTOR_WORDS = 103, 1600, 16
PAIRS = PIXELS * (PIXELS - 1) // 2
PAD = (-BANDS) % SECTOR_WORDS
NCU = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.1\target\windows-desktop-win7-x64\ncu.exe")
METRICS = {
    "issued": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l2_read": "lts__t_sectors_srcunit_tex_op_read.sum",
    "l2_miss": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
}


def digest(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf8") as dest:
        json.dump(value, dest, indent=2)
        dest.write("\n")


def oracle(rows):
    rows = rows.astype(np.int64)
    best, pair = -1, 0
    for row in range(1, len(rows)):
        delta = rows[:row] - rows[row]
        scores = np.sum(delta * delta, axis=1)
        col = int(scores.argmax())
        if scores[col] > best:
            best, pair = int(scores[col]), row * (row - 1) // 2 + col
    return {"score": best, "pair": pair}


def geometry(pitch):
    issued = unique = 0
    previous_end = -1
    for row in range(PIXELS):
        begin = row * pitch
        for band in range(0, BANDS, 32):
            first = (begin + band) // SECTOR_WORDS
            last = (begin + min(band + 31, BANDS - 1)) // SECTOR_WORDS
            issued += (last - first + 1) * (PIXELS - 1)
            unique += last - max(previous_end, first - 1)
            previous_end = last
    return {"issued": issued, "unique": unique, "input_bytes": PIXELS * pitch * 2}


def freeze(root, case=None):
    mat = root / "PaviaU.mat"
    image = loadmat(mat)["paviaU"]
    if image.shape != (610, 340, BANDS) or image.dtype.kind != "u" or image.dtype.itemsize != 2:
        raise ValueError("Expected original 610 x 340 x 103 uint16 Pavia University")
    cases = {}
    regions = {"top_left": (0, 0), "center": (285, 150)}
    if case:
        regions = {case: regions[case]}
    for name, (y, x) in regions.items():
        rows = image[y:y+40, x:x+40, :].reshape(PIXELS, BANDS).astype("<u2")
        path = root / f"{name}.u16"
        with path.open("xb") as output:
            output.write(rows.tobytes())
        cases[name] = dict(y=y, x=x, sha256=digest(path), expected=oracle(rows))
    save(root / "plan.json", dict(pad=PAD, pairs=PAIRS, pairs_per_batch=12, cases=cases,
                                 source_url=(root / "source-url.txt").read_text().strip(),
                                 input_sha256=digest(mat), binary_sha256=digest(root / "build/bench.exe"),
                                 geometry={str(p):geometry(BANDS+p) for p in (0, PAD)},
                                 cache_miss_prediction=None, time_prediction=None))


def measure(root):
    plan = json.loads((root / "plan.json").read_bytes())
    exe = root / "build/bench.exe"
    assert digest(exe) == plan["binary_sha256"]
    results = {}
    for name, case in plan["cases"].items():
        path = root / f"{name}.u16"
        assert digest(path) == case["sha256"]

        def run(pad, prefix=()):
            output = subprocess.check_output([*map(str, prefix), str(exe), str(path), str(pad)], text=True)
            result, = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
            assert result["ms"] > 0
            assert all(result[k] == v for k, v in case["expected"].items())
            return result

        records = []
        for batch in range(2):
            for pair in range(plan["pairs_per_batch"]):
                order = (0, PAD) if (batch + pair) % 2 == 0 else (PAD, 0)
                records.append(dict(batch=batch, order=order, arms={str(p): run(p) for p in order}))
                print(f"PROGRESS 配对计时 {len(records)}/{2 * plan['pairs_per_batch']}", flush=True)
        save(root / f"{name}_timing.json", records)
        times = [median(r["arms"][str(p)]["ms"] for r in records) for p in (0, PAD)]
        print("PROGRESS Nsight Compute 正在采集缓存计数", flush=True)
        profiled = []
        for trial in (1, 2):
            arms = {}
            for pad in (0, PAD):
                log = root / f"{name}_{pad}_{trial}.csv"
                run(pad, [NCU, "--csv", "--page", "raw", "--log-file", log, "--cache-control", "none",
                          "--clock-control", "none", "--kernel-name-base", "function", "--kernel-name",
                          "spectral_pair", "--launch-count", 1, "--metrics", ",".join(METRICS.values())])
                lines = log.read_text(encoding="utf-8-sig").splitlines()
                header = next(i for i, text in enumerate(lines) if text.startswith('"ID"'))
                row, = [r for r in csv.DictReader(lines[header:]) if r["ID"].isdigit()]
                assert row["profiler__replayer_passes"] == "1"
                arms[str(pad)] = {k: int(row[v].replace(",", "")) for k, v in METRICS.items()}
            profiled.append(arms)
        results[name] = dict(times_ms=times, speedup=times[0] / times[1], profiles=profiled)
        print(name, results[name], flush=True)
    save(root / "summary.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "measure"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--case", choices=("top_left", "center"), help="Freeze only the selected region")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root.resolve(), args.case)
    elif args.case:
        parser.error("--case only applies to freeze")
    else:
        measure(args.root.resolve())
