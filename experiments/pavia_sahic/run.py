"""Freeze per-warp sector counts, then time and profile the six CUDA stages."""
import argparse
import csv
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import median
import subprocess

METRICS = {
    "issued": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1_miss": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l2_read": "lts__t_sectors_srcunit_tex_op_read.sum",
    "l2_miss": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
}
NCU = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.1\target\windows-desktop-win7-x64\ncu.exe")


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plane_requests(rows, width, stride, element_bytes):
    count = 0
    # 16x16 tiles, two different spatial rows per warp.
    for row in range(rows):
        for x in range(0, width, 16):
            first = (row * stride + x) * element_bytes
            last = (row * stride + min(x + 16, width) - 1) * element_bytes
            count += last // 32 - first // 32 + 1
    return count


def freeze(root):
    plan = read(root / "plan.json")
    height, width, _ = plan["shape"]
    classes, n = plan["classes"], plan["training_samples"]
    spatial = height * width
    vectors = height * width * classes // 4
    labels = ceil(n / 8) + sum(
        (label * n + min(start + 32, n) - 1) // 4 - (label * n + start) // 4 + 1
        for label in range(classes)
        for start in range(0, n, 32)
    )
    complex_reads = plane_requests(width, height, height, 16)
    requests = {
        "q_kernel": {"0": 8 * vectors},
        "fixq_kernel": {"0": labels},
        "nu_aux3_sum_kernel": {"0": 2 * vectors},
        "Vs_kernel": {"0": 7 * vectors},
        "V4_kernel": {"0": complex_reads},
        "V5_kernel": {str(pad): 2 * complex_reads + 2 * plane_requests(width, height, height + pad, 8)
                      for pad in (0, plan["pads"]["V5_kernel"])},
    }
    assert spatial % 4 == 0 and n > 0
    sources = [root / "build/bench.exe", root / "plan.json", root / "training_indices.i32",
               *(root / f"{case}.{ext}" for case in requests for ext in ("in", "ref"))]
    save(root / "registered.json", dict(source_sha256={str(p.relative_to(root)): digest(p) for p in sources},
                                         read_sector_requests=requests, pairs_per_batch=12))


def run(command):
    raw = subprocess.check_output(list(map(str, command)), text=True)
    result, = [json.loads(line) for line in raw.splitlines() if line.startswith("{")]
    if result["ms"] <= 0 or result["cells"] <= 0:
        raise ValueError("Missing timing or validation")
    return result


def profile(args, root, kernel, pad, trial, ncu):
    log = root / f"{kernel}_{pad}_{trial}.csv"
    checked = run([ncu, "--csv", "--page", "raw", "--log-file", log, "--cache-control", "none",
                   "--clock-control", "none", "--kernel-name-base", "function", "--kernel-name", kernel,
                   "--launch-count", 1, "--metrics", ",".join(METRICS.values()), *args])
    lines = log.read_text(encoding="utf-8-sig").splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    row, = [row for row in csv.DictReader(lines[first:]) if row["ID"].isdigit()]
    if row["profiler__replayer_passes"] != "1":
        raise ValueError(f"Expected a single profiler pass for {kernel}")
    return dict(counters={k: int(row[v].replace(",", "")) for k, v in METRICS.items()},
                checked_output=checked)


def measure(root, ncu):
    plan = read(root / "plan.json")
    registration = read(root / "registered.json")
    if any(digest(root / name) != fingerprint for name, fingerprint in registration["source_sha256"].items()):
        raise ValueError("Frozen inputs changed")
    exe = root / "build/bench.exe"
    totals = {}
    for kernel, pad in plan["pads"].items():
        arms = (0, pad) if pad else (0,)
        args = {p: [exe, kernel, p, root, plan["training_samples"]] for p in arms}
        timing_path = root / f"{kernel}_timing.json"
        if timing_path.exists():
            records = read(timing_path)
        else:
            records = []
            for batch in (1, 2):
                for pair in range(registration["pairs_per_batch"]):
                    order = arms if (batch + pair) % 2 == 0 else arms[::-1]
                    records.append(dict(batch=batch, order=order, arms={str(p): run(args[p]) for p in order}))
            save(timing_path, records)
        times = {str(p): median(r["arms"][str(p)]["ms"] for r in records) for p in arms}
        print(kernel, "timing", times, flush=True)
        profiles = []
        for trial in (1, 2):
            profiles.append({str(p): profile(args[p], root, kernel, p, trial, ncu) for p in arms})
        totals[kernel] = dict(pad=pad, median_ms=times, profiles=profiles,
                              speedup=times["0"] / times[str(pad)] if pad else None,
                              batch_speedup=[
                                  median(r["arms"]["0"]["ms"] for r in records if r["batch"] == batch) /
                                  median(r["arms"][str(pad)]["ms"] for r in records if r["batch"] == batch)
                                  for batch in (1, 2)] if pad else [],
                              predicted_issued=registration["read_sector_requests"][kernel])
        print(kernel, [p["0"]["counters"] for p in profiles], flush=True)
    save(root / "summary.json", totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "measure"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ncu", type=Path, default=NCU)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.directory.resolve())
    else:
        measure(args.directory.resolve(), args.ncu)

