"""Derive a dual-window pad, then measure and profile the two frozen layouts."""
import re
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess


CASES = {"original": (56, 24), "holdout": (56, 24), "family-104-16-80": (104, 16),
         "family-136-16-80": (136, 16), "family-104-8-80": (104, 8)}
NCU = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.1\target\windows-desktop-win7-x64\ncu.exe")
METRICS = {
    "issued": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l2_reads": "lts__t_sectors_srcunit_tex_op_read.sum",
    "l2_miss": "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
}


def pad_for(width, offset):
    # The row phases modulo 32 are precisely the multiples of gcd(pitch, 32).
    # The highest phase is 32-gcd(pitch,32). With 8-word alignment, the
    # second window fits in the first block iff gcd(pitch,32) >= offset+8.
    modulus = 1 << (offset + 7).bit_length()
    return (-width) % modulus, modulus


def save(path, obj):
    with path.open("x", encoding="utf-8") as output:
        json.dump(obj, output, indent=2, ensure_ascii=False)
        output.write("\n")


def run(cmd):
    output = subprocess.check_output(list(map(str, cmd)), text=True)
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def args_for(exe, case, pad, mode):
    return [exe, mode, case, pad]


def resolve_case(case):
    if case in CASES:
        return CASES[case]
    match = re.fullmatch(r"family-(\d{1,4})-(8|16|24)-(\d{1,3})", case)
    if match:
        width, offset, percent = map(int, match.groups())
        if 64 <= width <= 4096 and width % 8 == 0 and width % 32 and 1 <= percent <= 100:
            return width, offset
    raise ValueError("双窗口要求行宽为 64–4096 内的 8 的倍数且非 32 的倍数，第二窗口取 8、16 或 24。")


def freeze(exe, root, case=None):
    plans = {}
    selected = {case: resolve_case(case)} if case else CASES
    for case, (width, offset) in selected.items():
        pad, modulus = pad_for(width, offset)
        info, = run(args_for(exe, case, pad, "info"))
        assert (width, offset, pad) == (info["logical_words"], info["second_window_words"], info["pad_words"])
        assert info["pad_tag_count"] == info["rows"] and info["rows"] >= 4
        assert info["pad_allocation_bytes"] - info["base_allocation_bytes"] <= info["extra_budget_bytes"]
        plans[case] = dict(info=info, pitch_modulus_words=modulus,
                           issued=2 * info["rows"] * info["rounds"],
                           unique_input_sectors=2 * info["rows"],
                           padded_tag_span_bytes=128 * info["pad_tag_count"],
                           baseline_tag_span_bytes=128 * info["base_tag_count"],
                           warm_perfect_retention_l2_miss=[0, 0],
                           equal_cost_serial_perfect_retention_speedup=1.0,
                           hardware_l2_miss=None, hardware_speedup=None)
    save(root / "plan.json", dict(cases=plans, executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                                 pairs_per_case=12))


def profile(exe, case, pad, mode, log):
    cmd = [NCU, "--csv", "--page", "raw", "--log-file", log, "--cache-control", "none",
           "--clock-control", "none", "--kernel-name-base", "function", "--kernel-name", "row_tail_target",
           "--launch-count", 1, "--metrics", ",".join(METRICS.values()), *args_for(exe, case, pad, mode)]
    _, timed = run(cmd)
    assert timed["verified"]
    lines = log.read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    row, = [row for row in csv.DictReader(lines[start:]) if row["ID"].isdigit()]
    assert row["profiler__replayer_passes"] == "1" and row["profiler__replayer_passes_type_warmup"] == "0"
    return {key: int(row[name].replace(",", "")) for key, name in METRICS.items()}


def measure(exe, root):
    plan = json.loads((root / "plan.json").read_bytes())
    assert hashlib.sha256(exe.read_bytes()).hexdigest() == plan["executable_sha256"]
    for case, case_plan in plan["cases"].items():
        info = case_plan["info"]
        pad = info["pad_words"]
        pairs = []
        for pair in range(plan["pairs_per_case"]):
            order = ("base", "pad") if pair % 2 == 0 else ("pad", "base")
            arms = {}
            for mode in order:
                returned, arm = run(args_for(exe, case, pad, mode))
                assert returned == info and arm["verified"] and arm["kernel_ms"] > 0
                assert arm["pad_words"] == (pad if mode == "pad" else 0)
                arms[mode] = arm
            pairs.append(dict(order=order, arms=arms))
            print(f"PROGRESS 配对计时 {pair + 1}/{plan['pairs_per_case']}", flush=True)
        base, padded = [median(pair["arms"][mode]["kernel_ms"] for pair in pairs)
                        for mode in ("base", "pad")]
        save(root / f"{case}_timing.json", dict(pairs=pairs, base_ms=base, pad_ms=padded, speedup=base / padded))
        print(case, base, padded, base / padded, flush=True)
    for case, case_plan in plan["cases"].items():
        print("PROGRESS Nsight Compute 正在采集缓存计数", flush=True)
        counts = [{mode: profile(exe, case, case_plan["info"]["pad_words"], mode,
                                 root / f"{case}_{mode}_{trial}.csv") for mode in ("base", "pad")}
                  for trial in (1, 2)]
        save(root / f"{case}_counters.json", counts)
        print(case, counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure"))
    parser.add_argument("exe", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--case", help="Freeze one case, including family-W-G-F")
    parsed = parser.parse_args()
    exe, root = parsed.exe.resolve(), parsed.results.resolve()
    if parsed.phase == "freeze":
        freeze(exe, root, parsed.case)
    elif parsed.case:
        parser.error("--case only applies to freeze")
    else:
        measure(exe, root)
