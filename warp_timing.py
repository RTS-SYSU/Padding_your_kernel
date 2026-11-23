"""Diagnose the load-use dependency in the existing dual-window kernel on the CPU.

The paired histogram comes from a cache scenario. A hit-to-miss latency increment
comes from an independent single-thread probe. The resulting unit is cycles
per warp iteration until *both values are ready*, relative to two hits. This
quantity excludes parallel warp scheduling and is not kernel elapsed time.
"""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path


def affected_groups(rows, rounds, sector_misses):
    """Bounds from an aggregate counter, if each requested sector is looked up once."""
    if rows <= 0 or rows % 4 or rounds <= 0 or not 0 <= sector_misses <= 2 * rows * rounds:
        raise ValueError("Expected full warp iterations and at most two missed sectors per row and pass")
    pairs = rows * rounds // 4
    # One warp reads four distinct rows. Each of the two instructions requests
    # one sector from each row. Only the number of occupied groups is bounded.
    return {
        "sector_misses": sector_misses,
        "warp_iterations": pairs,
        "warp_loads_with_miss_min_max": [(sector_misses + 3) // 4, min(2 * pairs, sector_misses)],
        "warp_iterations_with_miss_min_max": [(sector_misses + 7) // 8, min(pairs, sector_misses)],
    }


def dependency(binary, arch):
    sass = subprocess.run(
        ["cuobjdump", "--dump-sass", "--gpu-architecture", arch, "--function", "row_tail_target", str(binary)],
        check=True, capture_output=True, text=True,
    ).stdout
    if sass.count("Function : row_tail_target") != 1:
        raise ValueError("Expected one row_tail_target function for the requested architecture")
    program = re.findall(r"/\*([0-9a-f]+)\*/\s+([^;\n]+);", sass)
    loads = [i for i, (_, code) in enumerate(program) if code.strip().startswith("LDG.")]
    if len(loads) != 2 or loads[1] != loads[0] + 1:
        raise ValueError("Expected two consecutive unconditional global loads")
    outputs = []
    for i in loads:
        match = re.fullmatch(r"LDG[.\w]*\s+(R\d+),\s*\[(R\d+)\.64\]\s*", program[i][1].strip())
        if not match:
            raise ValueError("Unrecognized global load operands")
        dest, base = match.groups()
        if i == loads[1] and (dest in outputs or outputs[0] in (base, f"R{int(base[1:]) + 1}")):
            raise ValueError("The second load overwrites or depends on the first load result")
        outputs.append(dest)
    for use in range(loads[1] + 1, len(program) - 1):
        code = program[use][1].strip()
        if "BRA" in code or code.startswith(("@", "EXIT")):
            break
        registers = set(re.findall(r"\bR\d+\b", code))
        if registers.intersection(outputs):
            sources = set(re.findall(r"\bR\d+\b", code.partition(",")[2]))
            branch = re.fullmatch(r"@!?P\d+\s+BRA\s+0x([\da-f]+)", program[use + 1][1].strip())
            if (not code.startswith("IADD3 ") or not set(outputs).issubset(sources)
                    or not branch or int(branch[1], 16) >= int(program[loads[0]][0], 16)):
                break
            return {
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "arch": arch, "load_pcs": [program[i][0] for i in loads],
                "common_use_pc": program[use][0], "loop_back_to_pc": branch[1],
            }
    raise ValueError("The two loaded values do not first feed a common sum before the loop back edge")


def control_latency(path):
    hot, cold, differences = [], [], []
    info = None
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("kind") == "info":
                if info is not None:
                    raise ValueError("Expected controls from exactly one run")
                info = sample
            # Compare the same sector after scrub and after a repeat, with no
            # inserted competitors. Other probe patterns test cache rules.
            if (sample.get("kind") == "sample" and sample.get("pattern") == "neighbor"
                    and sample["sector_word"] == 0 and sample["competitors"] == 0):
                h, c = sample["hot_cycles"], sample["cold_cycles"]
                if type(h) is not int or type(c) is not int or min(h, c) <= 0:
                    raise ValueError("Probe control clocks must be positive integer cycles")
                hot.append(h)
                cold.append(c)
                differences.append(c - h)
    if info is None or not differences or statistics.median(differences) <= 0:
        raise ValueError("No positive median post-scrub minus immediate-repeat latency in probe controls")
    return {
        "probe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "samples": len(hot),
        "device": info["device"], "l2_bytes": info["l2_bytes"],
        "pattern": "neighbor / sector_word=0 / competitors=0",
        "hot_median_cycles": statistics.median(hot), "post_scrub_median_cycles": statistics.median(cold),
        "paired_delta_median_cycles": statistics.median(differences),
        "positive_delta_fraction": sum(d > 0 for d in differences) / len(differences),
    }


def summarize(paired, delta):
    if (not isinstance(paired, list) or len(paired) != 5
            or any(not isinstance(row, list) or len(row) != 5 for row in paired)
            or any(type(value) is not int or value < 0 for row in paired for value in row)):
        raise ValueError("Expected a 5 x 5 paired_missing_sector_counts histogram of nonnegative integers")
    total = sum(map(sum, paired))
    if not total:
        raise ValueError("Empty warp histogram")
    # If any missed sector adds the same d cycles to its warp instruction,
    # two hits are both ready at second_issue + h. With a miss in the second
    # load this grows by d. With only a first-load miss it grows by
    # max(0, d - (second_issue - first_issue)). No gap is silently assumed.
    second = sum(sum(row[1:]) for row in paired)
    any_miss = total - paired[0][0]
    return {
        "warp_iterations": total, "iterations_with_any_miss": any_miss,
        "iterations_second_load_misses": second,
        "mean_extra_cycles_to_both_values_ready_conditional_range": [delta * second / total, delta * any_miss / total],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    counts = commands.add_parser("counts", help="Group-count bounds given an aggregate lookup miss count")
    for name in ("rows", "rounds", "sector-misses"):
        counts.add_argument("--" + name, type=int, required=True)
    paired = commands.add_parser("paired", help="Use predicted joint groups and independent probe controls")
    paired.add_argument("--estimate", type=Path, required=True, help="Cache-scenario JSON")
    paired.add_argument("--controls", type=Path, required=True, help="Independent calibration probe JSONL")
    paired.add_argument("--binary", type=Path, required=True, help="Exact target cubin or executable")
    paired.add_argument("--arch", required=True, help="cuobjdump architecture, e.g. sm_89")
    args = parser.parse_args()
    if args.command == "counts":
        result = affected_groups(args.rows, args.rounds, args.sector_misses)
    else:
        controls = control_latency(args.controls)
        scenario = json.loads(args.estimate.read_text(encoding="utf-8-sig"))
        if scenario["l2_bytes"] != controls["l2_bytes"]:
            raise ValueError("Independent probe and cache scenario use different L2 capacities")
        groups = {}
        for key in ("base", "padded"):
            arm = scenario[key]
            histogram = arm["paired_missing_sector_counts"]
            group = summarize(histogram, controls["paired_delta_median_cycles"])
            missed = sum((i + j) * n for i, row in enumerate(histogram) for j, n in enumerate(row))
            if arm["read_sectors"] != 8 * group["warp_iterations"] or arm["sector_misses"] != missed:
                raise ValueError("Sector totals disagree with the paired warp histogram")
            groups[key] = group
        result = {
            "target_dependency": dependency(args.binary, args.arch), "single_thread_controls": controls,
            "cache_scenario_sha256": hashlib.sha256(args.estimate.read_bytes()).hexdigest(),
            "conditional_scenarios": groups,
            "meaning": "One common positive load penalty per missed instruction; unknown nonnegative issue gap. "
                       "These conditional local readiness ranges exclude other warps, SM scheduling, and kernel time.",
            "kernel_speedup_point": None,
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
