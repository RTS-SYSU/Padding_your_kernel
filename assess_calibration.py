"""Cross-check independent .cg revisit timings against held-out allocations."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from run import ISSUED, L2_READ, L2_MISS, METRICS, PASSES


def predicted_slow(row, info, index, ways=16):
    """Sector-on-demand LRU for A, competitors, witness, then the probed sector.

    Use the logged virtual block numbers as a conditional address model.
    The capacity control uses an unrelated allocation and is scored separately.
    """
    if row["pattern"] == "capacity":
        return None
    if row["sector_word"]:
        return True
    sets = info["l2_bytes"] // (ways * info["block_bytes"])

    def group(line):
        return (line if index == "modulo" else line ^ (line // sets)) % sets

    base = row["allocation_base_block"]
    anchor = base + row["base_block"]
    others = {anchor + i * row["stride_blocks"] for i in range(1, row["competitors"] + 1)}
    others.add(base + (row["base_block"] ^ info["virtual_witness_block_xor"]))
    return sum(group(line) == group(anchor) for line in others) >= ways


def assess(records):
    info, = (row for row in records if row["kind"] == "info")
    if info.get("mode") is not None or info["block_bytes"] != 128 or info["l2_bytes"] % (16 * 128):
        raise ValueError("Expected the default 128-byte-block probe")
    samples = [row for row in records if row["kind"] == "sample"]
    allocations = sorted({row["allocation"] for row in samples})
    if len(allocations) < 2:
        raise ValueError("At least two allocations are needed to check transfer")
    key_fields = ("pattern", "stride_blocks", "competitors", "sector_word")
    identities = set()
    depths = defaultdict(Counter)
    by_arm = defaultdict(list)
    for row in samples:
        if row["pattern"] not in ("strided", "neighbor", "capacity"):
            raise ValueError("Unexpected probe pattern")
        design = tuple(row[key] for key in key_fields)
        arm = row["allocation"]
        identity = (arm, design, row["trial"])
        if identity in identities:
            raise ValueError(f"Duplicate trial {identity}")
        identities.add(identity)
        depths[arm][design] += 1
        by_arm[arm].append(row)
        if min(row[k] for k in ("cold_cycles", "hot_cycles", "probe_cycles")) <= 0:
            raise ValueError("Invalid device clock interval")
    if any(depths[a] != depths[allocations[0]] for a in allocations):
        raise ValueError("Incomplete or unbalanced allocation grids")
    capacity = info["l2_bytes"] // info["block_bytes"]
    strides = {1, 64, 512} | {capacity // d for d in (128, 64, 32, 16, 8)}
    expected = {("strided", s, n, 0) for s in strides for n in (4, 8, 16, 24, 32, 48, 64)}
    expected.update(("neighbor", 0, 0, k) for k in (0, 8, 16, 24))
    expected.add(("capacity", 0, 0, 0))
    if (set(allocations) != {0, 1, 2} or set(depths[allocations[0]]) != expected
            or any(n != 12 for n in depths[allocations[0]].values())):
        raise ValueError("Expected three allocations with twelve trials per design")
    controls = {}
    grouped = defaultdict(dict)
    match_counts = defaultdict(Counter)
    for arm in allocations:
        train = [r for other in allocations if other != arm for r in by_arm[other]
                 if r["pattern"] == "strided"]
        hot = median(r["hot_cycles"] for r in train)
        cold = median(r["cold_cycles"] for r in train)
        if cold <= hot:
            raise ValueError("Flush and hot controls are inseparable in training")
        threshold = (cold + hot) / 2
        test = [r for r in by_arm[arm] if r["pattern"] == "strided"]
        controls[arm] = {
            "held_out_strided_trials": len(test),
            "other_allocations_midpoint_cycles": threshold,
            "hot_median_cycles": median(r["hot_cycles"] for r in test),
            "flushed_median_cycles": median(r["cold_cycles"] for r in test),
            "hot_below_threshold_fraction": sum(r["hot_cycles"] < threshold for r in test) / len(test),
            "flushed_at_or_above_threshold_fraction": sum(r["cold_cycles"] >= threshold for r in test) / len(test),
        }
        values = defaultdict(list)
        for r in by_arm[arm]:
            values[tuple(r[k] for k in key_fields)].append(r["probe_cycles"])
            slow = r["probe_cycles"] >= threshold
            for index in ("modulo", "xor-fold"):
                expected = predicted_slow(r, info, index)
                if expected is not None:
                    key = f"expected_{'slow' if expected else 'fast'}_observed_{'slow' if slow else 'fast'}"
                    match_counts[(r["pattern"], index)][key] += 1
        for design, ticks in values.items():
            grouped[design][arm] = {
                "trials": len(ticks), "median_probe_cycles": median(ticks),
                "at_or_above_threshold": sum(t >= threshold for t in ticks),
            }
    return {
        "probe": info, "n": len(samples), "controls_by_held_out_allocation": controls,
        "interpretation": "Clock classification relative to independent hot/flush controls. "
                          "This signal can also include translation or scheduling effects.",
        "groups": [{**dict(zip(key_fields, design)), "by_allocation": data}
                   for design, data in sorted(grouped.items())],
        "conditional_rule_checks_16ways": [
            {"pattern": pattern, "index": index, "counts": counts}
            for (pattern, index), counts in sorted(match_counts.items())],
    }


def score_anchor_counters(raw):
    lines = raw.splitlines()
    records = [json.loads(line) for line in lines if line.startswith("{")]
    info, = (r for r in records if r["kind"] == "info")
    cases = [r for r in records if r["kind"] == "counter_case"]
    if (info["mode"] not in ("counter_check", "matched_counter")
            or [r["ordinal"] for r in cases] != list(range(info["planned_samples"]))):
        raise ValueError("Incomplete or misordered single-load controls")
    metrics = defaultdict(dict)
    for r in csv.DictReader(line for line in lines if line.startswith('"')):
        if r["Kernel Name"] != "probe_read_anchor" or r["Metric Name"] not in METRICS:
            raise ValueError("Unexpected kernel or metric")
        launch = metrics[int(r["ID"])]
        if r["Metric Name"] in launch:
            raise ValueError("Duplicate counter for a launch")
        launch[r["Metric Name"]] = int(r["Metric Value"].replace(",", ""))
    if set(metrics) != set(range(len(cases))):
        raise ValueError("Missing profiler launch")
    observed = defaultdict(Counter)
    paired = defaultdict(Counter)
    thresholds = {}
    twin_thresholds = {}
    twin_comparisons = defaultdict(Counter)
    if cases and all("probe_cycles" in r for r in cases):
        for arm in sorted({r["allocation"] for r in cases}):
            train = [r for r in cases if r["allocation"] != arm]
            hot, cold = (median(r["probe_cycles"] for r in train if r["pattern"] == control)
                         for control in ("pressure_empty", "pressure_capacity"))
            if cold <= hot:
                raise ValueError("Indistinguishable clock controls")
            thresholds[arm] = (hot + cold) / 2
            if info["mode"] == "matched_counter":
                hot, cold = (median(r["native_cycles"] for r in train if r["pattern"] == control)
                             for control in ("pressure_empty", "pressure_capacity"))
                if cold <= hot:
                    raise ValueError("Indistinguishable unprofiled twin clock controls")
                twin_thresholds[arm] = (hot + cold) / 2
    for r in cases:
        launch = metrics[r["ordinal"]]
        if set(launch) != set(METRICS) or (launch[ISSUED], launch[L2_READ], launch[PASSES]) != (1, 1, 1):
            raise ValueError("Expected exactly one issued sector, one L2 lookup, one profiler pass")
        miss = launch[L2_MISS]
        if miss not in (0, 1):
            raise ValueError("Invalid miss count for one lookup")
        outcome = "miss" if miss else "hit"
        observed[r["pattern"]][outcome] += 1
        if thresholds:
            latency = "slow" if r["probe_cycles"] >= thresholds[r["allocation"]] else "fast"
            paired[r["pattern"]][f"{latency}_{outcome}"] += 1
        if twin_thresholds:
            twin_latency = "slow" if r["native_cycles"] >= twin_thresholds[r["allocation"]] else "fast"
            twin_comparisons[r["pattern"]][f"twin_{twin_latency}_profiled_{outcome}"] += 1
    result = {"device": info["device"], "verified_single_lookup_launches": len(cases),
              "observed_l2_counts": observed, "cycle_threshold_on_other_allocations": thresholds,
              "paired_clock_and_counter_by_condition": paired}
    if twin_thresholds:
        result.update(twin_threshold_on_other_allocations=twin_thresholds,
                      separate_reconditioned_same_allocation_twin=twin_comparisons)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--single-load-counters", action="store_true",
                        help="Score NCU CSV and probe launch ordinals from --counter-check")
    args = parser.parse_args()
    raw = args.raw.read_text(encoding="utf-8-sig")
    result = (score_anchor_counters(raw) if args.single_load_counters else
              assess([json.loads(line) for line in raw.splitlines() if line.strip()]))
    print(json.dumps(result, indent=2, allow_nan=False))
