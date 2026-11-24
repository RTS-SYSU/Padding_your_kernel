"""Check two preselected 16-way LRU set functions against independent anchor revisits."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

import numpy as np


WAYS = 16
PATTERNS = ("pressure_strided", "pressure_jitter")


def pressure_offsets(count, stride, seed, jitter):
    """Distinct positive block offsets in the bins of calibrate_l2.cu::pressure_offset."""
    if count < 1 or stride < 1 or stride & (stride - 1) or count * stride >= 2**32:
        raise ValueError("invalid pressure geometry")
    bins = np.arange(count, dtype=np.uint32)
    offsets = (bins + 1) * stride
    if jitter and count > 2:
        interior = bins[1:-1]
        value = interior ^ np.uint32(seed)
        value ^= value >> 16
        value *= np.uint32(0x7feb352d)
        value ^= value >> 15
        value *= np.uint32(0x846ca68b)
        value ^= value >> 16
        offsets[1:-1] = interior * stride + 1 + (value & (stride - 1))
    return offsets.astype(np.int64)


def competing_lines(offsets, anchor, witness, sets):
    """Count distinct, intervening same-set lines under each candidate.

    Anchor was just read, so the initial contents and order of the parallel
    intervening accesses are immaterial for these true LRU candidates.
    """
    lines = offsets + anchor
    witness_offset = witness - anchor
    position = np.searchsorted(offsets, witness_offset)
    add_witness = position == len(offsets) or offsets[position] != witness_offset
    counts = {}
    for name, indices, target, neighbor in (
            ("modulo", lines, anchor, witness),
            ("xor-fold", lines ^ (lines // sets), anchor ^ (anchor // sets),
             witness ^ (witness // sets))):
        group = target % sets
        counts[name] = int(np.count_nonzero(indices % sets == group)) + int(
            add_witness and neighbor % sets == group)
    return counts


def confusions(counter):
    result = {}
    for label, predicted in (("retained", False), ("evicted", True)):
        fast = counter[predicted, False]
        slow = counter[predicted, True]
        total = fast + slow
        result[label] = {"total": total, "slow": slow,
                         "slow_fraction": slow / total if total else None}
    return result


def score(records):
    info, = (r for r in records if r["kind"] == "info")
    rows = [r for r in records if r["kind"] == "sample"]
    if info.get("mode") != "pressure" or info["block_bytes"] != 128:
        raise ValueError("expected pressure records with 128-byte address blocks")
    if info["l2_bytes"] <= 0 or info["l2_bytes"] % (128 * WAYS):
        raise ValueError("the device capacity does not divide into 16-way sets")
    if len(rows) != info["planned_samples"]:
        raise ValueError("pressure file is incomplete")
    by_arm = defaultdict(list)
    identities = defaultdict(set)
    for row in rows:
        arm = row["allocation"]
        identity = (row["trial"], row["pattern"], row["stride_blocks"], row["competitors"])
        if identity in identities[arm] or row["sector_word"] != 0:
            raise ValueError("duplicate trial or unsupported sector offset")
        if row["probe_cycles"] <= 0 or row["span_blocks"] != row["stride_blocks"] * row["competitors"]:
            raise ValueError("invalid probe duration or address span")
        if row["pattern"] not in (*PATTERNS, "pressure_empty", "pressure_capacity"):
            raise ValueError("unrecognized pressure pattern")
        identities[arm].add(identity)
        by_arm[arm].append(row)
    arms = sorted(by_arm)
    if len(arms) < 2 or any(identities[a] != identities[arms[0]] for a in arms):
        raise ValueError("unbalanced allocation holdouts")
    controls, thresholds = {}, {}
    for arm in arms:
        train = [r for a in arms if arm != a for r in by_arm[a]]
        hot, cold = (median(r["probe_cycles"] for r in train if r["pattern"] == pattern)
                     for pattern in ("pressure_empty", "pressure_capacity"))
        if cold <= hot:
            raise ValueError("training controls lack a hot/cold ordering")
        thresholds[arm] = (hot + cold) / 2
        control_rows = {}
        for pattern in ("pressure_empty", "pressure_capacity"):
            ticks = [r["probe_cycles"] for r in by_arm[arm] if r["pattern"] == pattern]
            control_rows[pattern] = {"total": len(ticks), "median_cycles": median(ticks),
                                     "slow": sum(t >= thresholds[arm] for t in ticks)}
        controls[arm] = {"threshold_from_other_allocations": thresholds[arm], **control_rows}

    by_design = defaultdict(list)
    for r in rows:
        if r["pattern"] in PATTERNS:
            by_design[r["pattern"], r["competitors"], r["stride_blocks"], r["pressure_seed"]].append(r)
    per_arm = defaultdict(Counter)
    per_geometry = defaultdict(Counter)
    capacity = info["l2_bytes"] // 128
    for (pattern, count, stride, seed), trials in sorted(by_design.items()):
        offsets = pressure_offsets(count, stride, seed, pattern == "pressure_jitter")
        for r in trials:
            anchor = r["allocation_base_block"] + r["base_block"]
            witness = r["allocation_base_block"] + (r["base_block"] ^ info["virtual_witness_block_xor"])
            counts = competing_lines(offsets, anchor, witness, capacity // WAYS)
            arm = r["allocation"]
            slow = r["probe_cycles"] >= thresholds[arm]
            for model, distinct in counts.items():
                outcome = (distinct >= WAYS, slow)
                per_arm[model, arm][outcome] += 1
                per_geometry[model, pattern, count, stride][outcome] += 1

    return {
        "probe": info, "threshold_controls_by_held_out_allocation": controls,
        "candidate_assumptions": "Linearly contiguous CUDA virtual block numbers, 128 B tags, "
            "true 16-way LRU, anchor-competitor and witness loads only; other allocations' "
            "empty/flush midpoint classifies slow rereads. Slow is a timing observation.",
        "index_functions": {"modulo": "line % sets", "xor-fold": "(line ^ (line // sets)) % sets"},
        "by_allocation": {model: {arm: confusions(per_arm[model, arm]) for arm in arms}
                          for model in ("modulo", "xor-fold")},
        "by_geometry": [
            {"model": model, "pattern": pattern, "competitors": count, "stride_blocks": stride,
             "distinct_128b_blocks_over_nominal_capacity": count / capacity,
             "touched_32b_sectors_over_nominal_capacity": count / (4 * capacity),
             "span_blocks": count * stride, **confusions(counts)}
            for (model, pattern, count, stride), counts in sorted(per_geometry.items())],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_jsonl", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.raw_jsonl.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    print(json.dumps(score(rows), indent=2, allow_nan=False))
