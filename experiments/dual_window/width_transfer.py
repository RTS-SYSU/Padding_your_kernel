"""Fixed-device empirical transfer to previously unseen dual-window row pitches.

fit_profile(reference_dir) reads independent paired references for six fixed widths.
estimate(profile, case_id) uses only that JSON-serializable profile and case geometry.
The CLI prints both, so its stdout can be frozen before either target layout is timed.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

if __package__:
    from .prediction import BLOCK_WORDS, ROUNDS, family_geometry, window_block_count
else:
    from prediction import BLOCK_WORDS, ROUNDS, family_geometry, window_block_count


WIDTHS = (168, 176, 184, 200, 208, 216)
PERCENTS = (75, 77, 79, 81, 83, 85)
FEATURES = ("footprint_per_row", "load", "footprint_x_load", "inverse_stride",
            "stride_power_of_two", "row_start_step", "row_start_step_x_load")
RIDGE = 1.0
MODEL = "dualwindow-disjoint-pitch-logridge-v1"


def _pitch(width, arm):
    return width if arm == "base" else width + (-width) % BLOCK_WORDS


def _features(pitch, rows, capacity):
    # The benchmark starts both layouts at a block boundary. Occupancy here is
    # address-block coverage at 128 B granularity, not a measured miss rate.
    blocks = window_block_count(rows, pitch, 16)
    coverage = blocks / rows
    load = 128 * rows / capacity
    stride_sectors = pitch // 8
    trailing_zeros = (stride_sectors & -stride_sectors).bit_length() - 1
    phase_step = (pitch % BLOCK_WORDS) / BLOCK_WORDS
    return (coverage, load, coverage * load, BLOCK_WORDS / pitch, trailing_zeros,
            phase_step, phase_step * load)


def _scale(values, lower, upper):
    # Keep each effect within its calibration range for unsupported pitch sizes.
    return [0.0 if lo == hi else 2 * (min(hi, max(lo, x)) - lo) / (hi - lo) - 1
            for x, lo, hi in zip(values, lower, upper)]


def _solve(matrix, response):
    """Solve the small positive-definite ridge normal equations by elimination."""
    rows = [row[:] + [y] for row, y in zip(matrix, response)]
    size = len(rows)
    for i in range(size):
        pivot = rows[i][i]
        if pivot <= 0 or not math.isfinite(pivot):
            raise ValueError("Invalid ridge system")
        for j in range(i + 1, size):
            factor = rows[j][i] / pivot
            for k in range(i, size + 1):
                rows[j][k] -= factor * rows[i][k]
    result = [0.0] * size
    for i in reversed(range(size)):
        other = sum(rows[i][j] * result[j] for j in range(i + 1, size))
        result[i] = (rows[i][-1] - other) / rows[i][i]
    return result


def fit_profile(reference_dir):
    """Fit the fixed profile from matched-scan reference records; no target labels."""
    directory = Path(reference_dir)
    log = (directory / "matched.jsonl").read_bytes()
    protocol = (directory / "protocol.json").read_bytes()
    # Only these provenance fields are used from the archived reference manifest.
    manifest = json.loads((directory / "plan.json").read_bytes())
    log_hash = hashlib.sha256(log).hexdigest()
    if (log_hash != manifest["matched_log_sha256"] or
            hashlib.sha256(protocol).hexdigest() != manifest["protocol_sha256"]):
        raise ValueError("Reference provenance does not match its frozen manifest")
    samples = {}
    device = None
    selected = hashlib.sha256()
    for line in log.splitlines():
        record = json.loads(line)
        rep, width, percent, rounds = record["job"]
        if width not in WIDTHS or rounds != ROUNDS or percent not in PERCENTS:
            continue
        info, *arms = record["records"]
        if device is None:
            device = info
        if (device != info or info["kind"] != "matched_device" or not info["uuid"]
                or [arm["arm"] for arm in arms] != ["base", "pad"] or rep not in (0, 1, 2)):
            raise ValueError("Invalid paired reference device or job")
        selected.update(line + b"\n")
        rows = info["l2_bytes"] * percent // (100 * 128 * 256) * 256
        for arm in arms:
            key = (width, percent, arm["arm"])
            times = samples.setdefault(key, {})
            expected = {"kind": "matched_scan", "verified": True, "rounds": ROUNDS,
                        "blocks": info["blocks"], "threads": info["threads"],
                        "second_window_words": 16, "rows": rows, "percent": percent,
                        "logical_words": width, "pitch_words": _pitch(width, arm["arm"]),
                        "source_sectors": 2 * rows * ROUNDS}
            ms = arm["kernel_ms"]
            if (rep in times or any(arm.get(k) != v for k, v in expected.items())
                    or not math.isfinite(ms) or ms <= 0):
                raise ValueError(f"Invalid reference for {key}, repetition {rep}")
            times[rep] = ms / arm["source_sectors"]
    if (len(samples) != len(WIDTHS) * len(PERCENTS) * 2
            or any(set(reps) != {0, 1, 2} for reps in samples.values())):
        raise ValueError("Incomplete fixed-width calibration")

    features, responses = [], []
    for (width, percent, arm), times in sorted(samples.items()):
        rows = family_geometry(f"family-{width}-16-{percent}", device["l2_bytes"])["rows"]
        features.append(_features(_pitch(width, arm), rows, device["l2_bytes"]))
        responses.append(math.log(statistics.median(times.values())))
    lower = [min(column) for column in zip(*features)]
    upper = [max(column) for column in zip(*features)]
    design = [[1.0, *_scale(f, lower, upper)] for f in features]
    columns = list(zip(*design))
    matrix = [[sum(x * y for x, y in zip(a, b)) for b in columns] for a in columns]
    for i in range(1, len(matrix)):
        matrix[i][i] += RIDGE
    response = [sum(x * y for x, y in zip(col, responses)) for col in columns]
    return {"model": MODEL, "device": device, "calibration_widths": list(WIDTHS),
            "calibration_pitches": sorted({_pitch(w, a) for w in WIDTHS for a in ("base", "pad")}),
            "calibration_percents": list(PERCENTS), "calibration_arms": len(samples),
            "feature_names": list(FEATURES), "feature_lower": lower, "feature_upper": upper,
            "coefficients": _solve(matrix, response), "ridge": RIDGE,
            "source_log_sha256": log_hash, "selected_rows_sha256": selected.hexdigest()}


def estimate(profile, case_id, *, rounds=ROUNDS, device_uuid=None):
    """Estimate one case; return a reason and null times outside the declared domain."""
    if profile["model"] != MODEL:
        raise ValueError("Unsupported device profile")
    device = profile["device"]
    capacity = device["l2_bytes"]
    geometry = family_geometry(case_id, capacity)
    width, rows = geometry["logical_words"], geometry["rows"]
    pitches = [_pitch(width, a) for a in ("base", "pad")]
    shared = sorted(set(pitches).intersection(profile["calibration_pitches"]))
    result = {"case": case_id, "model": MODEL, "gpu_uuid": device["uuid"], "rows": rows,
              "pad_words": geometry["pad_words"], "shared_pitches": shared,
              "width_extrapolation": not min(WIDTHS) <= width <= max(WIDTHS),
              "base_ms": None, "pad_ms": None, "kernel_speedup_point": None}
    lo, hi = [capacity * f // (100 * 128 * 256) * 256 for f in (min(PERCENTS), max(PERCENTS))]
    if device_uuid is not None and device["uuid"] != device_uuid:
        reason = "device_uuid_mismatch"
    elif geometry["second_window_words"] != 16 or rounds != ROUNDS:
        reason = "unsupported_window_or_rounds"
    elif not lo <= rows <= hi:
        reason = "rows_outside_calibration_range"
    elif shared:
        reason = "target_pitch_seen_in_calibration"
    else:
        coeff = profile["coefficients"]
        for arm, pitch in zip(("base", "pad"), pitches):
            f = _scale(_features(pitch, rows, capacity),
                       profile["feature_lower"], profile["feature_upper"])
            ms_per_sector = math.exp(coeff[0] + sum(c * x for c, x in zip(coeff[1:], f)))
            result[arm + "_ms"] = ms_per_sector * 2 * rows * rounds
        result["kernel_speedup_point"] = result["base_ms"] / result["pad_ms"]
        reason = "empirical_estimate"
    result["status"] = reason
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    args = parser.parse_args()
    profile = fit_profile(args.reference)
    print(json.dumps({"profile": profile, "predictions": [estimate(profile, c) for c in args.case]},
                     ensure_ascii=False, indent=2, allow_nan=False))
