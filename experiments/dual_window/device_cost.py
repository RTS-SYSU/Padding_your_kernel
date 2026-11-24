"""Estimate double-window timing from the fixed 48-word v1 device probe."""

import hashlib
import json
import math
from pathlib import Path
from statistics import median

from prediction import BLOCK_BYTES, ROUNDS, ROW_MULTIPLE, family_geometry


PROBE_PITCH = 48
PROBE_WINDOW = 24
PROBE_SAMPLES = 5


def load_profile(path, device_plan):
    """Fit the two archived probe rates without reading target plans or timings.

    The dedicated calibration log has no GPU identity field. Its caller supplies
    the device descriptor and may pin the log via calibration_log_sha256.
    """
    capacity = device_plan["l2_bytes"]
    blocks = device_plan["blocks"]
    threads = device_plan["threads"]
    name = device_plan.get("name") or device_plan.get("device")
    if (not isinstance(name, str) or not name or threads != 256 or
            any(type(n) is not int or n <= 0 for n in (capacity, blocks, threads))):
        raise ValueError("Expected a double-window device descriptor with its launch grid")

    # Alternate probe rows cover one and two distinct 128-byte blocks.
    bytes_per_row = 3 * BLOCK_BYTES // 2
    unit = ROW_MULTIPLE * bytes_per_row
    design = {"resident": capacity // (2 * unit) * ROW_MULTIPLE,
              "streaming": (2 * capacity) // unit * ROW_MULTIPLE}
    if not design["resident"] or design["streaming"] * bytes_per_row <= capacity:
        raise ValueError("Capacity is too small for the fixed v1 probe")

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if device_plan.get("calibration_log_sha256", digest) != digest:
        raise ValueError("Calibration log does not match the supplied digest")
    samples = {regime: {} for regime in design}
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        regime, sample = record["regime"], record["sample"]
        if (regime not in design or type(sample) is not int or
                not 0 <= sample < PROBE_SAMPLES or sample in samples[regime]):
            raise ValueError(f"Invalid or duplicate probe sample {regime}/{sample}")
        rows = design[regime]
        expected = {"kind": "calibration", "rows": rows, "rounds": ROUNDS,
                    "pitch_words": PROBE_PITCH, "source_sectors": 2 * rows * ROUNDS,
                    "coverage_bytes": rows * bytes_per_row, "checked_outputs": blocks * threads}
        if (any(record.get(k) != v for k, v in expected.items()) or
                record["verified"] is not True or
                record.get("second_window_words", PROBE_WINDOW) != PROBE_WINDOW):
            raise ValueError(f"Probe {regime}/{sample} disagrees with the fixed design")
        ms = record["kernel_ms"]
        if type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"Invalid duration for probe {regime}/{sample}")
        samples[regime][sample] = ms

    rates = {}
    for regime, rows in design.items():
        if len(samples[regime]) != PROBE_SAMPLES:
            raise ValueError(f"Expected {PROBE_SAMPLES} verified {regime} samples")
        sectors = 2 * rows * ROUNDS
        center = median(samples[regime].values())
        rates[regime] = {"rows": rows, "samples": PROBE_SAMPLES, "issued_read_sectors": sectors,
                         "median_ms": center, "ms_per_sector": center / sectors}
    return {"device": name, "uuid": device_plan.get("uuid"), "l2_bytes": capacity,
            "blocks": blocks, "threads": threads, "calibration_log_sha256": digest,
            "probe_pitch_words": PROBE_PITCH, "probe_second_window_words": PROBE_WINDOW,
            "rounds": ROUNDS, "rates": rates}


def estimate(geometry, profile):
    """Apply the frozen v1 footprint threshold to a family_geometry or CUDA plan."""
    capacity = profile["l2_bytes"]
    case = geometry["case"]
    layout = family_geometry(case, capacity)
    if any(geometry.get(key) != value for key, value in layout.items()):
        raise ValueError(f"Target geometry disagrees with the device and case {case!r}")
    expected = {"device": profile["device"], "blocks": profile["blocks"],
                "threads": profile["threads"], "l2_bytes": capacity, "rounds": ROUNDS,
                "assumed_tag_bytes": BLOCK_BYTES, "warmup_rounds": 1, "grid_barrier": False}
    if any(geometry[key] != value for key, value in expected.items() if key in geometry):
        raise ValueError("Target launch or device disagrees with the probe profile")
    if geometry.get("uuid") and profile.get("uuid") and geometry["uuid"] != profile["uuid"]:
        raise ValueError("Target and calibration have different GPU UUIDs")
    sectors = layout["source_sectors_per_sweep"] * ROUNDS
    point = {"issued_read_sectors_per_arm": sectors}
    for arm in ("base", "pad"):
        footprint = layout[f"{arm}_tag_count"] * BLOCK_BYTES
        regime = "resident" if footprint <= capacity else "streaming"
        cost = profile["rates"][regime]["ms_per_sector"]
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0:
            raise ValueError(f"Invalid calibrated sector cost for {regime}")
        point[f"{arm}_footprint_bytes"] = footprint
        point[f"{arm}_regime"] = regime
        point[f"{arm}_ms"] = sectors * cost
    point["kernel_speedup_point"] = point["base_ms"] / point["pad_ms"]
    return point
