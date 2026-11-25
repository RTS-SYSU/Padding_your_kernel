"""Export Experiment 23's native primed timings from the research artifacts."""

import argparse
import csv
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median


PRESET_BY_CASE = {
    ("atax", "default"): "atax",
    ("bicg", "default"): "bicg",
    ("mvt", "n2048"): "mvt",
    ("mvt", "n4096"): "mvt-4096",
    ("gemm", "default"): "gemm",
    ("2mm", "default"): "2mm",
    ("syrk", "default"): "syrk",
    ("syr2k", "default"): "syr2k",
    ("3dconv", "default"): "3dconv",
    ("fdtd", "default"): "fdtd",
    ("hotspot", "default"): "hotspot",
    ("srad", "default"): "srad",
    ("lud", "default"): "lud",
    ("gaussian", "default"): "gaussian",
    ("backprop", "default"): "backprop",
    ("bfs", "default"): "bfs",
    ("kmeans", "default"): "kmeans",
    ("pathfinder", "default"): "pathfinder",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def by_case(rows, artifact):
    cases = {(row["slug"], row["case"]): row for row in rows}
    check(len(cases) == len(rows) and cases.keys() == PRESET_BY_CASE.keys(),
          f"{artifact} must contain each of the 18 Experiment 23 cases exactly once")
    return cases


def positive(value):
    number = float(value)
    check(isfinite(number) and number > 0, f"Invalid timing or speedup {value!r}")
    return number


def measurement(source_dir, preset, prediction, summary, frozen):
    slug, case, pad = (prediction[key] for key in ("slug", "case", "p"))
    folder = source_dir / slug
    if case != "default":
        folder /= case
    result = read_json(folder / "result.json")
    check(result["prediction"] == prediction and result["prediction_id"] == frozen["frozen_utc"],
          f"{preset}: result does not match the frozen prediction")
    check(type(pad) is int and pad >= 0 and summary["operator"] == prediction["operator"]
          and int(summary["p"]) == pad and result["measured_pads"] == ([0, pad] if pad else [0]),
          f"{preset}: unexpected operator or measured padding")

    process_ids = list(range(frozen["processes"]))
    processes = sorted((row for row in csv_rows(folder / "process_summary.csv") if row["mode"] == "primed"),
                       key=lambda row: int(row["process"]))
    check([int(row["process"]) for row in processes] == process_ids
          and all(int(row["p"]) == pad for row in processes), f"{preset}: invalid primed process rows")
    baseline = median(positive(row["p0_us"]) for row in processes)
    padded = median(positive(row["pad_us"]) for row in processes) if pad else None
    ratios = [positive(row["paired_speedup"]) for row in processes] if pad else []
    if not pad:
        check(all(not row["pad_us"] and not row["paired_speedup"] for row in processes),
              f"{preset}: unexpected padded timings for a baseline-only case")
    speedup = median(ratios) if pad else None
    faster = sum(ratio > 1 for ratio in ratios) if pad else None
    status = "baseline_only" if not pad else "accepted" if faster == len(process_ids) else "not_accepted"
    reported = result["preparation"]["primed"]
    check((reported["p0_us"], reported["pad_us"], reported["paired_speedup"],
           reported["faster_processes"], reported["process_ratios"]) == (baseline, padded, speedup, faster, ratios),
          f"{preset}: primed result differs from the process summary")
    check((summary["primed_p0_us"] and float(summary["primed_p0_us"])) == baseline
          and (float(summary["primed_pad_us"]) if summary["primed_pad_us"] else None) == padded
          and (float(summary["primed_paired_speedup"]) if summary["primed_paired_speedup"] else None) == speedup
          and (int(summary["primed_faster_processes"]) if summary["primed_faster_processes"] else None) == faster
          and summary["primed_ratios"].split() == [f"{ratio:.9f}" for ratio in ratios]
          and result["status"] == summary["status"] == status,
          f"{preset}: suite summary differs from the primed result")

    checks = csv_rows(folder / "correctness.csv")
    check(bool(checks) and len(checks) == result["correctness_checks"] == int(summary["correctness_checks"])
          and all(row["passed"] == "1" and int(row["p"]) in result["measured_pads"] for row in checks),
          f"{preset}: incomplete or failed correctness checks")
    devices = result["devices"]
    launches = int(summary["launches"])
    check(sorted(row["process"] for row in devices) == process_ids and launches > 0
          and all(row["launches"] == launches for row in devices), f"{preset}: inconsistent launch metadata")

    return {
        "preset": preset, "operator": prediction["operator"], "case": case, "source": prediction["source"],
        "p": pad, "status": status, "baseline_us": baseline, "padded_us": padded,
        "speedup": speedup, "faster_processes": faster, "process_speedups": ratios,
        "launches": launches, "correctness_checks": len(checks),
    }, {row["device"] for row in devices}


def export(source_dir):
    prediction_path = source_dir / "prediction.json"
    summary_path = source_dir / "summary.csv"
    frozen = read_json(prediction_path)
    samples = frozen["samples_per_round_and_mode"]
    check(all(type(value) is int and value > 0 for value in (frozen["processes"], frozen["rounds"], samples)),
          "Invalid process, round or sample count in prediction.json")
    predictions = by_case(frozen["rows"], prediction_path.name)
    summaries = by_case(csv_rows(summary_path), summary_path.name)
    measurements, devices = [], set()
    for key, preset in PRESET_BY_CASE.items():
        row, case_devices = measurement(source_dir, preset, predictions[key], summaries[key], frozen)
        measurements.append(row)
        devices.update(case_devices)
    check(len(devices) == 1 and all(isinstance(device, str) and device for device in devices),
          "Expected one CUDA device name across all processes and cases")
    return {
        "experiment": "Experiment 23",
        "device": devices.pop(),
        "preparation": "primed",
        "date": datetime.fromisoformat(frozen["frozen_utc"]).date().isoformat(),
        "processes": frozen["processes"], "rounds": frozen["rounds"], "samples_per_round": samples,
        "correctness_checks": sum(row["correctness_checks"] for row in measurements),
        "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in (prediction_path, summary_path)},
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Experiment 23 directory with summary.csv, prediction.json and case results")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file")
    args = parser.parse_args()
    try:
        data = export(args.source_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
