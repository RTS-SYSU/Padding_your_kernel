"""Freeze the row-tail plan, time one pair and optionally collect separate GPU counters."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone

from prediction import predict


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results"
ISSUED = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
L2_READ = "lts__t_sectors_srcunit_tex_op_read.sum"
L2_MISS = "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
PASSES = "profiler__replayer_passes"
METRICS = (ISSUED, L2_READ, L2_MISS, PASSES)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def capture(command, folder, name, gpu_uuid=None):
    args = [str(arg) for arg in command]
    env = None if gpu_uuid is None else {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_uuid}
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    (folder / (name + ".log")).write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; see {folder / (name + '.log')}")
    return result.stdout.decode("utf-8", errors="replace")


def records(raw, kind):
    data = [json.loads(line) for line in raw.splitlines() if line.startswith("{")]
    return [obj for obj in data if obj.get("kind") == kind]


def counters(raw):
    header = None
    values = {}
    for row in csv.reader(io.StringIO(raw)):
        if "Metric Name" in row and "Metric Value" in row:
            header = row
        elif header and len(header) == len(row):
            item = dict(zip(header, row))
            key = item["Metric Name"]
            if key in METRICS:
                if key in values:
                    raise ValueError(f"Multiple target counter values for {key}")
                values[key] = int(item["Metric Value"].replace(",", ""))
    if set(values) != set(METRICS) or values[PASSES] != 1:
        raise ValueError(f"Expected one replay pass with all requested metrics; got {values}")
    return values


CASES = ("holdout", "original")
# Hold the number of logical rows fixed across the six target geometries.
FAMILY_CASES = tuple(f"family-{width}-{offset}-80"
                     for width in (104, 112, 120) for offset in (16, 24))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(exe):
    return {p.name: digest(p) for p in (HERE / "bench.cu", HERE / "run.py", HERE / "prediction.py", exe)}


def read_forecast(raw):
    from prior_forecast import verify

    forecast = json.loads(raw)
    verify(forecast)
    uuid = forecast["hardware"]["uuid"]
    if not isinstance(uuid, str) or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
        raise ValueError("Offline forecast must identify one full physical GPU UUID")
    if not forecast["cases"]:
        raise ValueError("Offline forecast has no target cases")
    return forecast


def prediction_for_run(case, info, forecast):
    point = predict(case, info)
    if forecast is None:
        return point
    hardware = forecast["hardware"]
    if (info["device"] != hardware["name"] or
            any(info[k] != hardware[k] for k in ("l2_bytes", "blocks", "threads")) or
            info.get("uuid", hardware["uuid"]) != hardware["uuid"]):
        raise ValueError("CUDA device or launch grid differs from the offline forecast")
    entry = forecast["cases"][case]
    if any(info.get(key) != value for key, value in entry["geometry"].items()):
        raise ValueError(f"CUDA geometry differs from the offline forecast for {case}")
    model = entry["primary_model"]
    if model not in ("fixed_probe", "width_transfer"):
        raise ValueError(f"Unknown empirical prediction model for {case}")
    value = entry[model]["kernel_speedup_point"]
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid empirical prediction for {case}")
    return {**point, "kernel_speedup_status": "empirical_prerun_estimate",
            "kernel_speedup_point": value, "empirical_model": model}


def prepare(exe, folder, target_cases=None, forecast_path=None):
    forecast = None
    forecast_digest = None
    gpu_uuid = None
    if forecast_path is not None:
        raw = Path(forecast_path).read_bytes()
        forecast = read_forecast(raw)
        if target_cases is not None and set(target_cases) != set(forecast["cases"]):
            raise ValueError("Explicit cases differ from the cases in the offline forecast")
        target_cases = tuple(forecast["cases"])
        gpu_uuid = forecast["hardware"]["uuid"]
        forecast_digest = hashlib.sha256(raw).hexdigest()
        with (folder / "offline_forecast.json").open("xb") as copy:
            copy.write(raw)
    elif target_cases is None:
        target_cases = CASES
    devices = {}
    cases = {}
    for case in target_cases:
        case_dir = folder / case
        case_dir.mkdir()
        info, = records(capture([exe, "info", case], case_dir, "device", gpu_uuid), "plan")
        cases[case] = {"device_plan": info, "prediction": prediction_for_run(case, info, forecast)}
        devices[case] = info
    if len({(v["device"], v["l2_bytes"], v["blocks"]) for v in devices.values()}) != 1:
        raise ValueError("Expected the same device and grid for all cases")
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "sha256": fingerprints(exe),
                "target_cases": list(target_cases), "device_plans": devices}
    if forecast_digest is not None:
        protocol["offline_forecast_sha256"] = forecast_digest
    save(folder / "protocol.json", protocol)
    save(folder / "plan.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                                "protocol_sha256": digest(folder / "protocol.json"), "cases": cases})
    for case, item in cases.items():
        point = item["prediction"]
        speedup = point["kernel_speedup_point"]
        display = "undetermined" if speedup is None else f'{speedup:.6f} ({point["empirical_model"]})'
        print(f'{case}: pad {point["pad_words"]}, 128-byte footprint '
              f'{point["base_footprint_bytes"]} -> {point["pad_footprint_bytes"]} bytes, '
              f'kernel speedup {display}', flush=True)
    print("Frozen plan SHA256:", digest(folder / "plan.json"), flush=True)


def measure(exe, folder, case, item, gpu_uuid=None):
    info = item["device_plan"]
    raw = capture([exe, "both", case], folder, "native", gpu_uuid)
    if records(raw, "plan") != [info]:
        raise ValueError("Native device plan differs from frozen plan")
    base, padded = records(raw, "measurement")
    for arm, name, pad in ((base, "base", 0), (padded, "pad", info["pad_words"])):
        if (arm["arm"] != name or arm["checked_outputs"] != info["blocks"] * info["threads"]
                or arm["allocation_bytes"] != info[name + "_allocation_bytes"]
                or arm["peak_bytes"] > arm["device_free_before"] or arm["pad_words"] != pad
                or arm["pitch_words"] != info["logical_words"] + pad or arm["rows"] != info["rows"]
                or arm["rounds"] != info["rounds"] or arm["verified"] is not True
                or not math.isfinite(arm["kernel_ms"]) or arm["kernel_ms"] <= 0):
            raise ValueError("Invalid native timing or output verification")
    ratio = base["kernel_ms"] / padded["kernel_ms"]
    print(f'{case} measured: base {base["kernel_ms"]:.6f} ms, pad {padded["kernel_ms"]:.6f} ms, '
          f'speedup {ratio:.6f}', flush=True)
    return {"base": base, "pad": padded, "kernel_speedup": ratio, "observed_speedup_above_one": ratio > 1,
            "profile": {"status": "not_requested"}}


def profile(exe, folder, case, info, gpu_uuid=None):
    ncu = shutil.which("ncu")
    if not ncu:
        raise RuntimeError("Nsight Compute (ncu) not found")
    arms = {}
    for mode in ("base", "pad"):
        cmd = [ncu, "--config-file", "off", "--replay-mode", "kernel", "--cache-control", "none",
               "--clock-control", "none", "--kernel-name", "regex:row_tail_target",
               "--launch-count", "1", "--metrics", ",".join(METRICS), "--csv", exe, mode, case]
        arms[mode] = counters(capture(cmd, folder, "profile_" + mode, gpu_uuid))
    a, b = arms["base"], arms["pad"]
    return {"status": "ok", "counters": arms,
            "issued_matches_plan": a[ISSUED] == b[ISSUED] == 2 * info["rows"] * info["rounds"],
            "fewer_l2_read_miss_sectors": b[L2_MISS] < a[L2_MISS]}


def evaluate(exe, folder, with_counters):
    plan = json.loads((folder / "plan.json").read_bytes())
    protocol = json.loads((folder / "protocol.json").read_bytes())
    if digest(folder / "protocol.json") != plan["protocol_sha256"] or fingerprints(exe) != protocol["sha256"]:
        raise ValueError("Frozen inputs have changed")
    target_cases = protocol["target_cases"]
    forecast = None
    gpu_uuid = None
    if "offline_forecast_sha256" in protocol:
        raw = (folder / "offline_forecast.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != protocol["offline_forecast_sha256"]:
            raise ValueError("Frozen offline forecast has changed")
        forecast = read_forecast(raw)
        if list(forecast["cases"]) != target_cases:
            raise ValueError("Offline forecast cases differ from frozen protocol")
        gpu_uuid = forecast["hardware"]["uuid"]
    if (list(plan["cases"]) != target_cases or list(protocol["device_plans"]) != target_cases
            or any(plan["cases"][case] != {"device_plan": info, "prediction": prediction_for_run(case, info, forecast)}
                   for case, info in protocol["device_plans"].items())):
        raise ValueError("Frozen geometry differs from the registered inputs")
    plan_hash = digest(folder / "plan.json")
    with (folder / "evaluation_started.json").open("x", encoding="utf-8") as record:
        json.dump({"utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_hash}, record)
    results = {}
    for case in target_cases:
        results[case] = measure(exe, folder / case, case, plan["cases"][case], gpu_uuid)
        save(folder / case / "result.json", results[case])
    summary = {**results.get("original", {}), "cases": results, "plan_sha256": plan_hash}
    save(folder / "result.json", summary)
    if with_counters:
        for case in target_cases:
            try:
                results[case]["profile"] = profile(exe, folder / case, case, plan["cases"][case]["device_plan"],
                                                   gpu_uuid)
            except (RuntimeError, ValueError, OSError) as exc:
                results[case]["profile"] = {"status": "unverified", "reason": str(exc)}
            save(folder / case / "result.json", results[case])
        if "original" in results:
            summary["profile"] = results["original"]["profile"]
        save(folder / "result.json", summary)
        if any(results[case]["profile"]["status"] != "ok" for case in target_cases):
            raise RuntimeError("Native timing completed; see result.json for incomplete counters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path, help="Compiled bench executable")
    parser.add_argument("--results-root", type=Path, default=ARCHIVE)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--freeze-only", action="store_true", help="Stop after saving predictions")
    phase.add_argument("--evaluate", type=Path, metavar="FROZEN_FOLDER", help="Evaluate a frozen plan once")
    parser.add_argument("--forecast", type=Path, metavar="FILE",
                        help="Use a verified CPU-frozen forecast and its GPU and cases")
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument("--family", action="store_true", help="Use the six registered row-window cases")
    cases.add_argument("--family-case", type=int, nargs=3, metavar=("WIDTH", "OFFSET", "L2_PERCENT"),
                       action="append", help="Register a two-window geometry; repeat for additional cases")
    parser.add_argument("--profile", action="store_true", help="Profile after all native timings")
    args = parser.parse_args()
    if args.freeze_only and args.profile:
        parser.error("Pass --profile at evaluation time")
    if args.evaluate and (args.family or args.family_case or args.forecast):
        parser.error("Evaluation takes its cases and forecast from the frozen protocol")
    target_cases = FAMILY_CASES if args.family else None
    if args.family_case:
        target_cases = tuple(f"family-{w}-{g}-{f}" for w, g, f in args.family_case)
        if len(set(target_cases)) != len(target_cases):
            parser.error("Repeated family case")
    exe = args.exe.resolve(strict=True)
    if args.evaluate:
        folder = args.evaluate.resolve(strict=True)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prefix = "row_window_family_" if args.family or args.family_case or args.forecast else "row_tail_l2_"
        folder = args.results_root.resolve() / (prefix + stamp)
        folder.mkdir(parents=True, exist_ok=False)
    print(f"Results: {folder}", flush=True)
    if not args.evaluate:
        prepare(exe, folder, target_cases, args.forecast)
    if not args.freeze_only:
        evaluate(exe, folder, args.profile)


if __name__ == "__main__":
    main()
