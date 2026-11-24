"""Export paired native CUDA timings and the raw evidence used to audit them."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


# Frozen allowlist. Each entry is a separate ordinary base/pad evaluation.
# The same archives also contain calibration, replay, repeatability and mirror files.
ORACLE = "path_oracle_v1_20261120T0617201026388Z"
ORACLE_RUN = "row_window_family_20261120T061720435357Z"
V100_SERVICE = "v100_row_tail_20261120T123210Z"
V100_MATCHED = "v100_matched_20261120T124125Z"
RTX_MATCHED = "matched_scan_4060ti_20261120124135Z"
RUNS = {
    "archives": (
        "row_tail_l2_20261119T030623219423Z",
        "row_tail_l2_20261119T032717546870Z",
        "row_window_family_20261119T091359510588Z",
        "row_window_family_20261119T095931560717Z",
    ),
    "results": (
        "row_tail_v1_pair_20261120T0603232351399Z/row_window_family_20261120T060326748635Z",
        f"{ORACLE}/{ORACLE_RUN}",
        f"{ORACLE}/service_study/row_window_family_20261120T062640524105Z",
        f"{ORACLE}/reference_reuse/row_window_family_20261120T063311081033Z",
        f"{V100_SERVICE}/results/row_window_family_20261120T123439772780Z",
        f"{V100_SERVICE}/reuse/row_window_family_20261120T123457951177Z",
        f"{V100_MATCHED}/results/row_window_family_20261120T124235806267Z",
        f"{V100_MATCHED}/client_check/row_window_family_20261120T125014441302Z",
        f"{RTX_MATCHED}/row_window_family_20261120T124139832141Z",
        f"{RTX_MATCHED}/reuse/row_window_family_20261120T124735912673Z",
    ),
}
MISS = "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
ISSUED = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
PASSES = "profiler__replayer_passes"
ARMS = ("base", "pad")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Evidence:
    def __init__(self, roots, output):
        self.roots = roots
        self.output = output
        self.index = {}

    def capture(self, source_root, file):
        relative = file.relative_to(self.roots[source_root]).as_posix()
        # Keep the portable path short enough for Windows ZIP extraction while
        # retaining the complete original location separately in the index.
        portable = relative
        for number, run in enumerate(RUNS.get(source_root, ()), 1):
            if relative.startswith(run + "/"):
                portable = f"run{number:02}/{relative[len(run) + 1:]}"
                break
        if relative.startswith(ORACLE + "/") and portable == relative:
            portable = "counter_validation/" + relative[len(ORACLE) + 1:]
        path = f"evidence/{source_root}/{portable}"
        data = file.read_bytes()
        if path not in self.index:
            target = self.output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            self.index[path] = {
                "path": path, "source_root": source_root, "source_path": relative,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            }
        return path, data


def geometry(case):
    if case in ("original", "holdout"):
        return 56, 24, 75 if case == "original" else 80
    prefix, width, second, percent = case.split("-")
    require(prefix == "family", f"Unexpected target case {case}")
    return int(width), int(second), int(percent)


def read_native(data, expected_plan, case_result, case):
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    require(len(records) == 3, f"{case}: expected one plan and one measurement per layout")
    plan, base, pad = records
    width, gap, percent = geometry(case)
    # The fixed W56 executable predates the second_window_words field.
    window = plan.get("second_window_words", 24 if case in ("original", "holdout") else None)
    require(plan == expected_plan and plan["kind"] == "plan" and plan["case"] == case
            and (width, gap) == (plan["logical_words"], window), f"{case}: plan mismatch")
    require(plan["pad_words"] == (-width) % 32 and plan["rounds"] == 40
            and plan["assumed_tag_bytes"] == 128 and plan["threads"] == 256
            and plan["warmup_rounds"] == 1 and plan["grid_barrier"] is False,
            f"{case}: incompatible target protocol")
    expected_rows = plan["l2_bytes"] * percent // 100 // 128 // plan["threads"] * plan["threads"]
    require(plan["rows"] == expected_rows and plan["source_sectors_per_sweep"] == 2 * expected_rows,
            f"{case}: workload size mismatch")
    for arm, measurement in zip(ARMS, (base, pad)):
        padding = 0 if arm == "base" else plan["pad_words"]
        require(measurement == case_result[arm] and measurement["kind"] == "measurement"
                and measurement["arm"] == arm and measurement["verified"] is True
                and measurement["rows"] == expected_rows and measurement["rounds"] == plan["rounds"]
                and measurement["pitch_words"] == width + padding and measurement["pad_words"] == padding
                and measurement["checked_outputs"] == plan["blocks"] * plan["threads"]
                and measurement["allocation_bytes"] == plan[f"{arm}_allocation_bytes"]
                and measurement["peak_bytes"] <= measurement["device_free_before"]
                and math.isfinite(measurement["kernel_ms"]) and measurement["kernel_ms"] > 0,
                f"{case}: inconsistent {arm} timing, geometry, memory or output check")
    require(pad["allocation_bytes"] - base["allocation_bytes"] <= plan["extra_budget_bytes"],
            f"{case}: space budget exceeded")
    speedup = base["kernel_ms"] / pad["kernel_ms"]
    require(math.isclose(speedup, case_result["kernel_speedup"], rel_tol=1e-9),
            f"{case}: native times do not reproduce recorded speedup")
    return {
        "gpu_name": plan["device"], "rows": expected_rows, "logical_words": width,
        "second_window_words": gap, "target_l2_percent": percent, "pad_words": plan["pad_words"],
        "blocks": plan["blocks"], "threads": plan["threads"], "l2_bytes": plan["l2_bytes"],
        "rounds": plan["rounds"], "base_ms": base["kernel_ms"], "pad_ms": pad["kernel_ms"],
        "base_pitch_words": base["pitch_words"], "pad_pitch_words": pad["pitch_words"],
        "base_verified": base["verified"], "pad_verified": pad["verified"], "speedup": speedup,
    }


def parse_profile(data, plan, arm):
    lines = data.decode("utf-8-sig").splitlines()
    native = [json.loads(line) for line in lines if line.startswith("{")]
    require(len(native) == 2 and native[0] == plan and native[1]["arm"] == arm
            and native[1]["verified"] is True, f"{plan['case']}: unrelated {arm} profile")
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"'))
    counters = {}
    for row in csv.DictReader(lines[start:]):
        metric = row["Metric Name"]
        require(row["Kernel Name"] == "row_tail_target" and row["ID"] == "0" and metric not in counters,
                f"{plan['case']}: ambiguous profile scope")
        counters[metric] = int(row["Metric Value"].replace(",", ""))
    require(counters[PASSES] == 1 and counters[ISSUED] == plan["rows"] * plan["rounds"] * 2,
            f"{plan['case']}: invalid replay count or issued sectors")
    return counters


def physical_device(evidence, namespace, folder, protocol, refs):
    for field in ("scan_reference", "matched_reference"):
        if field in protocol:
            # The reference client checks the current CUDA device UUID before
            # freezing this protocol. Its UUID is independent of the GPU name.
            return protocol[field]["gpu_uuid"]
    monitor_files = {
        "row_window_family_20261120T124235806267Z": ("target_evaluation.activity.jsonl", "card"),
        "row_window_family_20261120T123439772780Z": ("activity_evaluate.jsonl", "state"),
    }
    if folder.name not in monitor_files:
        return None
    session = folder.parent.parent
    log_name, state = monitor_files[folder.name]
    refs["gpu_registration"], data = evidence.capture(namespace, session / "registered.json")
    registered = json.loads(data)["gpu_uuid"]
    refs["gpu_monitor"], data = evidence.capture(namespace, session / "logs" / log_name)
    monitored = {json.loads(line)[state].split(",")[1].strip() for line in data.splitlines() if line.strip()}
    require(monitored == {registered}, f"{folder.name}: monitored GPU mismatch")
    return registered


def read_run(evidence, namespace, folder, observations, seen_native):
    run = folder.name
    paths, docs = {}, {}
    for name in ("protocol", "plan", "evaluation_started", "result"):
        paths[name], data = evidence.capture(namespace, folder / f"{name}.json")
        docs[name] = json.loads(data)
    plan_sha = evidence.index[paths["plan"]]["sha256"]
    plan = docs["plan"]
    require(docs["evaluation_started"]["plan_sha256"] == docs["result"]["plan_sha256"] == plan_sha
            and plan["protocol_sha256"] == evidence.index[paths["protocol"]]["sha256"],
            f"{run}: broken frozen plan/protocol chain")
    results = docs["result"]["cases"]
    require(results.keys() == plan["cases"].keys() == docs["protocol"]["device_plans"].keys(),
            f"{run}: frozen cases differ from evaluated cases")
    gpu_uuid = physical_device(evidence, namespace, folder, docs["protocol"], paths)
    external = None
    if run == ORACLE_RUN:
        paths["counter_result"], data = evidence.capture(namespace, folder.parent / "miss_validation.json")
        external = json.loads(data)
        require(external["v1_plan_sha256"] == plan_sha and external["v1_native_result_sha256"]
                == evidence.index[paths["result"]]["sha256"], f"{run}: unrelated external profile")
    for case, summary in sorted(results.items()):
        refs = paths.copy()
        refs["native"], data = evidence.capture(namespace, folder / case / "native.log")
        digest = evidence.index[refs["native"]]["sha256"]
        require(digest not in seen_native, f"{run}/{case}: duplicated native timing evidence")
        seen_native.add(digest)
        device_plan = plan["cases"][case]["device_plan"]
        require(device_plan == docs["protocol"]["device_plans"][case], f"{case}: device plan changed")
        record = read_native(data, device_plan, summary, case)
        record.update({"observation_id": f"{run}/{case}", "case_id": case, "run_id": run,
                       "label_status": "verified", "gpu_uuid": gpu_uuid,
                       "bench_cu_sha256": docs["protocol"]["sha256"]["bench.cu"],
                       "counter_status": "unavailable", "base_l2_read_miss_sectors": None,
                       "pad_l2_read_miss_sectors": None, "evidence": refs})
        if external or summary["profile"]["status"] == "ok":
            profile = external["cases"][case] if external else summary["profile"]
            directory = folder.parent / "target_miss_evidence" / case if external else folder / case
            for arm in ARMS:
                file = directory / f"profile_{arm}.log"
                ref, raw = evidence.capture(namespace, file)
                if external:
                    key = file.relative_to(folder.parent).as_posix().replace("/", "\\")
                    require(external["raw_sha256"][key] == evidence.index[ref]["sha256"],
                            f"{case}: profile hash differs from archived validation")
                counters = parse_profile(raw, device_plan, arm)
                require(counters == profile["counters"][arm], f"{case}: raw profile mismatch")
                refs[f"profile_{arm}"] = ref
                record[f"{arm}_l2_read_miss_sectors"] = counters[MISS]
            record["counter_status"] = "one_pass_ncu"
        observations.append(record)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_observations(archives_root: Path, results_root: Path, output_dir: Path):
    """Write measured.jsonl, reported.jsonl, evidence_index.jsonl and evidence/."""
    roots = {"archives": Path(archives_root), "results": Path(results_root),
             "declaration": Path(__file__).resolve().parent}
    output = Path(output_dir)
    evidence = Evidence(roots, output)
    measured, reported, seen_native = [], [], set()
    for namespace, runs in RUNS.items():
        for run in runs:
            read_run(evidence, namespace, roots[namespace] / run, measured, seen_native)
    ref, data = evidence.capture("declaration", roots["declaration"] / "observations_reported.json")
    for declared in json.loads(data)["records"]:
        case = declared["case_id"]
        width, gap, percent = geometry(case)
        # Only the reported input, ratio and counts are observations. Geometry is
        # the named workload's definition. The original timings and pass count are unknown.
        reported.append({
            **declared, "observation_id": f"user-report-3060/{case}", "run_id": "user-report-3060",
            "label_status": "reported", "gpu_name": "NVIDIA GeForce RTX 3060", "gpu_uuid": None,
            "logical_words": width, "second_window_words": gap, "target_l2_percent": percent,
            "pad_words": 8, "base_pitch_words": width, "pad_pitch_words": width + 8,
            "blocks": None, "threads": None, "l2_bytes": None, "rounds": None,
            "base_ms": None, "pad_ms": None, "base_verified": None, "pad_verified": None,
            "base_l2_read_miss_sectors": None, "pad_l2_read_miss_sectors": None,
            "bench_cu_sha256": None, "counter_status": "reported", "evidence": {"declaration": ref},
        })
    write_jsonl(output / "measured.jsonl", sorted(measured, key=lambda row: row["observation_id"]))
    write_jsonl(output / "reported.jsonl", reported)
    write_jsonl(output / "evidence_index.jsonl", [evidence.index[k] for k in sorted(evidence.index)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_observations(args.archives_root, args.results_root, args.output_dir)
