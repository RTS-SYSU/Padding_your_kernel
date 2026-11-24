"""Fit a device profile once, then freeze a dual-window forecast entirely on the CPU."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import device_cost
import width_transfer
from prediction import family_geometry


HERE = Path(__file__).resolve().parent
SOURCES = ("prior_forecast.py", "device_cost.py", "width_transfer.py", "prediction.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def publish(path, obj):
    # Write a new file only; a prior prediction must survive re-runs unchanged.
    payload = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as out:
        out.write(payload)
    print(f"{path}  SHA256 {digest(path)}")


def source_hashes():
    return {name: digest(HERE / name) for name in SOURCES}


def fit(reference, log):
    manifest = read(reference / "plan.json")
    fixed_manifest = log.with_name("plan.json")
    if digest(fixed_manifest) != manifest["v1_plan_sha256"]:
        raise ValueError("The two calibrations are not linked by the same device run")
    device = {key: manifest["device"][key] for key in ("name", "uuid", "l2_bytes", "blocks", "threads")}
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", device["uuid"]):
        raise ValueError("The reference device has no full GPU UUID")
    transfer = width_transfer.fit_profile(reference)
    if any(transfer["device"].get(k) != v for k, v in device.items()):
        raise ValueError("Reference devices disagree")
    fixed = device_cost.load_profile(log, dict(device, calibration_log_sha256=
                                               read(fixed_manifest)["calibration_log_sha256"]))
    return {"format": 1, "hardware": device, "width_transfer": transfer, "fixed_probe": fixed,
            "source_sha256": source_hashes()}


def forecast_case(profile, case):
    device = profile["hardware"]
    geometry = family_geometry(case, device["l2_bytes"])
    transfer = width_transfer.estimate(profile["width_transfer"], case, device_uuid=device["uuid"])
    fixed = device_cost.estimate(geometry, profile["fixed_probe"])
    method = "width_transfer" if transfer["kernel_speedup_point"] is not None else "fixed_probe"
    return {"geometry": geometry, "primary_model": method, "width_transfer": transfer, "fixed_probe": fixed}


def verify(forecast):
    profile = forecast["profile"]
    if (forecast["format"] != 1 or profile["format"] != 1 or not forecast["cases"]
            or profile["source_sha256"] != source_hashes() or forecast["hardware"] != profile["hardware"]):
        raise ValueError("Forecast version, device or model source changed")
    hardware = profile["hardware"]
    if (not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", hardware["uuid"])
            or any(profile["width_transfer"]["device"].get(k) != v for k, v in hardware.items())
            or any(profile["fixed_probe"].get("device" if k == "name" else k) != v
                   for k, v in hardware.items())):
        raise ValueError("Inconsistent device identity in the frozen profile")
    for case, row in forecast["cases"].items():
        if row != forecast_case(profile, case):
            raise ValueError(f"The frozen prediction differs from the device profile for {case}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    fit_cmd = modes.add_parser("fit", help="Fit and save a fixed device profile from historical calibration")
    fit_cmd.add_argument("--reference", type=Path, required=True)
    fit_cmd.add_argument("--calibration-log", type=Path, required=True)
    fit_cmd.add_argument("--out", type=Path, required=True)
    freeze_cmd = modes.add_parser("freeze", help="Freeze predictions without a GPU or timing records")
    freeze_cmd.add_argument("--profile", type=Path, required=True)
    freeze_cmd.add_argument("--case", action="append", required=True)
    freeze_cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "fit":
        publish(args.out, fit(args.reference, args.calibration_log))
        return
    if len(set(args.case)) != len(args.case):
        parser.error("Repeated target case")
    profile = read(args.profile)
    result = {"format": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "profile_file_sha256": digest(args.profile), "profile": profile, "hardware": profile["hardware"],
              "cases": {case: forecast_case(profile, case) for case in args.case}}
    verify(result)
    publish(args.out, result)


if __name__ == "__main__":
    main()
