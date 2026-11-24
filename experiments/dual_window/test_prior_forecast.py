"""CPU regression checks for unseen row pitches and pre-timing freeze."""

import contextlib
import copy
import io
import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prior_forecast
import run
import width_transfer
from prediction import family_geometry


def synthetic_profile():
    device = {"name": "fixture", "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
              "l2_bytes": 8 * 1024**2, "blocks": 32, "threads": 256}
    # A constant unit cost makes the expected cross-width speedup exactly one.
    features = len(width_transfer.FEATURES)
    pitches = [168, 176, 184, 192, 200, 208, 216, 224]
    transfer = {"model": width_transfer.MODEL, "device": dict(device), "calibration_pitches": pitches,
                "coefficients": [math.log(1e-7)] + [0.] * features,
                "feature_lower": [0.] * features, "feature_upper": [1.] * features}
    fixed = dict(device, device=device["name"],
                 rates={"resident": {"ms_per_sector": 1e-7}, "streaming": {"ms_per_sector": 4e-7}})
    return {"format": 1, "hardware": device, "fixed_probe": fixed, "width_transfer": transfer,
            "source_sha256": prior_forecast.source_hashes()}


class PriorForecast(unittest.TestCase):
    def setUp(self):
        self.profile = synthetic_profile()

    def test_both_physical_pitches_must_be_new(self):
        ref = self.profile["width_transfer"]
        seen = width_transfer.estimate(ref, "family-168-16-80")
        self.assertEqual(seen["shared_pitches"], [168, 192])
        self.assertIsNone(seen["kernel_speedup_point"])
        # Isolate a shared padded pitch even if the original width was never profiled.
        ref["calibration_pitches"].remove(176)
        shared = width_transfer.estimate(ref, "family-176-16-80")
        self.assertEqual(shared["shared_pitches"], [192])
        self.assertEqual(shared["status"], "target_pitch_seen_in_calibration")
        self.assertIsNone(shared["kernel_speedup_point"])
        for width in (232, 264):
            with self.subTest(width=width):
                case = f"family-{width}-16-80"
                row = prior_forecast.forecast_case(self.profile, case)
                self.assertEqual(row["primary_model"], "width_transfer")
                self.assertEqual(row["width_transfer"]["shared_pitches"], [])
                self.assertEqual(row["width_transfer"]["kernel_speedup_point"], 1.)

    def test_unmatched_protocol_or_rows(self):
        ref = self.profile["width_transfer"]
        for case, reason in (("family-232-24-80", "unsupported_window_or_rounds"),
                             ("family-232-16-40", "rows_outside_calibration_range"),
                             ("family-232-16-100", "rows_outside_calibration_range")):
            with self.subTest(case=case):
                row = prior_forecast.forecast_case(self.profile, case)
                self.assertEqual(row["primary_model"], "fixed_probe")
                self.assertEqual(row["width_transfer"]["status"], reason)
                self.assertIsNone(row["width_transfer"]["kernel_speedup_point"])
                self.assertEqual(row["fixed_probe"]["kernel_speedup_point"],
                                 1. if case.endswith("-40") else 4.)
        self.assertEqual(width_transfer.estimate(ref, "family-232-16-80", rounds=24)["status"],
                         "unsupported_window_or_rounds")
        self.assertEqual(width_transfer.estimate(ref, "family-232-16-80", device_uuid="other")["status"],
                         "device_uuid_mismatch")

    def test_freeze_before_measurement_and_reject_tampering(self):
        cases = ("family-264-16-80", "family-232-16-80")
        hw = self.profile["hardware"]
        forecast = {"format": 1, "hardware": hw, "profile": self.profile,
                    "cases": {case: prior_forecast.forecast_case(self.profile, case) for case in cases}}
        wrong_device = copy.deepcopy(forecast)
        wrong_device["profile"]["width_transfer"]["device"]["uuid"] = "other"
        with self.assertRaises(ValueError):
            prior_forecast.verify(wrong_device)
        changed_point = copy.deepcopy(forecast)
        changed_point["cases"][cases[0]]["width_transfer"]["kernel_speedup_point"] = 99.
        with self.assertRaises(ValueError):
            prior_forecast.verify(changed_point)

        def cuda_info(command, folder, name, uuid):
            self.assertEqual((command[1], name, uuid), ("info", "device", hw["uuid"]))
            info = family_geometry(command[2], hw["l2_bytes"])
            info.update(kind="plan", device=hw["name"], l2_bytes=hw["l2_bytes"], blocks=hw["blocks"],
                        threads=hw["threads"], rounds=40, assumed_tag_bytes=128,
                        grid_barrier=False, warmup_rounds=1)
            return json.dumps(info)

        with TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            path = folder / "input.json"
            path.write_text(json.dumps(forecast), encoding="utf-8")
            exe = folder / "fixture.exe"
            exe.write_bytes(b"never executed")
            with patch.object(run, "capture", side_effect=cuda_info) as info_only:
                run.prepare(exe, folder, forecast_path=path)
            self.assertEqual([c.args[0][2] for c in info_only.call_args_list], list(cases))
            plan = json.loads((folder / "plan.json").read_bytes())
            self.assertEqual(tuple(plan["cases"]), cases)
            self.assertEqual(plan["cases"][cases[0]]["prediction"]["kernel_speedup_point"], 1.)
            self.assertFalse((folder / "evaluation_started.json").exists())
            frozen = folder / "offline_forecast.json"
            self.assertEqual(frozen.read_bytes(), path.read_bytes())
            frozen.write_bytes(frozen.read_bytes() + b" ")
            with patch.object(run, "measure") as measure, self.assertRaisesRegex(ValueError, "changed"):
                run.evaluate(exe, folder, False)
            measure.assert_not_called()
            self.assertFalse((folder / "evaluation_started.json").exists())


if __name__ == "__main__":
    unittest.main()
