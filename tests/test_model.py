import json
import unittest

from padding_your_kernel.model import adjacent_matrix_reuse_distance, analyze_adjacent_columns
from padding_your_kernel.presets import analyze_preset, list_presets


def line_ids(width, pad, column, base=0):
    return [base + (lane * (width + pad) + column) // 32 for lane in range(32)]


def direct_distance(before, after, lane, sets):
    target = before[lane]
    if target != after[lane]:
        return None
    interval = before[lane + 1:] + after[:lane]
    return len({tag for tag in interval if tag != target and (tag - target) % sets == 0})


class AdjacentColumnsTest(unittest.TestCase):
    def test_formula_matches_signed_address_differences(self):
        for width in (2048, 4096):
            for pad, peak in ((0, 31), (8, 4), (16, 2), (24, 1), (32, 0)):
                peaks_by_set_count = [0, 0]
                for col in range(32):
                    before = line_ids(width, pad, col, base=17)
                    after = line_ids(width, pad, col + 1, base=17)
                    for lane in range(32):
                        if before[lane] != after[lane]:
                            continue
                        phase = (lane * (width + pad) + col) % 32
                        actual = adjacent_matrix_reuse_distance(lane, phase, pad)
                        for i, sets in enumerate((32, 64)):
                            expected = direct_distance(before, after, lane, sets)
                            self.assertEqual(actual, expected)
                            peaks_by_set_count[i] = max(peaks_by_set_count[i], actual)
                self.assertEqual(peaks_by_set_count, [peak, peak])

    def test_smallest_padding_by_reserve(self):
        for reserve, expected in ((0, 16), (1, 16), (2, 24), (3, 32)):
            with self.subTest(reserve=reserve):
                result = analyze_adjacent_columns(4096, 65, reserved_lines=reserve, base_alignment_bytes=32)
                self.assertEqual(result.p, expected)
                self.assertEqual(result.status, "recommend_padding")
                self.assertEqual(result.layouts[0].logical_width, 4096)
                self.assertEqual(result.layouts[0].physical_pitch, 4096 + expected)
                self.assertEqual(result.extra_bytes, 65 * 4 * expected)
                self.assertEqual(json.loads(json.dumps(result.to_dict()))["p"], expected)

    def test_unsupported_is_distinct_from_compact(self):
        wrong_width = analyze_adjacent_columns(128, 128)
        only_partial_warp = analyze_adjacent_columns(2048, 16)
        poor_alignment = analyze_adjacent_columns(2048, 32, base_alignment_bytes=16)
        full_reserve = analyze_adjacent_columns(2048, 32, reserved_lines=4)
        for result in (wrong_width, only_partial_warp, poor_alignment, full_reserve):
            self.assertEqual(result.status, "unsupported")
            self.assertIsNone(result.p)
            self.assertIsNone(result.extra_bytes)
            self.assertEqual(result.layouts, ())
            self.assertTrue(result.reason)

    def test_invalid_values_raise(self):
        with self.assertRaises(ValueError):
            analyze_adjacent_columns(True, 32)
        with self.assertRaises(ValueError):
            analyze_adjacent_columns(2048, 0)
        with self.assertRaises(ValueError):
            analyze_adjacent_columns(2048, 32, reserved_lines=-1)
        with self.assertRaises(ValueError):
            adjacent_matrix_reuse_distance(0, 31, 16)
        with self.assertRaises(ValueError):
            adjacent_matrix_reuse_distance(32, 0, 16)


class FixedPresetTest(unittest.TestCase):
    def test_expected_preset_decisions(self):
        expected = {
            "atax": 16, "bicg": 16, "mvt": 16, "mvt-4096": 16,
            "gemm": 0, "2mm": 0, "syrk": 8, "syr2k": 32,
            "3dconv": 0, "fdtd": 0, "hotspot": 0, "srad": 16,
            "lud": 0, "gaussian": 8, "backprop": 0, "bfs": 0,
            "kmeans": 0, "pathfinder": 0,
        }
        presets = list_presets()
        self.assertEqual(len({preset.operator for preset in presets}), 17)
        self.assertEqual(expected.keys(), {preset.name for preset in presets})
        for preset in presets:
            with self.subTest(preset=preset.name):
                result = analyze_preset(preset.name)
                self.assertEqual(result.p, expected[preset.name])
                if preset.name == "bfs":
                    self.assertEqual(result.status, "unsupported")
                    self.assertEqual(result.layouts, ())
                elif expected[preset.name]:
                    self.assertEqual(result.status, "recommend_padding")
                else:
                    self.assertEqual(result.status, "keep_baseline")
                for layout in result.layouts:
                    self.assertEqual(layout.physical_pitch, layout.logical_width + result.p)
                self.assertEqual(result.extra_bytes, sum(layout.extra_bytes for layout in result.layouts))
                self.assertTrue(result.reason)
                self.assertTrue(result.source)
                self.assertTrue(preset.to_dict()["summary"])

    def test_physical_costs_for_positive_layouts(self):
        for name, expected in (("mvt", 131072), ("mvt-4096", 262144), ("syrk", 12288),
                               ("syr2k", 98304), ("srad", 196608), ("gaussian", 8192)):
            self.assertEqual(analyze_preset(name).extra_bytes, expected)
        two_mm = analyze_preset("2mm")
        self.assertEqual([(item.rows, item.logical_width) for item in two_mm.layouts], [(256, 512), (512, 384)])
        self.assertEqual(analyze_preset("3dconv").layouts[0].rows, 64 * 128)

    def test_width_256_signed_witness(self):
        # As in the SYRK profile, certify own-line distance at the selected
        # width and base alignment, independently of the generic width rule.
        result = analyze_preset("syrk")
        for sets in (32, 64):
            peak = 0
            for col in range(32):
                before = line_ids(256, result.p, col)
                after = line_ids(256, result.p, col + 1)
                for lane in range(32):
                    value = direct_distance(before, after, lane, sets)
                    if value is not None:
                        peak = max(peak, value)
            self.assertEqual(peak, 1)

    def test_unknown_preset(self):
        self.assertEqual(analyze_preset(" SYRK ").p, 8)
        with self.assertRaises(KeyError):
            analyze_preset("unknown")


if __name__ == "__main__":
    unittest.main()
