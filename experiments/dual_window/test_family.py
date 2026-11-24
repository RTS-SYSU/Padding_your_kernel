"""CPU checks for the two-window GPU experiment's geometry and frozen prediction."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction import family_geometry, predict, window_block_count


class FamilyExperiment(unittest.TestCase):
    def test_blocks_match_individual_word_addresses(self):
        for rows in (1, 3, 4, 7, 8, 9):
            for pitch in (56, 64, 72, 80, 88, 96, 4072, 4096):
                for gap in (8, 16, 24):
                    with self.subTest(rows=rows, pitch=pitch, gap=gap):
                        addresses = [4 * (r * pitch + start + lane)
                                     for r in range(rows) for start in (0, gap)
                                     for lane in range(8)]
                        self.assertEqual(window_block_count(rows, pitch, gap),
                                         len({addr // 128 for addr in addresses}))
                        self.assertEqual(len({addr // 32 for addr in addresses}), 2 * rows)

    def test_same_coverage_leaves_misses_and_time_undetermined(self):
        capacity = 32 * 1024 * 1024
        for width in (104, 136):
            case = f"family-{width}-16-80"
            layout = family_geometry(case, capacity)
            rows = layout["rows"]
            self.assertEqual(layout["pad_words"], 24)
            info = dict(layout, rounds=40, assumed_tag_bytes=128, grid_barrier=False,
                        warmup_rounds=1, l2_bytes=capacity)
            point = predict(case, info)
            self.assertEqual(point["base_footprint_bytes"], rows * 192)
            self.assertEqual(point["pad_footprint_bytes"], rows * 128)
            self.assertEqual(point["issued_read_sectors_per_arm"], rows * 2 * 40)
            self.assertEqual(point["l2_read_miss_status"], "undetermined")
            self.assertEqual(point["kernel_speedup_status"], "undetermined")
            self.assertIsNone(point["kernel_speedup_point"])
            with self.assertRaises(ValueError):
                predict(case, dict(info, base_tag_count=rows))

    def test_original_and_holdout_shapes(self):
        capacity = 32 * 1024 * 1024
        for case, rows in (("original", 196608), ("holdout", 209664)):
            layout = family_geometry(case, capacity)
            self.assertEqual(layout["rows"], rows)
            self.assertEqual(layout["pad_words"], 8)
            self.assertEqual(layout["base_tag_count"], 7 * rows // 4)
            self.assertEqual(layout["pad_tag_count"], rows)


if __name__ == "__main__":
    unittest.main()
