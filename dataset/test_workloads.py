import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np

if __package__:
    from . import workloads
else:
    import workloads


class WorkloadTests(unittest.TestCase):
    def test_catalog_geometry_and_historical_ids(self):
        cases = workloads.catalog()
        self.assertEqual(Counter(c["design_stratum"] for c in cases),
                         {"main_grid": 180, "historical_81": 3, "legacy_56": 2, "capacity_controls": 12})
        self.assertEqual(len({c["case_id"] for c in cases}), len(cases))
        self.assertEqual({c["case_id"] for c in cases if c["l2_percent"] == 81},
                         {"family-104-24-81", "family-112-16-81", "family-176-16-81"})
        self.assertEqual({(c["logical_words"], c["second_window_words"], c["l2_percent"])
                          for c in cases if c["design_stratum"] == "capacity_controls"},
                         {(w, g, f) for w in (104, 136) for g in (8, 16, 24) for f in (40, 100)})
        for case in cases:
            self.assertEqual(case["pad_words"], (-case["logical_words"]) % 32)
            self.assertLessEqual(case["second_window_words"] + 8, case["logical_words"])
        self.assertEqual([(c["case_id"], c["logical_words"], c["second_window_words"], c["l2_percent"])
                          for c in cases if c["design_stratum"] == "legacy_56"],
                         [("original", 56, 24, 75), ("holdout", 56, 24, 80)])

    def test_block_means_match_all_accessed_addresses(self):
        cases = {case["case_id"]: case for case in workloads.catalog()}
        for width in (104, 136):
            for offset, mean in ((8, 1.25), (16, 1.5), (24, 1.75)):
                case = cases[f"family-{width}-{offset}-80"]
                self.assertEqual(case["base_mean_128b_blocks_per_row"], mean)
                self.assertEqual(case["pad_mean_128b_blocks_per_row"], 1.0)
        for case in cases.values():
            offset = case["second_window_words"]
            columns = [*range(8), *range(offset, offset + 8)]
            for arm, pad in (("base", 0), ("pad", case["pad_words"])):
                pitch = case["logical_words"] + pad
                total = sum(len({(row * pitch + col) * 4 // 128 for col in columns}) for row in range(8))
                self.assertEqual(total / 8, case[f"{arm}_mean_128b_blocks_per_row"])

    def test_device_rows_and_lower_boundary(self):
        l2 = 3 * 1024 * 1024
        self.assertEqual(workloads.target_rows(l2, 75), 18432)
        self.assertEqual(workloads.target_rows(l2, 80), 19456)
        self.assertEqual(workloads.target_rows(40960, 80), 256)
        with self.assertRaises(ValueError):
            workloads.target_rows(40959, 80)

    def test_payload_matches_cuda_on_both_sides_of_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "fixture"
            meta = workloads.materialize("family-104-16-80", folder, rows=513)
            self.assertEqual(meta, json.loads((folder / "metadata.json").read_text(encoding="utf-8")))
            self.assertEqual(meta["logical_file"], meta["base_file"])
            self.assertEqual(meta["windows_half_open"], [[0, 8], [16, 24]])
            self.assertEqual((meta["base_mean_128b_blocks_per_row"], meta["pad_mean_128b_blocks_per_row"]),
                             (1.5, 1.0))
            base = np.load(folder / meta["base_file"], allow_pickle=False)
            pad = np.load(folder / meta["padded_file"], allow_pickle=False)
            self.assertEqual((base.shape, pad.shape), ((513, 104), (513, 128)))
            self.assertEqual((base.dtype.str, pad.dtype.str), ("<u4", "<u4"))
            self.assertTrue(base.flags.c_contiguous and pad.flags.c_contiguous)
            self.assertEqual((int(base[0, 0]), int(base[0, 16]), int(base[1, 103])),
                             (0x1BDCB3C6, 0x7A9F0856, 0x5BA4B33A))
            for row in (0, 511, 512):
                expected = [((2654435761 * (row + 1)) ^ (2246822519 * (col + 1))) & 0xffffffff
                            for col in range(104)]
                np.testing.assert_array_equal(base[row], expected)
            np.testing.assert_array_equal(base, pad[:, :104])
            self.assertTrue(np.all(pad[:, 104:] == 0xDEADC0DE))
            with self.assertRaises(FileExistsError):
                workloads.materialize("family-104-16-80", folder, rows=4)

    def test_device_materialization_and_invalid_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "device"
            for sizes in ({}, {"rows": 0}, {"rows": 4, "l2_bytes": 40960}, {"rows": 2**31}):
                with self.assertRaises(ValueError):
                    workloads.materialize("holdout", folder, **sizes)
                self.assertFalse(folder.exists())
            meta = workloads.materialize("holdout", folder, l2_bytes=40960)
            self.assertEqual((meta["rows"], meta["row_source"], meta["l2_bytes"]),
                             (256, "l2_capacity", 40960))


if __name__ == "__main__":
    unittest.main()
