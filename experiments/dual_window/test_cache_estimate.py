"""Check conditional cache replay against literal warp and sector enumeration."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_estimate import Cache, estimate_arm, issue_rows, target_periods


def reference(rows, pitch, gap, blocks, capacity, ways, index, schedule, rounds, base):
    # Independent slow implementation. Each cache set is an MRU-first list of
    # (line, set of explicitly filled sectors), not the estimator's LRU dict.
    sets = [[] for _ in range(capacity // (128 * ways))]
    def miss(byte):
        line = byte // 128
        sector = byte % 128 // 32
        selected = line if index == "modulo" else line ^ (line // len(sets))
        tags = sets[selected % len(sets)]
        valid = set()
        for i, (tag, bits) in enumerate(tags):
            if tag == line:
                valid = bits
                tags.pop(i)
                break
        missed = sector not in valid
        valid.add(sector)
        tags.insert(0, (line, valid))
        del tags[ways:]
        return missed

    def group_scans(warps, scans):
        all_warps = 8 * blocks
        trace = []
        for _ in range(scans):
            for step in range((rows + 4 * all_warps - 1) // (4 * all_warps)):
                for warp in warps:
                    first_tid = 32 * warp
                    initial_row = first_tid // 8 + step * blocks * 256 // 8
                    if initial_row >= rows:
                        continue
                    packed = 0
                    for window, col in enumerate((0, gap)):
                        # Form per-warp sectors from all actual thread addresses.
                        addresses = []
                        for lane in range(32):
                            row = initial_row + lane // 8
                            word = row * pitch + lane % 8 + col
                            addresses.append((base * 128 + 4 * word) // 32)
                        for group, sector in enumerate(dict.fromkeys(addresses)):
                            packed |= int(miss(32 * sector)) << (4 * window + group)
                    trace.append(packed)
        return trace

    all_warps = 8 * blocks
    group_scans(range(all_warps), 1)
    trace = []
    if schedule == "sweep":
        return group_scans(range(all_warps), rounds)
    for i in range(0, rounds, 2):
        scans = min(2, rounds - i)
        trace.extend(group_scans(range(all_warps // 2), scans))
        trace.extend(group_scans(range(all_warps // 2, all_warps), scans))
    return trace


class ConditionalCacheModel(unittest.TestCase):
    def test_sector_validity_and_lru_eviction(self):
        cache = Cache(256, 2, "modulo")
        self.assertEqual([cache.read(i) for i in (0, 1, 0, 4, 0, 8, 0, 1, 4, 5)],
                         [True, True, False, True, False, True, False, False, True, True])

    def test_ordered_miss_masks_match_literal_replay(self):
        for schedule in ("sweep", "staggered"):
            for index in ("modulo", "xor-fold"):
                for ways, sets in ((2, 19), (3, 17)):
                    params = (76, 104, 16, 2, ways * sets * 128, ways, index, schedule, 7, 39)
                    with self.subTest(params=params):
                        actual = []
                        for repeats, length, masks in target_periods(*params):
                            self.assertEqual(len(masks), len(list(issue_rows(76, 2, schedule, length))))
                            actual.extend(masks * repeats)
                        wanted = reference(*params)
                        self.assertEqual(actual, wanted)
                        summary = estimate_arm(*params)
                        self.assertEqual(summary["sector_misses"], sum(mask.bit_count() for mask in wanted))
                        joint = [[0] * 5 for _ in range(5)]
                        for mask in wanted:
                            joint[(mask & 15).bit_count()][(mask >> 4).bit_count()] += 1
                        self.assertEqual(summary["paired_missing_sector_counts"], joint)


if __name__ == "__main__":
    unittest.main()
