"""Conditional L2 sector-miss baseline for the unchanged paired-window kernel.

The set function, associativity, true LRU, table base, sector-on-demand fill
and warp interleaving are explicit candidate device rules. These runs are
conditional estimates, not measured GPU parameters or prediction intervals.
The model tracks table reads only. After the scrub it assumes no table lines
resident, runs the separate warm sweep, and starts counting at target entry.
Output stores, other traffic, in-flight merging and fills spanning sectors are
not modeled. A set lookup/fill here is instantaneous in the chosen order.
"""

import argparse
from array import array
import json

from prediction import BLOCK_BYTES, BLOCK_WORDS, ROUNDS, WINDOW_WORDS, family_geometry


SECTORS_PER_LINE = BLOCK_WORDS // WINDOW_WORDS
ROWS_PER_WARP = 32 // WINDOW_WORDS
WARPS_PER_BLOCK = 256 // 32


class Cache:
    def __init__(self, l2_bytes, ways, index):
        if ways <= 0 or l2_bytes <= 0 or l2_bytes % (BLOCK_BYTES * ways):
            raise ValueError("capacity must be a positive whole number of sets")
        if index not in ("modulo", "xor-fold"):
            raise ValueError("unknown set index")
        self.ways = ways
        self.set_count = l2_bytes // (BLOCK_BYTES * ways)
        self.fold = index == "xor-fold"
        self.sets = [{} for _ in range(self.set_count)]

    def read(self, sector):
        line, offset = divmod(sector, SECTORS_PER_LINE)
        index = line ^ (line // self.set_count) if self.fold else line
        tags = self.sets[index % self.set_count]
        valid = tags.pop(line, 0)
        if not valid and len(tags) == self.ways:
            del tags[next(iter(tags))]
        bit = 1 << offset
        tags[line] = valid | bit
        return not (valid & bit)

    def state(self):
        # Ordered tags preserve LRU; zero separates sets (all valid masks > 0).
        state = array("Q")
        for tags in self.sets:
            state.extend((line << SECTORS_PER_LINE) | bits for line, bits in tags.items())
            state.append(0)
        return state


def issue_rows(rows, blocks, schedule, passes):
    """Yield (zero-based local pass, first row of warp's four) in issue order.

    In 'sweep', all warps finish one scan before any begins the next. In
    'staggered', each half of the grid finishes the supplied period before
    the other half runs. The target period is two scans (one for an odd final
    scan). Both are candidate interleavings, not barriers in the kernel.
    Warp id is (first_row // 4) % (blocks * WARPS_PER_BLOCK). Each warp issues
    first-window then second-window load for this iteration.
    """
    if rows <= 0 or rows % ROWS_PER_WARP or blocks < 1 or passes < 1:
        raise ValueError("expected complete four-row warps and a positive launch")
    warps = blocks * WARPS_PER_BLOCK
    cohorts = [(0, warps)]
    if schedule == "staggered":
        cohorts = [(0, warps // 2), (warps // 2, warps)]
    elif schedule != "sweep":
        raise ValueError("unknown warp schedule")
    stride = warps * ROWS_PER_WARP
    for first_warp, end_warp in cohorts:
        for pass_index in range(passes):
            for start in range(0, rows, stride):
                first = start + first_warp * ROWS_PER_WARP
                end = min(start + end_warp * ROWS_PER_WARP, rows)
                for row in range(first, end, ROWS_PER_WARP):
                    yield pass_index, row


def run_scans(cache, rows, pitch, gap, blocks, base_line, schedule, passes):
    """Return one byte per warp iteration, low/high nibbles = load A/B miss masks."""
    stride = pitch // WINDOW_WORDS
    window_offset = gap // WINDOW_WORDS
    masks = bytearray()
    for _, row in issue_rows(rows, blocks, schedule, passes):
        start = base_line * SECTORS_PER_LINE + row * stride
        mask = 0
        for window, offset in enumerate((0, window_offset)):
            for group in range(ROWS_PER_WARP):
                if cache.read(start + group * stride + offset):
                    mask |= 1 << (window * ROWS_PER_WARP + group)
        masks.append(mask)
    return masks


def target_periods(rows, pitch, gap, blocks, l2_bytes, ways, index, schedule,
                   rounds=ROUNDS, base_line=0):
    """Yield (consecutive repetitions, passes per period, packed miss masks).

    Pair masks with issue_rows for the same period to recover exact ordered
    warp/load misses under this model. Repeat a period only after verifying
    the complete cache state including per-sector valid bits and LRU order.
    """
    if pitch < 56 or pitch % WINDOW_WORDS or gap not in (8, 16, 24):
        raise ValueError("unsupported double-window geometry")
    if base_line < 0 or rounds < 1:
        raise ValueError("expected nonnegative aligned base and positive round count")
    cache = Cache(l2_bytes, ways, index)
    run_scans(cache, rows, pitch, gap, blocks, base_line, "sweep", 1)
    period = 2 if schedule == "staggered" else 1
    remaining = rounds
    previous = cache.state()
    while remaining:
        length = min(period, remaining)
        masks = run_scans(cache, rows, pitch, gap, blocks, base_line, schedule, length)
        current = cache.state()
        repeats = remaining // length if current == previous else 1
        yield repeats, length, masks
        remaining -= repeats * length
        previous = current


def estimate_arm(rows, pitch, gap, blocks, l2_bytes, ways, index, schedule,
                 rounds=ROUNDS, base_line=0):
    counts = [[0] * (ROWS_PER_WARP + 1) for _ in range(ROWS_PER_WARP + 1)]
    for repeats, _, masks in target_periods(rows, pitch, gap, blocks, l2_bytes, ways,
                                            index, schedule, rounds, base_line):
        frequencies = [0] * 256
        for mask in masks:
            frequencies[mask] += 1
        for mask, frequency in enumerate(frequencies):
            counts[(mask & 15).bit_count()][(mask >> 4).bit_count()] += frequency * repeats
    per_load = [0] * (ROWS_PER_WARP + 1)
    for first, second_counts in enumerate(counts):
        for second, count in enumerate(second_counts):
            per_load[first] += count
            per_load[second] += count
    assert sum(per_load) * ROWS_PER_WARP == rows * 2 * rounds
    return {"pitch_words": pitch, "read_sectors": rows * 2 * rounds,
            "sector_misses": sum(i * n for i, n in enumerate(per_load)),
            "misses_per_warp_load": per_load, "paired_missing_sector_counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--ways", type=int, required=True)
    parser.add_argument("--index", choices=("modulo", "xor-fold"), default="modulo")
    parser.add_argument("--schedule", choices=("sweep", "staggered"), default="sweep")
    parser.add_argument("--base-line", type=int, default=0,
                        help="assumed original table base in 128-byte blocks")
    parser.add_argument("--padded-base-line", type=int, default=0,
                        help="assumed padded table base in 128-byte blocks; allocations are separate")
    args = parser.parse_args()
    layout = family_geometry(args.case, args.l2_bytes)
    result = {"status": "conditional_uncalibrated", **vars(args), "rows": layout["rows"],
              "address_model": "separate assumed linear block addresses; physical bases unknown",
              "warmup": "one sweep-order warm scan after no table lines resident",
              "replacement": "true LRU, demand sector fill, table reads only"}
    for name, pad, address in (("base", 0, args.base_line),
                               ("padded", layout["pad_words"], args.padded_base_line)):
        result[name] = estimate_arm(layout["rows"], layout["logical_words"] + pad,
                                    layout["second_window_words"], args.blocks,
                                    args.l2_bytes, args.ways, args.index, args.schedule,
                                    base_line=address)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
