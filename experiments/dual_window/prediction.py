"""Derive row-tail padding and count the address blocks and sectors of the two windows."""

import math
import re


BLOCK_BYTES = 128
BLOCK_WORDS = BLOCK_BYTES // 4
WINDOW_WORDS = 8
ROW_MULTIPLE = 256
ROUNDS = 40


def window_block_count(rows, pitch_words, second_window_words):
    """Count distinct 128-byte blocks across both eight-word windows of all rows."""
    if (type(rows) is not int or rows < 0 or type(pitch_words) is not int
            or pitch_words < 56 or pitch_words % WINDOW_WORDS or type(second_window_words) is not int
            or second_window_words not in (8, 16, 24)):
        raise ValueError("Expected disjoint row coverage and sector-aligned windows")
    # These pitches keep the last accessed block of a row separate from the next
    # row's first block. Each window is a sector and wholly inside one block.
    period = BLOCK_WORDS // math.gcd(pitch_words, BLOCK_WORDS)
    blocks_by_phase = []
    for row in range(period):
        phase = row * pitch_words % BLOCK_WORDS
        blocks_by_phase.append(1 + (phase + second_window_words) // BLOCK_WORDS)
    periods, tail = divmod(rows, period)
    return periods * sum(blocks_by_phase) + sum(blocks_by_phase[:tail])


def family_geometry(case, l2_bytes):
    """Derive the registered case from device capacity and fixed logical windows."""
    fixed_cases = {
        "original": (56, 24, 3, 4), "holdout": (56, 24, 4, 5),
        "w72_mid": (72, 24, 11, 16), "w80_high": (80, 24, 13, 16),
    }
    if case in fixed_cases:
        columns, gap, numerator, denominator = fixed_cases[case]
    else:
        match = re.fullmatch(r"family-([1-9][0-9]*)-([1-9][0-9]*)-([1-9][0-9]*)", case)
        if not match:
            raise ValueError(f"Unknown two-window case {case!r}")
        columns, gap, numerator = map(int, match.groups())
        denominator = 100
        if (not 64 <= columns <= 4096 or columns % WINDOW_WORDS or columns % BLOCK_WORDS == 0
                or gap not in (8, 16, 24) or numerator > denominator):
            raise ValueError(f"Unsupported width, window offset or percent for {case!r}")
    if type(l2_bytes) is not int or l2_bytes <= 0:
        raise ValueError("Expected a positive L2 capacity in bytes")
    rows = l2_bytes * numerator // (denominator * BLOCK_BYTES * ROW_MULTIPLE) * ROW_MULTIPLE
    if not rows:
        raise ValueError("L2 capacity and case percent give zero rows")
    pad = (-columns) % BLOCK_WORDS
    return {"case": case, "rows": rows, "logical_words": columns,
            "second_window_words": gap, "pad_words": pad,
            "base_tag_count": window_block_count(rows, columns, gap),
            "pad_tag_count": window_block_count(rows, columns + pad, gap),
            "source_sectors_per_sweep": 2 * rows}


def predict(case, info):
    """Verify the CUDA plan independently and freeze what the addresses determine."""
    layout = family_geometry(case, info["l2_bytes"])
    if (any(info.get(key) != value for key, value in layout.items())
            or info["rounds"] != ROUNDS or info["assumed_tag_bytes"] != BLOCK_BYTES
            or info["grid_barrier"] or info["warmup_rounds"] != 1):
        raise ValueError(f"Unexpected device plan for {case}")
    return {
        "pad_words": layout["pad_words"],
        "base_footprint_bytes": layout["base_tag_count"] * BLOCK_BYTES,
        "pad_footprint_bytes": layout["pad_tag_count"] * BLOCK_BYTES,
        "issued_read_sectors_per_arm": layout["source_sectors_per_sweep"] * ROUNDS,
        "l2_read_miss_status": "undetermined",
        "kernel_speedup_status": "undetermined",
        "kernel_speedup_point": None,
    }
