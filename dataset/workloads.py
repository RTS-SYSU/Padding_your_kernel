"""Rebuild the two-window workload catalog and CUDA initializer payloads on a CPU."""

import argparse
import json
from math import gcd
from pathlib import Path
import tempfile


WINDOW_WORDS = 8
BLOCK_BYTES = 128
BLOCK_WORDS = BLOCK_BYTES // 4
ROW_MULTIPLE = 256
PADDING_SENTINEL = 0xDEADC0DE
WIDTHS = tuple(w for w in range(104, 217, 8) if w % BLOCK_WORDS)
OFFSETS = (8, 16, 24)
PERCENTS = (76, 78, 80, 82, 84)
HISTORICAL_F81 = ((104, 24), (112, 16), (176, 16))
CONTROL_WIDTHS = (104, 136)
CONTROL_PERCENTS = (40, 100)


def _mean_blocks_per_row(pitch, offset):
    """Count distinct aligned 128-byte blocks per row over one row-start cycle."""
    period = BLOCK_WORDS // gcd(pitch, BLOCK_WORDS)
    blocks = 0
    for row in range(period):
        phase = (row * pitch) % BLOCK_WORDS
        # Both eight-word windows start on sector boundaries and fit in one block each.
        blocks += 1 if phase + offset < BLOCK_WORDS else 2
    return blocks / period


def catalog():
    """Return device-independent input specifications with bench.cu case names."""
    specs = [("original", 56, 24, 75, "legacy_56"), ("holdout", 56, 24, 80, "legacy_56")]
    for width in WIDTHS:
        for offset in OFFSETS:
            levels = [(percent, "main_grid") for percent in PERCENTS]
            if (width, offset) in HISTORICAL_F81:
                levels.append((81, "historical_81"))
            if width in CONTROL_WIDTHS:
                levels.extend((percent, "capacity_controls") for percent in CONTROL_PERCENTS)
            for percent, stratum in sorted(levels):
                specs.append((f"family-{width}-{offset}-{percent}", width, offset, percent, stratum))
    cases = []
    for name, width, offset, percent, stratum in specs:
        pad = (-width) % BLOCK_WORDS
        cases.append(dict(case_id=name, logical_words=width, second_window_words=offset,
                          l2_percent=percent, pad_words=pad, design_stratum=stratum,
                          base_mean_128b_blocks_per_row=_mean_blocks_per_row(width, offset),
                          pad_mean_128b_blocks_per_row=_mean_blocks_per_row(width + pad, offset)))
    return cases


def target_rows(l2_bytes, l2_percent):
    """Match the integer footprint and row-count truncation in bench.cu::make_plan."""
    if type(l2_bytes) is not int or l2_bytes <= 0:
        raise ValueError("l2_bytes must be a positive integer")
    if type(l2_percent) is not int or not 1 <= l2_percent <= 100:
        raise ValueError("l2_percent must be an integer between 1 and 100")
    rows = l2_bytes * l2_percent // (100 * BLOCK_BYTES * ROW_MULTIPLE) * ROW_MULTIPLE
    if rows == 0:
        raise ValueError("L2 capacity and footprint fraction give zero rows")
    return rows


def materialize(case_id, output, *, rows=None, l2_bytes=None):
    """Write the initialized logical/base and padded row-major matrices to a new directory.

    Supply rows for an explicit-size fixture or l2_bytes for CUDA's device-relative
    row count. NPY files contain the initialized table words only. CUDA's separate
    allocation rounding, cache conditioning, and kernel timing belong to the runner.
    The block means describe a full row-start cycle with a 128-byte-aligned table
    base, including when a small fixture ends partway through that cycle.
    """
    if (rows is None) == (l2_bytes is None):
        raise ValueError("Specify exactly one of rows and l2_bytes")
    spec = next((item for item in catalog() if item["case_id"] == case_id), None)
    if spec is None:
        raise ValueError(f"Unknown dataset case {case_id!r}")
    if l2_bytes is not None:
        rows = target_rows(l2_bytes, spec["l2_percent"])
    width = spec["logical_words"]
    pitch = width + spec["pad_words"]
    offset = spec["second_window_words"]
    if type(rows) is not int or rows <= 0 or rows * pitch > 2**31 - 1:
        raise ValueError("Invalid row count for this CUDA pitch")

    metadata = dict(**spec, rows=rows, row_source="explicit" if l2_bytes is None else "l2_capacity",
                    l2_bytes=l2_bytes, dtype="<u4", order="C", padding_sentinel=PADDING_SENTINEL,
                    windows_half_open=[[0, WINDOW_WORDS], [offset, offset + WINDOW_WORDS]],
                    logical_file="logical_base.npy", base_file="logical_base.npy", padded_file="padded.npy")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")

    import numpy as np

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as stage:
        payload = Path(stage) / "payload"
        payload.mkdir()
        with (payload / metadata["base_file"]).open("wb") as base, (payload / metadata["padded_file"]).open("wb") as padded:
            for file, columns in ((base, width), (padded, pitch)):
                header = dict(descr="<u4", fortran_order=False, shape=(rows, columns))
                np.lib.format.write_array_header_1_0(file, header)
            # bench.cu::field_value multiplies 1-based indices modulo 2^32 before XOR.
            column_terms = np.arange(1, width + 1, dtype=np.uint32) * np.uint32(2246822519)
            for start in range(0, rows, 512):
                end = min(start + 512, rows)
                row_terms = np.arange(start + 1, end + 1, dtype=np.uint32) * np.uint32(2654435761)
                values = (row_terms[:, None] ^ column_terms).astype("<u4", copy=False)
                base.write(values.tobytes())
                padded_values = np.empty((end - start, pitch), dtype="<u4")
                padded_values[:, :width] = values
                padded_values[:, width:] = PADDING_SENTINEL
                padded.write(padded_values.tobytes())
        (payload / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        payload.rename(destination)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Print the device-independent catalog as JSON")
    command = commands.add_parser("materialize", help="Write a pair of NPY matrices")
    command.add_argument("case_id")
    command.add_argument("--output", type=Path, required=True, help="New output directory")
    sizing = command.add_mutually_exclusive_group(required=True)
    sizing.add_argument("--rows", type=int, help="Explicit fixture size")
    sizing.add_argument("--l2-bytes", type=int, help="Device-reported L2 capacity in bytes")
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(catalog(), indent=2))
    else:
        metadata = materialize(args.case_id, args.output, rows=args.rows, l2_bytes=args.l2_bytes)
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
