"""Assemble PADGain-DualWindow v0.1 from recorded target runs."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from observations import write_observations
from verify_dataset import file_hash, read_lines, verify
from workloads import catalog, materialize


HERE = Path(__file__).resolve().parent
VERSION = "PADGain-DualWindow-v0.1"
BENCH_SHA256 = "a03abf0eaf66f47bacd2b05093c31872ade2bca363751882bd0f1602f5610401"


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def summarize(cases, measured, reported):
    counts = defaultdict(Counter)
    per_device = []
    for device in sorted({r["gpu_name"] for r in measured + reported}):
        native = [r for r in measured if r["gpu_name"] == device]
        summaries = [r for r in reported if r["gpu_name"] == device]
        per_device.append(dict(gpu_name=device, native_pairs=len(native),
                               native_distinct_cases=len({r["case_id"] for r in native}),
                               native_pairs_with_l2=sum(r["base_l2_read_miss_sectors"] is not None for r in native),
                               reported_pairs=len(summaries)))
        for row in native + summaries:
            counts[row["case_id"]][device] += 1
    return dict(version=VERSION, workload_count=len(cases), native_pairs=len(measured),
                native_distinct_cases=len({r["case_id"] for r in measured}), reported_pairs=len(reported),
                cases_without_observations=sum(c["case_id"] not in counts for c in cases),
                devices=per_device, observations_by_case={c["case_id"]: counts[c["case_id"]] for c in cases})


def package(output, archives, results, bench_source=None):
    if output.exists():
        raise FileExistsError(f"Use a new dataset directory: {output}")
    if bench_source is None:
        bench_source = results / VERSION / "tools" / "bench.cu"
    if not bench_source.is_file() or file_hash(bench_source) != BENCH_SHA256:
        raise ValueError(f"Supply the v0.1 bench.cu snapshot with --bench-source (SHA-256 {BENCH_SHA256})")
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        root = Path(temp) / VERSION
        root.mkdir()
        write_observations(archives, results, root)
        cases = catalog()
        for case in cases:
            case["pad_pitch_words"] = case["logical_words"] + case["pad_words"]
        with (root / "workloads.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for case in cases:
                stream.write(json.dumps(case) + "\n")
        measured = read_lines(root / "measured.jsonl")
        reported = read_lines(root / "reported.jsonl")
        write_json(root / "coverage.json", summarize(cases, measured, reported))
        families = defaultdict(list)
        for case in cases:
            families[case["pad_pitch_words"]].append(case["case_id"])
        write_json(root / "folds.json", [
            {"held_out_pad_pitch_words": pitch, "test_case_ids": members}
            for pitch, members in sorted(families.items())])

        # The archived V100 original run supplies the size of this reconstructed input.
        reference = next(r for r in measured if r["case_id"] == "original" and "V100" in r["gpu_name"])
        materialize("original", root / "example", l2_bytes=reference["l2_bytes"])
        write_json(root / "example" / "source.json",
                   {key: reference[key] for key in ("observation_id", "gpu_name", "case_id", "rows", "evidence")})
        shutil.copyfile(HERE / "README.md", root / "README.md")
        scripts = root / "tools"
        scripts.mkdir()
        shutil.copyfile(bench_source, scripts / "bench.cu")
        for src in [HERE / "workloads.py", HERE / "verify_dataset.py", HERE.parent / "tools" / "build.ps1"]:
            shutil.copyfile(src, scripts / src.name)
        paths = sorted(p for p in root.rglob("*") if p.is_file())
        write_json(root / "sha256.json", {p.relative_to(root).as_posix(): file_hash(p) for p in paths})
        verify(root)

        # Stable member order and timestamps, including when rebuilt on a different day.
        with zipfile.ZipFile(root / (VERSION + ".zip"), "w") as archive:
            for path in sorted(paths + [root / "sha256.json"]):
                entry = zipfile.ZipInfo(path.relative_to(root).as_posix(), (2026, 11, 22, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                with path.open("rb") as source, archive.open(entry, "w") as dest:
                    shutil.copyfileobj(source, dest)
        with zipfile.ZipFile(root / (VERSION + ".zip")) as archive:
            if archive.testzip() is not None:
                raise ValueError("Invalid ZIP member")
        root.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory under an existing parent")
    parser.add_argument("--bench-source", type=Path, help="Original bench.cu from the v0.1 release")
    args = parser.parse_args()
    package(args.output, args.archives_root, args.results_root, args.bench_source)
