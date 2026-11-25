"""Run one configured, warm-only native research benchmark at a time."""

import ast
import csv
from copy import deepcopy
from importlib.util import find_spec
import json
from math import isclose, isfinite, prod
import os
from pathlib import Path
import signal
from statistics import median
import subprocess
import sys
from tempfile import mkdtemp
from threading import Lock, Thread
import traceback
from uuid import uuid4

from .presets import analyze_preset, list_presets


SUITE = Path("experiment_23_suite_single_pad")
RUNNER = SUITE / "run_experiment.py"
PROTOCOL = (3, 11, 3)
PREPARE_TIMEOUT = 120
MEASURE_TIMEOUT = 300

# This fixed, CPU-only bridge asks the trusted local runner to freeze its
# sources and validate the actual fixture before involving CUDA. Arguments
# are normalized preset identifiers and a directory owned by the manager.
_PREPARE = """
import json
from pathlib import Path
import sys

out, suite = map(Path, sys.argv[1:3])
slug, case = sys.argv[3:5]
sys.path[:0] = [str(suite), str(suite.parent)]
from plan import freeze
from run_experiment import layouts
from workloads import workload

frozen = freeze(out)
if frozen["cache_preparation"] != ("primed",):
    raise ValueError("The source is not the warm-only runner")
rows = [r for r in frozen["rows"] if (r["slug"], r["case"]) == (slug, case)]
if len(rows) != 1:
    raise ValueError("Expected exactly one source case")
row = rows[0]
_, spec, _ = workload(row)
cases = layouts(row, spec)
fixture = {}
for p, data in cases.items():
    fixture[p] = {
        "buffers": {
            name: [list(value.shape), value.dtype.name]
            for name, value in data["buffers"].items()
        },
        "bytes": sum(int(value.nbytes) for value in data["buffers"].values()),
        "outputs": {name: int(value.size) for name, value in data["expected"].items()},
        "launches": len(data["steps"]),
    }
(out / "preflight.json").write_text(json.dumps(fixture), encoding="utf-8")
"""


class BusyError(RuntimeError):
    """A native job already owns this manager."""


class _RunError(RuntimeError):
    """A static, user-facing error that can be returned without leaking paths."""


def _preset_target(name):
    if not isinstance(name, str):
        raise ValueError("请选择允许原生实测的预设。")
    canonical = name.strip().lower()
    preset = next((item for item in list_presets() if item.name == canonical), None)
    if preset is None:
        raise ValueError("请选择允许原生实测的预设。")
    slug = "mvt" if canonical == "mvt-4096" else canonical
    case = preset.case if slug == "mvt" else "default"
    source = "experiment_07_mvt" if slug == "mvt" else preset.source.split("/", 1)[0]
    return canonical, slug, case, source, analyze_preset(canonical)


def _read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def _warm_only(path):
    """Read the runner's literal mode declaration without importing research code."""
    assignments = []
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MODES" for target in node.targets
        ):
            assignments.append(ast.literal_eval(node.value))
    return assignments == [("primed",)]


def _positive(value):
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ValueError("Expected a positive finite timing value")
    return value


def _fixture_matches(frozen, fixture, target):
    canonical, slug, case, source, recommendation = target
    rows = [row for row in frozen["rows"] if (row["slug"], row["case"]) == (slug, case)]
    if len(rows) != 1:
        raise ValueError("The frozen plan has no unique matching case")
    row = rows[0]
    if (row["source"] != source or row["p"] != recommendation.p
            or row["operator"].removeprefix("Rodinia ") != recommendation.operator):
        raise ValueError("The frozen plan differs from the current preset")
    if (frozen["cache_preparation"] != ["primed"]
            or (frozen["processes"], frozen["rounds"], frozen["samples_per_round_and_mode"]) != PROTOCOL):
        raise ValueError("The frozen plan uses a different measurement protocol")

    pads = [0, row["p"]] if row["p"] else [0]
    if set(fixture) != {str(p) for p in pads}:
        raise ValueError("Unexpected layouts in the selected source case")
    for pad in pads:
        buffers = fixture[str(pad)]["buffers"]
        for layout in recommendation.layouts:
            shape, dtype = buffers[layout.operand.lower()]
            want_dtype = "int32" if canonical == "pathfinder" else "float32"
            if (len(shape) < 2 or prod(shape[:-1]) != layout.rows
                    or shape[-1] != layout.logical_width + pad or dtype != want_dtype):
                raise ValueError("The selected physical shape differs from the fixed preset")
        if slug == "bfs":
            nodes, node_dtype = buffers["nodes"]
            edges, edge_dtype = buffers["edges"]
            if nodes != [65536, 2] or len(edges) != 1 or node_dtype != "int32" or edge_dtype != "int32":
                raise ValueError("The selected CSR fixture differs from the fixed BFS case")
    if fixture[str(row["p"])]["bytes"] - fixture["0"]["bytes"] != recommendation.extra_bytes:
        raise ValueError("The selected footprint differs from the fixed preset")
    return row


def _checked_correctness(path, pads, fixture, processes):
    expected = {
        (process, pad, name): elements
        for process in range(processes)
        for pad in pads
        for name, elements in fixture[str(pad)]["outputs"].items()
    }
    seen = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for entry in csv.DictReader(stream):
            key = (int(entry["process"]), int(entry["p"]), entry["output"])
            if entry["passed"] != "1" or key in seen:
                raise ValueError("Incorrect or duplicated result check")
            seen[key] = int(entry["elements"])
    if not expected or expected != seen:
        raise ValueError("Missing output correctness checks")
    return len(seen)


def _checked_sample_count(path, pads, processes, rounds, samples):
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for entry in csv.DictReader(stream):
            if entry["mode"] != "primed":
                raise ValueError("A non-warm result entered the warm-only run")
            key = tuple(int(entry[field]) for field in ("process", "round", "p", "sample"))
            process, round_index, pad, index = key
            if (key in seen or pad not in pads or not 0 <= process < processes
                    or not 0 <= round_index < rounds or not 0 <= index < samples):
                raise ValueError("Unexpected or duplicated timing sample")
            _positive(float(entry["time_us"]))
            seen.add(key)
    if len(seen) != processes * rounds * len(pads) * samples:
        raise ValueError("Missing timing samples")
    return len(seen)


def _read_measurement(folder, target, frozen, row, fixture):
    canonical, slug, case, _, recommendation = target
    p = recommendation.p
    pads = [0, p] if p else [0]
    path = folder / slug if case == "default" else folder / slug / case
    result = _read_json(path / "result.json")
    if (result["prediction_id"] != frozen["frozen_utc"] or result["prediction"] != row
            or result["measured_pads"] != pads or set(result["preparation"]) != {"primed"}):
        raise ValueError("Result does not belong to the frozen warm-only preset")
    processes, rounds, samples = PROTOCOL
    recorded_checks = _checked_correctness(path / "correctness.csv", pads, fixture, processes)
    recorded_samples = _checked_sample_count(path / "timing_samples.csv", pads, processes, rounds, samples)
    if result["correctness_checks"] != recorded_checks or result["timing_samples"] != recorded_samples:
        raise ValueError("Result totals disagree with verified records")

    devices = result["devices"]
    if len(devices) != processes or {x["process"] for x in devices} != set(range(processes)):
        raise ValueError("Unexpected number of worker devices")
    names = {x["device"] for x in devices}
    launches = fixture["0"]["launches"]
    if len(names) != 1 or any(not isinstance(name, str) or not name for name in names) or any(
        x["launches"] != launches for x in devices
    ) or type(launches) is not int or launches < 1:
        raise ValueError("Launch counts or device disagree across worker processes")

    measured = result["preparation"]["primed"]
    baseline = _positive(measured["p0_us"])
    ratios = measured["process_ratios"]
    padded, speedup, faster = measured["pad_us"], measured["paired_speedup"], measured["faster_processes"]
    if p:
        _positive(padded)
        _positive(speedup)
        if len(ratios) != processes:
            raise ValueError("Missing paired process medians")
        ratios = [_positive(value) for value in ratios]
        if faster != sum(value > 1 for value in ratios) or not isclose(speedup, median(ratios), rel_tol=1e-12):
            raise ValueError("Paired status disagrees with measured ratios")
        expected = "accepted" if faster == processes else "not_accepted"
    else:
        if ratios or any(value is not None for value in (padded, speedup, faster)):
            raise ValueError("An unpaired baseline must not have a speedup")
        expected = "baseline_only"
    if result["status"] != expected:
        raise ValueError("Unexpected measurement status")

    return {
        "measurement": {
            "preset": canonical, "operator": recommendation.operator, "case": case, "p": p,
            "status": expected, "baseline_us": baseline, "padded_us": padded, "speedup": speedup,
            "faster_processes": faster, "process_speedups": ratios, "launches": launches,
        },
        "device": names.pop(), "preparation": "primed", "processes": processes,
        "rounds": rounds, "samples_per_round": samples, "correctness_checks": recorded_checks,
    }


class LiveJobs:
    """Serialize native subprocess jobs and expose sanitized snapshots."""

    def __init__(self, research_root: Path | None, results_dir: Path):
        self.research_root = Path(research_root).resolve() if research_root is not None else None
        self.results_dir = Path(results_dir).resolve()
        self._lock = Lock()
        self._jobs = {}
        self._active = None
        self._unsafe = False

    def _unavailable(self):
        if self._unsafe:
            return "上一个测量的进程未能确认停止。请关闭残留的测量进程后重新启动工作台。"
        if self.research_root is None:
            return "未配置包含原生 CUDA 测量脚本的研究目录。"
        if os.name != "nt":
            return "当前原生 CUDA 测量脚本仅支持 Windows。"
        try:
            for item in (RUNNER, SUITE / "plan.py", SUITE / "workloads.py", Path("run_suite.py")):
                path = (self.research_root / item).resolve(strict=True)
                if not path.is_file() or not path.is_relative_to(self.research_root):
                    return "配置的研究目录不包含完整测量程序。"
            if not _warm_only(self.research_root / "run_suite.py"):
                return "配置的研究目录未使用仅预热测量协议。"
            if find_spec("numpy") is None:
                return "当前 Python 环境缺少原生测量依赖 NumPy。"
            cuda_path = os.environ.get("CUDA_PATH")
            if not cuda_path or not (Path(cuda_path) / "bin" / "nvrtc64_120_0.dll").is_file():
                return "未找到原生测量所需的 CUDA 12 NVRTC。"
        except (OSError, SyntaxError, ValueError, ImportError):
            return "无法确认研究目录和预热测量协议。"
        return None

    def capabilities(self):
        with self._lock:
            reason = self._unavailable()
            result = {"available": reason is None, "active": deepcopy(self._jobs.get(self._active))}
            if reason is not None:
                result["reason"] = reason
            return result

    def start(self, preset: str):
        target = _preset_target(preset)
        with self._lock:
            if self._active is not None:
                raise BusyError("已有一项正在运行的测量。")
            reason = self._unavailable()
            if reason is not None:
                raise RuntimeError(reason)
            ident = uuid4().hex
            job = {
                "id": ident, "preset": target[0], "status": "running",
                "stage": "preflight", "message": "正在检查固定布局和测量协议。",
            }
            self._jobs[ident] = job
            self._active = ident
            if len(self._jobs) > 64:
                del self._jobs[next(iter(self._jobs))]
            try:
                Thread(
                    target=self._run, args=(ident, target),
                    name=f"padding-job-{ident[:8]}", daemon=False,
                ).start()
            except RuntimeError as error:
                self._active = None
                del self._jobs[ident]
                raise RuntimeError("无法启动后台测量线程。") from error
            return deepcopy(job)

    def get(self, ident):
        with self._lock:
            return deepcopy(self._jobs.get(ident))

    def _update(self, ident, **fields):
        with self._lock:
            self._jobs[ident].update(fields)
            if self._jobs[ident]["status"] != "running":
                self._active = None

    @staticmethod
    def _terminate(process):
        if os.name == "nt":
            system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            try:
                stopped = subprocess.run(
                    [str(system_root / "System32" / "taskkill.exe"), "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                ).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                stopped = False
            if process.poll() is None:
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                stopped = True
            except ProcessLookupError:
                stopped = True
        process.wait(timeout=10)
        return stopped

    def _command(self, args, log, timeout):
        kwargs = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else
                  {"start_new_session": True})
        process = subprocess.Popen(
            args, cwd=self.research_root, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, **kwargs,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                stopped = self._terminate(process)
            except (OSError, subprocess.TimeoutExpired):
                stopped = False
            if not stopped:
                with self._lock:
                    self._unsafe = True
            raise _RunError("测量进程超过时限。请在配置的输出目录查看本次日志。")
        if code:
            raise _RunError(f"测量程序退出（代码 {code}）。请在配置的输出目录查看本次日志。")

    def _run(self, ident, target):
        folder = None
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            folder = Path(mkdtemp(prefix=f"measurement-{ident}-", dir=self.results_dir))
            with (folder / "output.log").open("ab", buffering=0) as log:
                log.write(b"CPU layout and protocol preflight\n")
                _, slug, case, _, _ = target
                self._command(
                    [sys.executable, "-E", "-B", "-u", "-c", _PREPARE,
                     str(folder), str(self.research_root / SUITE), slug, case],
                    log, PREPARE_TIMEOUT,
                )
                frozen = _read_json(folder / "prediction.json")
                fixture = _read_json(folder / "preflight.json")
                row = _fixture_matches(frozen, fixture, target)
                self._update(ident, stage="measuring", message="正在本机 GPU 上运行预热配对测量。")
                log.write(b"Native CUDA benchmark\n")
                self._command(
                    [sys.executable, "-B", "-u", str(self.research_root / RUNNER),
                     "--operator", slug, "--case", case, "--out", str(folder)],
                    log, MEASURE_TIMEOUT,
                )
                self._update(ident, stage="validating", message="正在核对设备、计时和输出校验。")
                result = _read_measurement(folder, target, frozen, row, fixture)
            self._update(ident, status="completed", stage="finished", message="本机测量完成。", **result)
        except Exception as error:
            if folder is not None:
                try:
                    with (folder / "output.log").open("a", encoding="utf-8") as log:
                        traceback.print_exc(file=log)
                except OSError:
                    pass
            message = str(error) if isinstance(error, _RunError) else (
                "无法完成测量或验证结果。请在配置的输出目录查看本次日志。"
            )
            self._update(ident, status="failed", message=message)
