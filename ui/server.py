"""Serve the local padding workbench and read benchmark records."""
import argparse
import ctypes
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "dual_window"))
from current_theory import CASES, pad_for
from jobs import BusyError, ExperimentJobs

CONFIGS = {
    "dual": tuple(CASES),
    "pavia": ("top_left", "center"),
}


@lru_cache(maxsize=1)
def dual_row_counts():
    """Read device 0's L2 capacity without launching a kernel."""
    try:
        driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int()
        capacity = ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, from cuda.h.
        if (driver.cuInit(0)
                or driver.cuDeviceGet(ctypes.byref(device), 0)
                or driver.cuDeviceGetAttribute(ctypes.byref(capacity), 38, device)
                or capacity.value <= 0):
            return {}
    except (OSError, AttributeError):
        return {}

    rows = {}
    for key in CONFIGS["dual"]:
        percent = 75 if key == "original" else 80 if key == "holdout" else int(key.rsplit("-", 1)[1])
        rows[key] = capacity.value * percent // 100 // 128 // 256 * 256
    return rows


def read(path):
    return json.loads(path.read_bytes())


def measured(path, plan):
    cases = {}
    if "geometry" in plan and "pad" in plan:
        pad = plan["pad"]
        summary = path / "summary.json"
        pavia = read(summary) if summary.is_file() else {}
        for key, source in plan["cases"].items():
            item = pavia.get(key, {})
            cases[key] = {
                "rows": 1600, "width": 103, "pad": pad, "offset": 32,
                "word_bytes": 2, "modulus": 16, "device": "RTX 4060 Ti" if path.name == "pavia_current_theory_4060ti_20261124" else None,
                "samples": plan["pairs_per_batch"] * 2, "pair_count": plan["pairs"],
                "roi": [source["y"], source["x"]], "coverage": [plan["geometry"][str(p)]["unique"] for p in (0, pad)],
                "expected": [plan["geometry"][str(p)]["issued"] for p in (0, pad)],
                "bytes": [plan["geometry"][str(p)]["input_bytes"] for p in (0, pad)],
                "time": item.get("times_ms"), "profiles": [
                    [trial[str(p)] for p in (0, pad)] for trial in item.get("profiles", [])],
            }
        return {"kind": "pavia", "run": path.name, "cases": cases}

    for key, source in plan["cases"].items():
        info = source["info"]
        timing, logs = path / f"{key}_timing.json", path / f"{key}_counters.json"
        times = read(timing) if timing.is_file() else {}
        cases[key] = {
            "rows": info["rows"], "width": info["logical_words"], "pad": info["pad_words"],
            "offset": info["second_window_words"], "rounds": info["rounds"], "word_bytes": 4,
            "modulus": source["pitch_modulus_words"], "device": info["device"], "samples": plan["pairs_per_case"],
            "coverage": [info["base_tag_count"], info["pad_tag_count"]], "expected": [source["issued"]] * 2,
            "bytes": [info["base_allocation_bytes"], info["pad_allocation_bytes"]],
            "time": [times["base_ms"], times["pad_ms"]] if times else None,
            "profiles": [[trial[a] for a in ("base", "pad")] for trial in read(logs)] if logs.is_file() else [],
        }
    return {"kind": "dual", "run": path.name, "cases": cases}


def collect(roots):
    runs, skipped = [], []
    for root in roots:
        if not root.is_dir():
            continue
        for plan_path in sorted(root.glob("*/plan.json"), reverse=True):
            try:
                if plan_path.parent.name.startswith("live_") and not (plan_path.parent / ".complete").is_file():
                    continue
                plan = read(plan_path)
                cases = plan.get("cases")
                if not isinstance(cases, dict) or not cases:
                    continue
                first = next(iter(cases.values()))
                if not isinstance(first, dict) or not ("pitch_modulus_words" in first or ("geometry" in plan and "pad" in plan)):
                    continue
                runs.append(measured(plan_path.parent, plan))
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                skipped.append(f"{plan_path.parent.name}: {error}")
    return {"runs": runs, "skipped": skipped}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, roots, jobs, **kwargs):
        self.roots = roots
        self.jobs = jobs
        super().__init__(*args, directory=Path(__file__).parent / "static", **kwargs)

    def send_json(self, body, status=200):
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path != "/api/job":
            return self.send_error(404)
        port = self.server.server_port
        if self.headers.get("Origin") not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            return self.send_json({"error": "请从本地工作台启动实验。"}, 403)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("无效的启动请求。")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("请选择有效的实验和输入。")
            kind, case = body.get("kind"), body.get("case")
            if not isinstance(kind, str) or not isinstance(case, str) or case not in CONFIGS.get(kind, ()):
                raise ValueError("请选择有效的实验和输入。")
            return self.send_json(self.jobs.start(kind, case), 202)
        except BusyError as error:
            return self.send_json({"error": str(error)}, 409)
        except ValueError as error:
            return self.send_json({"error": str(error)}, 400)

    def do_GET(self):
        location = urlsplit(self.path)
        if location.path == "/api/job":
            current = self.jobs.snapshot()
            requested = parse_qs(location.query).get("run", [""])[0]
            busy = bool(current and current["status"] == "running")
            job = current if current and requested == current["run"] else None
            return self.send_json({"job": job, "busy": busy, "configs": CONFIGS,
                                   "dual_rows": dual_row_counts()})
        if location.path == "/api/runs":
            return self.send_json(collect(self.roots))
        if location.path == "/api/pad":
            query = parse_qs(location.query)
            try:
                width, offset = int(query["width"][0]), int(query["offset"][0])
                if not 32 <= width <= 4096 or width % 8 or offset not in (8, 16, 24):
                    raise ValueError
                pad, modulus = pad_for(width, offset)
            except (KeyError, ValueError):
                return self.send_json({"error": "行宽须为 32 至 4096 之间的 8 的倍数，窗口起点须为 8、16 或 24。"}, 400)
            return self.send_json({"pad": pad, "modulus": modulus})
        if location.path not in ("/", "/index.html", "/style.css", "/app.js"):
            return self.send_error(404)
        return super().do_GET()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--results", type=Path, action="append", help="Directory containing benchmark run folders")
    args.add_argument("--port", type=int, default=8765)
    args.add_argument("--pavia-input", type=Path, help="Path to the original PaviaU.mat")
    opts = args.parse_args()
    output = ROOT / "results"
    roots = [output, *(opts.results or [ROOT.parent / "PADGain_results" / "results"])]
    roots = list(dict.fromkeys(path.resolve() for path in roots))
    jobs = ExperimentJobs(ROOT, output, roots, opts.pavia_input and opts.pavia_input.resolve(), measured)
    with ThreadingHTTPServer(("127.0.0.1", opts.port), partial(Handler, roots=roots, jobs=jobs)) as server:
        print(f"http://127.0.0.1:{opts.port}", flush=True)
        server.serve_forever()

