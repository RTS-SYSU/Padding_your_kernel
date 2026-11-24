"""Run one allowed benchmark at a time and keep its frozen plan and logs."""
from datetime import datetime
import json
import os
import shutil
import subprocess
import sys
from threading import Lock, Thread
from uuid import uuid4


class BusyError(Exception):
    pass


class ExperimentJobs:
    def __init__(self, repo, output, input_roots, pavia, parse_result):
        self.repo = repo
        self.output = output
        self.input_roots = input_roots
        self.pavia = pavia
        self.parse_result = parse_result
        self.lock = Lock()
        self.job = None

    def snapshot(self):
        with self.lock:
            return dict(self.job) if self.job else None

    def start(self, kind, case):
        with self.lock:
            if self.job and self.job["status"] == "running":
                raise BusyError("已有实验在使用 GPU。")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run = f"live_{timestamp}_{kind}_{uuid4().hex[:6]}"
            self.job = dict(run=run, kind=kind, case=case,
                            status="running", message="正在准备实验")
            worker = Thread(target=self._execute, args=(run, kind, case), daemon=True)
            worker.start()
            return dict(self.job)

    def _update(self, **fields):
        with self.lock:
            self.job.update(fields)

    def _pavia_input(self):
        if self.pavia:
            return self.pavia
        for root in self.input_roots:
            for pattern in ("PaviaU.mat", "pavia*/PaviaU.mat", "live_*_pavia_*/PaviaU.mat"):
                for path in root.glob(pattern):
                    if path.is_file():
                        return path
        raise FileNotFoundError("未找到 PaviaU.mat。启动服务时用 --pavia-input 指定原始影像。")

    def _command(self, command, folder, name, message):
        self._update(message=message)
        logfile = folder / (name + ".log")
        errorfile = folder / (name + ".stderr.log")
        with logfile.open("x", encoding="utf-8") as log, errorfile.open("xb") as errors:
            with subprocess.Popen(
                list(map(str, command)), cwd=self.repo, stdout=subprocess.PIPE, stderr=errors,
                text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUTF8": "1"},
            ) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("PROGRESS "):
                        self._update(message=line.removeprefix("PROGRESS ").strip())
                status = process.wait()
        if status:
            raw = errorfile.read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                raw = logfile.read_text(encoding="utf-8", errors="replace").strip()
            tail = raw.splitlines()[-4:]
            raise RuntimeError("；".join(tail)[-1000:] or f"{name} 退出码 {status}")

    def _execute(self, run, kind, case):
        try:
            out = self.output / run
            out.mkdir(parents=True, exist_ok=False)
            (out / "input.json").write_text(json.dumps(dict(kind=kind, case=case), indent=2) + "\n", encoding="utf-8")
            live = self.repo / "experiments" / ("dual_window" if kind == "dual" else "pavia")
            if kind == "pavia":
                source = self._pavia_input()
                shutil.copyfile(source, out / "PaviaU.mat")
                provenance = source.with_name("source-url.txt")
                (out / "source-url.txt").write_text(
                    provenance.read_text(encoding="utf-8") if provenance.is_file() else source.name,
                    encoding="utf-8",
                )
            shell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
            if not shell:
                raise RuntimeError("CUDA 构建需要 PowerShell。")
            self._command([shell, "-NoProfile", "-File", self.repo / "tools" / "build.ps1", "-SourceFile",
                           live / "bench.cu", "-BuildDir", out / "build"], out, "build", "正在编译 CUDA kernel")
            bench = out / "build" / "bench.exe"
            runner = live / ("current_theory.py" if kind == "dual" else "run.py")
            args = [bench, out] if kind == "dual" else [out]
            self._command([sys.executable, "-B", runner, "freeze", *args, "--case", case],
                          out, "freeze", "正在冻结补量与事前计数")
            plan = json.loads((out / "plan.json").read_bytes())
            self._command([sys.executable, "-B", runner, "measure", *args],
                          out, "measure", "正在运行配对计时")
            result = self.parse_result(out, plan)
            (out / ".complete").touch(exist_ok=False)
            self._update(status="done", message="本次实验已完成", result=result)
        except Exception as error:
            self._update(status="error", message=str(error))

