"""A loopback-only Web interface for the same functions exposed by the CLI."""

from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
from urllib.parse import urlsplit

from .live import BusyError, LiveJobs
from .model import analyze_adjacent_columns
from .presets import analyze_preset, list_presets


ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
BODY_LIMIT = 4096
RESEARCH_NAME = "Analytical GPU Padding Benefit Model"
RUNNER_PATH = Path("experiment_23_suite_single_pad/run_experiment.py")


class RequestError(ValueError):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


def analyze_request(body):
    if not isinstance(body, dict):
        raise ValueError("分析参数必须是 JSON 对象。")
    if body.get("mode") == "preset":
        if set(body) != {"mode", "preset"} or not isinstance(body["preset"], str):
            raise ValueError("请选择一个算子预设。")
        return analyze_preset(body["preset"]).to_dict()
    if body.get("mode") == "adjacent-columns":
        mandatory = {"mode", "width", "rows"}
        optional = {"reserved_lines", "base_alignment_bytes"}
        if not mandatory <= body.keys() <= mandatory | optional:
            raise ValueError("请输入逻辑行宽、行数和有效的可选参数。")
        fields = {key: value for key, value in body.items() if key != "mode"}
        if any(type(value) is not int for value in fields.values()):
            raise ValueError("分析参数必须是整数。")
        return analyze_adjacent_columns(**fields).to_dict()
    raise ValueError("请选择预设或自定义相邻列复用。")


class Handler(BaseHTTPRequestHandler):
    server_version = "PaddingYourKernel"

    def __init__(self, *args, jobs=None, **kwargs):
        self.jobs = jobs
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def local_request(self):
        port = self.server.server_port
        authorities = {f"localhost:{port}", f"127.0.0.1:{port}"}
        if port == 80:
            authorities.update(("localhost", "127.0.0.1"))
        if self.headers.get("Host", "").lower() not in authorities:
            self.respond_json({"error": "请从本机工作台访问。"}, HTTPStatus.FORBIDDEN)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{host}" for host in authorities}:
            self.respond_json({"error": "请从本机工作台发起分析。"}, HTTPStatus.FORBIDDEN)
            return False
        return True

    def respond(self, body, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
                         "style-src 'self' 'unsafe-inline'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def respond_json(self, data, status=HTTPStatus.OK):
        self.respond(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json", status)

    def do_GET(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/presets":
            self.respond_json({"presets": [item.to_dict() for item in list_presets()]})
        elif path == "/api/measurements":
            resource = files("padding_your_kernel.data").joinpath("experiment23_primed.json")
            try:
                payload = resource.read_bytes()
            except OSError:
                self.respond_json({"error": "随工具发布的真机记录无法读取。"}, HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                self.respond(payload, "application/json")
        elif path == "/api/live":
            availability = (self.jobs.capabilities() if self.jobs
                            else {"available": False, "reason": "现场运行未配置。", "active": None})
            self.respond_json(availability)
        elif path.startswith("/api/live/"):
            job = self.jobs.get(path.removeprefix("/api/live/")) if self.jobs else None
            if job is None:
                self.respond_json({"error": "找不到该运行记录。"}, HTTPStatus.NOT_FOUND)
            else:
                self.respond_json(job)
        elif path in ASSETS:
            filename, content_type = ASSETS[path]
            resource = files("padding_your_kernel.web").joinpath(filename)
            self.respond(resource.read_bytes(), content_type)
        else:
            self.respond_json({"error": "页面不存在。"}, HTTPStatus.NOT_FOUND)

    def do_HEAD(self):
        self.do_GET()

    def request_json(self):
        if self.headers.get_content_type() != "application/json":
            raise RequestError("请求需使用 JSON。", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        length = int(self.headers.get("Content-Length", "0"))
        if length > BODY_LIMIT:
            raise RequestError("分析参数过长。", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if length <= 0:
            raise ValueError("请求缺少分析参数。")
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("分析请求不完整。")
        return json.loads(payload)

    def start_measurement(self, body):
        if not isinstance(body, dict) or set(body) != {"preset"} or not isinstance(body["preset"], str):
            raise ValueError("请选择一个算子预设以运行真机实验。")
        if self.jobs is None:
            raise RequestError("现场运行未配置。", HTTPStatus.SERVICE_UNAVAILABLE)
        try:
            return self.jobs.start(body["preset"])
        except BusyError as error:
            raise RequestError(str(error), HTTPStatus.CONFLICT) from error
        except RuntimeError as error:
            raise RequestError(str(error), HTTPStatus.SERVICE_UNAVAILABLE) from error

    def do_POST(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path not in ("/api/analyze", "/api/live"):
            self.respond_json({"error": "接口不存在。"}, HTTPStatus.NOT_FOUND)
            return
        try:
            body = self.request_json()
            report = self.start_measurement(body) if path == "/api/live" else analyze_request(body)
        except RequestError as error:
            self.respond_json({"error": str(error)}, error.status)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            self.respond_json({"error": "分析参数格式无效。"}, HTTPStatus.BAD_REQUEST)
        except TimeoutError:
            self.respond_json({"error": "分析请求已超时。"}, HTTPStatus.REQUEST_TIMEOUT)
        except (KeyError, ValueError) as error:
            self.respond_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        else:
            status = HTTPStatus.ACCEPTED if path == "/api/live" else HTTPStatus.OK
            self.respond_json(report, status)


def detect_research_dir():
    here = Path.cwd()
    return next((root for root in (here, here.parent / RESEARCH_NAME)
                 if (root / RUNNER_PATH).is_file()), None)


def serve(port, *, research_dir=None, results_dir=Path("results/live")):
    root = research_dir.resolve() if research_dir else detect_research_dir()
    jobs = LiveJobs(root, results_dir.resolve())
    handler = partial(Handler, jobs=jobs)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"http://127.0.0.1:{httpd.server_port}/", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
