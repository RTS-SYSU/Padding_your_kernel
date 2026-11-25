"""A loopback-only Web interface for the same functions exposed by the CLI."""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from urllib.parse import urlsplit

from .model import analyze_adjacent_columns
from .presets import analyze_preset, list_presets


ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
BODY_LIMIT = 4096


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
        elif path in ASSETS:
            filename, content_type = ASSETS[path]
            resource = files("padding_your_kernel.web").joinpath(filename)
            self.respond(resource.read_bytes(), content_type)
        else:
            self.respond_json({"error": "页面不存在。"}, HTTPStatus.NOT_FOUND)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        if not self.local_request():
            return
        if urlsplit(self.path).path != "/api/analyze":
            self.respond_json({"error": "接口不存在。"}, HTTPStatus.NOT_FOUND)
            return
        if self.headers.get_content_type() != "application/json":
            self.respond_json({"error": "请求需使用 JSON。"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > BODY_LIMIT:
                self.respond_json({"error": "分析参数过长。"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            if length <= 0:
                raise ValueError("请求缺少分析参数。")
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("分析请求不完整。")
            body = json.loads(payload)
            report = analyze_request(body)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            self.respond_json({"error": "分析参数格式无效。"}, HTTPStatus.BAD_REQUEST)
        except TimeoutError:
            self.respond_json({"error": "分析请求已超时。"}, HTTPStatus.REQUEST_TIMEOUT)
        except (KeyError, ValueError) as error:
            self.respond_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        else:
            return self.respond_json(report)


def serve(port):
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"http://127.0.0.1:{httpd.server_port}/", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
