from functools import partial
import http.client
from http.server import ThreadingHTTPServer
import json
from threading import Thread
import unittest

from padding_your_kernel.live import BusyError
from padding_your_kernel.presets import analyze_preset, list_presets
from padding_your_kernel.server import Handler


class StubJobs:
    def __init__(self):
        self.active = None

    def capabilities(self):
        return {"available": True, "active": self.active}

    def start(self, preset):
        if preset not in {item.name for item in list_presets()}:
            raise ValueError("Unknown preset")
        if self.active:
            raise BusyError("A run is already in progress")
        self.active = {"id": "run-1", "preset": preset, "status": "running", "stage": "preflight",
                       "message": "Checking configuration"}
        return self.active

    def get(self, job_id):
        return self.active if self.active and job_id == self.active["id"] else None


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = StubJobs()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, jobs=cls.jobs))
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.jobs.active = None

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    def report(self, body, path="/api/analyze"):
        status, content_type, payload = self.request("POST", path, json.dumps(body),
                                                     {"Content-Type": "application/json"})
        self.assertEqual(content_type, "application/json; charset=utf-8")
        return status, json.loads(payload)

    def test_workbench_assets(self):
        for path, expected in (("/", "text/html"), ("/style.css", "text/css"), ("/app.js", "text/javascript")):
            with self.subTest(path=path):
                status, content_type, payload = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertTrue(content_type.startswith(expected))
                self.assertTrue(payload)

    def test_presets_and_independent_analysis(self):
        status, _, payload = self.request("GET", "/api/presets")
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(payload)["presets"]), 18)
        status, result = self.report({"mode": "preset", "preset": "syrk"})
        self.assertEqual((status, result["status"], result["p"]), (200, "recommend_padding", 8))
        status, result = self.report({"mode": "adjacent-columns", "width": 2048, "rows": 512})
        self.assertEqual((status, result["p"]), (200, 16))

    def test_type_and_shape_validation(self):
        for body in ([], {"mode": "bad"}, {"mode": "preset", "preset": "unknown"},
                     {"mode": "adjacent-columns", "width": True, "rows": 512},
                     {"mode": "adjacent-columns", "width": 2048, "rows": 2**31},
                     {"mode": "adjacent-columns", "width": 2048, "rows": 512, "other": 2}):
            with self.subTest(body=body):
                status, result = self.report(body)
                self.assertEqual(status, 400)
                self.assertTrue(result["error"])

    def test_published_measurements_match_presets(self):
        status, _, payload = self.request("GET", "/api/measurements")
        self.assertEqual(status, 200)
        snapshot = json.loads(payload)
        self.assertEqual((snapshot["preparation"], snapshot["processes"], snapshot["correctness_checks"]),
                         ("primed", 3, 141))
        rows = snapshot["measurements"]
        by_preset = {item["preset"]: item for item in rows}
        self.assertEqual(len(rows), len(by_preset))
        self.assertEqual(set(by_preset), {item.name for item in list_presets()})
        for name, item in by_preset.items():
            self.assertEqual(item["p"], analyze_preset(name).p)
            self.assertGreater(item["baseline_us"], 0)
            if item["p"]:
                self.assertEqual(len(item["process_speedups"]), 3)
            else:
                self.assertEqual(item["status"], "baseline_only")
                self.assertIsNone(item["speedup"])
        self.assertEqual(by_preset["mvt"]["case"], "n2048")
        self.assertEqual(by_preset["mvt-4096"]["case"], "n4096")
        self.assertAlmostEqual(by_preset["syrk"]["speedup"], 2.5727191304425374)
        self.assertEqual(by_preset["gaussian"]["status"], "not_accepted")

    def test_live_request_and_job_polling(self):
        path = "/api/live"
        status, _, payload = self.request("GET", path)
        self.assertEqual((status, json.loads(payload)["active"]), (200, None))
        self.assertEqual(self.report({"preset": "unknown"}, path)[0], 400)
        self.assertEqual(self.report({"preset": "syrk", "command": "arbitrary"}, path)[0], 400)
        self.assertEqual(self.report({"mode": "adjacent-columns", "width": 2048}, path)[0], 400)
        status, job = self.report({"preset": "syrk"}, path)
        self.assertEqual((status, job["preset"], job["status"]), (202, "syrk", "running"))
        self.assertEqual(self.report({"preset": "gemm"}, path)[0], 409)
        status, _, payload = self.request("GET", f"{path}/{job['id']}")
        self.assertEqual((status, json.loads(payload)), (200, job))
        self.assertEqual(self.request("GET", f"{path}/missing")[0], 404)

    def test_server_scope(self):
        status, _, _ = self.request("GET", "/api/presets", headers={"Host": "example.org"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("GET", "/api/presets", headers={"Origin": "http://example.org"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("GET", "/not-served")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
