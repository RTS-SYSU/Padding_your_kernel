import http.client
from http.server import ThreadingHTTPServer
import json
from threading import Thread
import unittest

from padding_your_kernel.server import Handler


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    def report(self, body):
        status, content_type, payload = self.request("POST", "/api/analyze", json.dumps(body),
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

    def test_server_scope(self):
        status, _, _ = self.request("GET", "/api/presets", headers={"Host": "example.org"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("GET", "/api/presets", headers={"Origin": "http://example.org"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("GET", "/not-served")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
