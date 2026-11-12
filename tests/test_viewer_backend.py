#!/usr/bin/env python3
"""Automated tests for Wrongway Viewer backend and series runner."""

import http.client
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parent.parent

from environment import Environment
from serve import ReplayRequestHandler
from series import run_series
from arena import run_match, save_json
import http.server


class TestViewerBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="ww_test_"))
        cls.logs_dir = cls.temp_dir / "logs"
        cls.logs_dir.mkdir(parents=True)

        # Generate a sample match fixture
        res = run_match(
            ROOT / "submissions" / "greedy.py",
            ROOT / "submissions" / "random_agent.py",
            max_turns=20,
            seed=42
        )
        cls.sample_match_path = cls.logs_dir / "sample_match.json"
        save_json(cls.sample_match_path, res)

        # Start a test ThreadingHTTPServer on an ephemeral port
        def handler_factory(*args, **kwargs):
            return ReplayRequestHandler(*args, logs_dir=cls.logs_dir, **kwargs)

        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
        cls.port = cls.httpd.server_address[1]
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        res = conn.getresponse()
        body = res.read()
        conn.close()
        return res.status, res.headers, body

    def test_static_index(self):
        status, headers, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Wrongway - Replay Viewer", body)

    def test_static_css_and_js(self):
        status, _, body = self._get("/css/style.css")
        self.assertEqual(status, 200)
        self.assertIn(b"--p0-primary", body)

        status, _, body = self._get("/js/app.js")
        self.assertEqual(status, 200)
        self.assertIn(b"BoardRenderer", body)

    def test_api_logs_listing(self):
        status, headers, body = self._get("/api/logs")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)

        first = data[0]
        self.assertEqual(first["name"], "sample_match.json")
        self.assertEqual(first["type"], "match")
        self.assertIn("player1", first)
        self.assertIn("player2", first)
        self.assertIn("total_turns", first)
        self.assertIn("mtime", first)

    def test_api_get_specific_log(self):
        status, _, body = self._get("/api/logs/sample_match.json")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data.get("format_version"), 1)
        self.assertIn("initial_state", data)
        self.assertIn("turns", data)

    def test_api_get_nonexistent_log(self):
        status, _, _ = self._get("/api/logs/does_not_exist.json")
        self.assertEqual(status, 404)

    def test_api_directory_traversal_prevention(self):
        # Attempt traversal outside logs dir
        status, _, _ = self._get("/api/logs/..%2f..%2farena.py")
        self.assertIn(status, (400, 404))

    def test_series_runner(self):
        out_series = self.logs_dir / "series_test.json"
        data = run_series(
            ROOT / "submissions" / "greedy.py",
            ROOT / "submissions" / "wall_builder.py",
            games=2,
            swap_sides=True,
            seed=10,
            max_turns=30,
            out_path=out_series
        )
        self.assertEqual(data["type"], "series")
        self.assertEqual(data["format_version"], 1)
        self.assertEqual(len(data["matches"]), 2)
        self.assertIn("summary", data)
        self.assertIn("side_wins", data["summary"])
        self.assertTrue(out_series.is_file())

    def test_bfs_distance_consistency(self):
        env = Environment()
        d0 = env.distance_map(0)
        d1 = env.distance_map(1)
        # P0 starts at [8, 4] with goal row 0 -> shortest open distance is 8
        self.assertEqual(d0[8][4], 8)
        # P1 starts at [0, 4] with goal row 8 -> shortest open distance is 8
        self.assertEqual(d1[0][4], 8)


if __name__ == "__main__":
    unittest.main()
