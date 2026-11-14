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
        cls.db_path = cls.temp_dir / "test_ranking.db"

        def handler_factory(*args, **kwargs):
            return ReplayRequestHandler(
                *args,
                logs_dir=cls.logs_dir,
                db_path=cls.db_path,
                **kwargs,
            )

        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
        cls.port = cls.httpd.server_address[1]
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        mgr = getattr(cls.httpd, "_human_session_manager", None)
        if mgr is not None:
            mgr.close_all()
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

    def _post(self, path, payload):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body_bytes = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body_bytes, headers={"Content-Type": "application/json"})
        res = conn.getresponse()
        body = res.read()
        conn.close()
        return res.status, res.headers, body

    def test_static_index(self):
        status, headers, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Wrongway - Replay Viewer", body)
        self.assertIn(b"ww-tab-btn-history", body)
        self.assertIn(b"ww-history-tab-panel", body)
        self.assertIn(b"ww-bvb-grid", body)
        self.assertNotIn(b"ww-subtab-humanvsbot", body)
        self.assertIn(b"ww-btn-resign", body)

    def test_static_css_and_js(self):
        status, _, body = self._get("/css/style.css")
        self.assertEqual(status, 200)
        self.assertIn(b"--p0-primary", body)
        self.assertIn(b"100vh", body)

        status, _, body = self._get("/js/app.js")
        self.assertEqual(status, 200)
        self.assertIn(b"BoardRenderer", body)
        self.assertIn(b"switchMainTab", body)

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

    def test_api_recursive_leaf_logs(self):
        nested_dir = self.logs_dir / "tournament" / "series"
        nested_dir.mkdir(parents=True)
        try:
            nested_match = nested_dir / "leaf_match.json"
            sample_payload = json.loads(self.sample_match_path.read_text(encoding="utf-8"))
            save_json(nested_match, sample_payload)

            summary_path = self.logs_dir / "tournament" / "summary.json"
            save_json(summary_path, {
                "type": "tournament",
                "agents": ["greedy", "random_agent"],
                "config": {"games_per_pair": 5, "total_matches": 5},
                "standings": [{"rank": 1, "name": "greedy"}],
                "matches": [],
            })

            unrelated_path = self.logs_dir / "tournament" / "notes.json"
            save_json(unrelated_path, {"unrelated": True})

            status, _, body = self._get("/api/logs")
            self.assertEqual(status, 200)
            entries = {item["name"]: item for item in json.loads(body.decode("utf-8"))}

            self.assertIn("tournament/series/leaf_match.json", entries)
            self.assertEqual(entries["tournament/series/leaf_match.json"]["type"], "match")
            self.assertIn("tournament/summary.json", entries)
            self.assertEqual(entries["tournament/summary.json"]["type"], "tournament")
            self.assertNotIn("tournament/notes.json", entries)

            # Fetch nested leaf file via encoded slash and literal slash
            for req_path in (
                "/api/logs/tournament%2Fseries%2Fleaf_match.json",
                "/api/logs/tournament/series/leaf_match.json",
            ):
                s, _, b = self._get(req_path)
                self.assertEqual(s, 200)
                self.assertIn("turns", json.loads(b.decode("utf-8")))

            # Nested traversal still blocked
            s, _, _ = self._get("/api/logs/tournament%2F..%2F..%2Farena.py")
            self.assertEqual(s, 400)
        finally:
            shutil.rmtree(self.logs_dir / "tournament", ignore_errors=True)

    def test_api_leaderboard(self):
        status, _, body = self._get("/api/leaderboard")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIsInstance(data, list)

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

    def test_human_vs_bot_lifecycle_and_stats(self):
        # 1. List available bots
        status, _, body = self._get("/api/human/bots")
        self.assertEqual(status, 200)
        bots = json.loads(body.decode("utf-8"))
        bot_names = {b["name"] for b in bots}
        self.assertIn("greedy", bot_names)
        self.assertIn("optimized_bot", bot_names)

        # 2. Start match as P0 vs greedy
        status, _, body = self._post("/api/human/start", {"bot": "greedy", "human_side": 0, "human_name": "Tester"})
        self.assertEqual(status, 200)
        match = json.loads(body.decode("utf-8"))
        sid = match["session_id"]
        self.assertFalse(match["done"])
        self.assertEqual(match["state"]["current_player"], 0)
        self.assertIn([7, 4], match["legal_moves"])

        # 3. Invalid move should return 400 without forfeiting turn
        status, _, _ = self._post(f"/api/human/match/{sid}/move", {"type": "move", "to": [0, 0]})
        self.assertEqual(status, 400)
        status, _, body = self._get(f"/api/human/match/{sid}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body.decode("utf-8"))["state"]["turn"], 0)

        # 4. Valid move advances turn and triggers bot reply
        status, _, body = self._post(f"/api/human/match/{sid}/move", {"type": "move", "to": [7, 4]})
        self.assertEqual(status, 200)
        after_move = json.loads(body.decode("utf-8"))
        self.assertEqual(after_move["state"]["turn"], 2)
        self.assertEqual(len(after_move["turns"]), 2)

        # 5. Resign records bot win & saves replay log
        status, _, body = self._post(f"/api/human/match/{sid}/resign", {})
        self.assertEqual(status, 200)
        final_data = json.loads(body.decode("utf-8"))
        self.assertTrue(final_data["done"])
        self.assertEqual(final_data["winner"], 1)
        self.assertEqual(final_data["reason"], "resign")
        self.assertTrue((self.logs_dir / final_data["log_file"]).is_file())

        # 6. Check per-bot Human stats
        status, _, body = self._get("/api/human/stats")
        self.assertEqual(status, 200)
        stats_payload = json.loads(body.decode("utf-8"))
        per_bot = {b["bot_name"]: b for b in stats_payload["bots"]}
        self.assertEqual(per_bot["greedy"]["games"], 1)
        self.assertEqual(per_bot["greedy"]["bot_wins"], 1)
        self.assertEqual(per_bot["greedy"]["bot_losses"], 0)
        self.assertEqual(len(stats_payload["recent_matches"]), 1)

        # 7. Starting as P1 triggers bot's Turn 0 automatically
        status, _, body = self._post("/api/human/start", {"bot": "random_agent", "human_side": 1})
        self.assertEqual(status, 200)
        p1_match = json.loads(body.decode("utf-8"))
        self.assertEqual(p1_match["state"]["turn"], 1)
        self.assertEqual(p1_match["state"]["current_player"], 1)
        self._post(f"/api/human/match/{p1_match['session_id']}/resign", {})

    def test_api_leaderboard_agent_h2h_and_matches(self):
        from ranking import RankingSystem

        ranking = RankingSystem(db_path=self.db_path)
        series_path = str(self.logs_dir / "tournament" / "series" / "Alpha_vs_Beta.json")
        ranking.update_match("Alpha", "Beta", winner=0, turns=28, log_path=series_path)
        ranking.update_match("Beta", "Alpha", winner=0, turns=36, log_path=series_path)
        ranking.update_match("Alpha", "Gamma", winner=0, turns=22)

        status, _, body = self._get("/api/leaderboard/Alpha")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        h2h = {row["opponent"]: row for row in data["head_to_head"]}
        self.assertEqual(h2h["Beta"]["wins"], 1)
        self.assertEqual(h2h["Beta"]["losses"], 1)
        self.assertEqual(h2h["Beta"]["log_rel_path"], "tournament/series/Alpha_vs_Beta.json")
        self.assertEqual(h2h["Gamma"]["wins"], 1)
        self.assertEqual(len(data["recent_matches"]), 3)

        status, _, body = self._get("/api/leaderboard/Alpha?opponent=Beta&limit=10")
        self.assertEqual(status, 200)
        filtered = json.loads(body.decode("utf-8"))
        self.assertEqual(len(filtered["recent_matches"]), 2)
        self.assertEqual(
            sorted(m["result"] for m in filtered["recent_matches"]),
            ["LOSS", "WIN"],
        )


if __name__ == "__main__":
    unittest.main()
