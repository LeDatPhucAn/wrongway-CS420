#!/usr/bin/env python3
"""Lightweight local HTTP server for Wrongway Replay Viewer.

Standard library only. Binds to 127.0.0.1.
Serves static viewer files and provides a read-only JSON API for match logs.
"""

import argparse
import http.server
import json
import os
from pathlib import Path
import sys
import threading
import urllib.parse
import webbrowser

ROOT = Path(__file__).resolve().parent
VIEWER_DIR = ROOT / "viewer"
DEFAULT_LOGS_DIR = ROOT / "logs"
DEFAULT_SUBMISSIONS_DIR = ROOT / "submissions"
DEFAULT_DB_PATH = ROOT / "ranking.db"

_DEFAULT_MANAGER_LOCK = threading.Lock()


class ReplayRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, logs_dir=None, submissions_dir=None, db_path=None, session_manager=None, **kwargs):
        self.logs_dir = Path(logs_dir or DEFAULT_LOGS_DIR).resolve()
        self.submissions_dir = Path(submissions_dir or DEFAULT_SUBMISSIONS_DIR).resolve()
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self._explicit_session_manager = session_manager
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def _get_session_manager(self):
        if self._explicit_session_manager is not None:
            return self._explicit_session_manager
        with _DEFAULT_MANAGER_LOCK:
            mgr = getattr(self.server, "_human_session_manager", None)
            if mgr is None:
                from human_match import HumanSessionManager
                mgr = HumanSessionManager(
                    submissions_dir=self.submissions_dir,
                    logs_dir=self.logs_dir,
                    db_path=self.db_path,
                )
                self.server._human_session_manager = mgr
            return mgr

    def end_headers(self):
        # Enable CORS for local development and prevent stale caching of logs
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/logs":
            self._handle_api_logs()
        elif path.startswith("/api/logs/"):
            filename = urllib.parse.unquote(path[len("/api/logs/"):])
            self._handle_api_get_log(filename)
        elif path == "/api/leaderboard":
            self._handle_api_leaderboard()
        elif path.startswith("/api/leaderboard/"):
            agent_name = urllib.parse.unquote(path[len("/api/leaderboard/"):])
            self._handle_api_get_agent(agent_name, query_str=parsed.query)
        elif path == "/api/human/bots":
            self._handle_api_human_bots()
        elif path == "/api/human/stats":
            self._handle_api_human_stats()
        elif path.startswith("/api/human/stats/"):
            bot_name = urllib.parse.unquote(path[len("/api/human/stats/"):])
            self._handle_api_human_bot_history(bot_name)
        elif path.startswith("/api/human/match/"):
            session_id = urllib.parse.unquote(path[len("/api/human/match/"):]).strip("/")
            self._handle_api_human_get_match(session_id)
        else:
            # Default to index.html for root path
            if path in ("", "/"):
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/human/start":
            self._handle_api_human_start()
        elif path == "/api/human/stats/reset":
            self._handle_api_human_reset_stats()
        elif path.startswith("/api/human/match/"):
            rest = urllib.parse.unquote(path[len("/api/human/match/"):]).strip("/")
            parts = rest.split("/")
            if len(parts) == 2 and parts[1] == "move":
                self._handle_api_human_move(parts[0])
            elif len(parts) == 2 and parts[1] == "resign":
                self._handle_api_human_resign(parts[0])
            else:
                self._send_json_error(404, "Unknown human match endpoint")
        else:
            self._send_json_error(404, "Endpoint not found")

    def _handle_api_logs(self):
        """Return summary list of all JSON logs recursively under the logs directory."""
        if not self.logs_dir.exists():
            self._send_json([])
            return

        logs = []
        for file_path in self.logs_dir.rglob("*.json"):
            if not file_path.is_file():
                continue
            try:
                resolved = file_path.resolve()
                resolved.relative_to(self.logs_dir)
                rel_name = resolved.relative_to(self.logs_dir).as_posix()
                stat = resolved.stat()
                with open(resolved, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    continue

                if data.get("type") == "series" and isinstance(data.get("matches"), list):
                    agents = data.get("agents", [])
                    p1 = agents[0] if len(agents) > 0 else "Unknown"
                    p2 = agents[1] if len(agents) > 1 else "Unknown"
                    winner = "Series"
                    total_turns = len(data.get("matches", []))
                    log_type = "series"
                elif data.get("type") == "tournament" and isinstance(data.get("standings"), list):
                    agents = data.get("agents", [])
                    standings = data.get("standings", [])
                    cfg = data.get("config", {})
                    p1 = f"Tournament ({len(agents)} bots)"
                    p2 = f"{cfg.get('games_per_pair', '?')} games/pair"
                    winner = standings[0].get("name") if standings else None
                    total_turns = cfg.get("total_matches", len(data.get("matches", [])))
                    log_type = "tournament"
                elif "initial_state" in data and "turns" in data:
                    players = data.get("players", [])
                    p1 = players[0]["name"] if len(players) > 0 else "Player 0"
                    p2 = players[1]["name"] if len(players) > 1 else "Player 1"
                    winner = data.get("winner_name")
                    total_turns = data.get("total_turns", len(data.get("turns", [])))
                    log_type = "match"
                else:
                    continue

                logs.append({
                    "name": rel_name,
                    "type": log_type,
                    "player1": p1,
                    "player2": p2,
                    "winner": winner,
                    "total_turns": total_turns,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size
                })
            except Exception:
                continue

        # Sort newest first
        logs.sort(key=lambda x: x["mtime"], reverse=True)
        self._send_json(logs)

    def _handle_api_get_log(self, filename):
        """Safely fetch and return a specific JSON log file."""
        if not filename or Path(filename).is_absolute() or Path(filename).drive:
            self.send_error(400, "Bad path")
            return
        target_path = (self.logs_dir / filename).resolve()
        try:
            # Verify no directory traversal
            target_path.relative_to(self.logs_dir)
        except ValueError:
            self.send_error(400, "Bad path")
            return

        if not target_path.is_file() or target_path.suffix.lower() != ".json":
            self.send_error(404, "Log file not found")
            return

        try:
            content = target_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading log: {e}")

    def _handle_api_leaderboard(self):
        """Return the current TrueSkill leaderboard from ranking.db."""
        try:
            from ranking import RankingSystem
            ranking = RankingSystem(db_path=self.db_path)
            self._send_json(ranking.get_leaderboard())
        except Exception as e:
            self.send_error(500, f"Error reading leaderboard: {e}")

    def _handle_api_get_agent(self, agent_name, query_str=""):
        """Return rating profile, per-opponent head-to-head stats, and bot-vs-bot matches for an agent."""
        try:
            from ranking import RankingSystem
            params = urllib.parse.parse_qs(query_str)
            opponent = params.get("opponent", [None])[0] or None
            limit_raw = params.get("limit", ["200"])[0]
            try:
                limit = max(1, min(1000, int(limit_raw)))
            except (TypeError, ValueError):
                limit = 200
            ranking = RankingSystem(db_path=self.db_path)
            self._send_json(
                ranking.get_agent_history(
                    agent_name,
                    limit=limit,
                    opponent=opponent,
                    logs_dir=self.logs_dir,
                )
            )
        except Exception as e:
            self.send_error(500, f"Error reading agent history: {e}")

    def _handle_api_human_bots(self):
        """List all submission bots with Human vs Bot W/L records and AI rankings."""
        try:
            from human_match import list_available_bots
            bots = list_available_bots(
                submissions_dir=self.submissions_dir,
                db_path=self.db_path,
            )
            self._send_json(bots)
        except Exception as e:
            self._send_json_error(500, f"Error listing bots: {e}")

    def _handle_api_human_stats(self):
        """Return per-bot Human vs Bot win/loss stats and recent match history."""
        try:
            from ranking import RankingSystem
            from tournament import discover_agents
            try:
                bot_names = sorted(p.stem for p in discover_agents(submissions_dir=self.submissions_dir))
            except ValueError:
                bot_names = []
            ranking = RankingSystem(db_path=self.db_path)
            per_bot = ranking.get_human_bot_stats(bot_names=bot_names)
            recent = ranking.get_human_match_history(limit=50)
            self._send_json({
                "bots": per_bot,
                "recent_matches": recent,
            })
        except Exception as e:
            self._send_json_error(500, f"Error loading human match stats: {e}")

    def _handle_api_human_bot_history(self, bot_name):
        """Return Human vs Bot W/L stats and recent matches for a specific bot."""
        try:
            from ranking import RankingSystem
            ranking = RankingSystem(db_path=self.db_path)
            stats = ranking.get_human_bot_stats(bot_names=[bot_name])[0]
            matches = ranking.get_human_match_history(bot_name=bot_name, limit=50)
            self._send_json({
                "bot": stats,
                "recent_matches": matches,
            })
        except Exception as e:
            self._send_json_error(500, f"Error loading bot human stats: {e}")

    def _handle_api_human_get_match(self, session_id):
        try:
            mgr = self._get_session_manager()
            self._send_json(mgr.get_session_state(session_id))
        except KeyError as e:
            self._send_json_error(404, str(e))
        except Exception as e:
            self._send_json_error(500, f"Error fetching session: {e}")

    def _handle_api_human_start(self):
        try:
            body = self._read_json_body()
            bot = body.get("bot")
            if not bot:
                self._send_json_error(400, "Missing required field 'bot'")
                return
            human_side = int(body.get("human_side", 0))
            human_name = str(body.get("human_name") or "Human")
            timeout = float(body.get("timeout", 1.0))
            max_turns = int(body.get("max_turns", 500))
            seed = int(body.get("seed", 0))
            mgr = self._get_session_manager()
            payload = mgr.start_session(
                bot_name=bot,
                human_side=human_side,
                human_name=human_name,
                timeout=timeout,
                max_turns=max_turns,
                seed=seed,
            )
            self._send_json(payload)
        except ValueError as e:
            self._send_json_error(400, str(e))
        except Exception as e:
            self._send_json_error(500, f"Failed to start human match: {e}")

    def _handle_api_human_move(self, session_id):
        try:
            action = self._read_json_body()
            mgr = self._get_session_manager()
            payload = mgr.play_move(session_id, action)
            self._send_json(payload)
        except KeyError as e:
            self._send_json_error(404, str(e))
        except ValueError as e:
            self._send_json_error(400, str(e))
        except Exception as e:
            self._send_json_error(500, f"Error applying move: {e}")

    def _handle_api_human_resign(self, session_id):
        try:
            mgr = self._get_session_manager()
            payload = mgr.resign_session(session_id)
            self._send_json(payload)
        except KeyError as e:
            self._send_json_error(404, str(e))
        except Exception as e:
            self._send_json_error(500, f"Error resigning session: {e}")

    def _handle_api_human_reset_stats(self):
        try:
            from ranking import RankingSystem
            ranking = RankingSystem(db_path=self.db_path)
            ranking.reset_human_matches()
            self._send_json({"ok": True})
        except Exception as e:
            self._send_json_error(500, f"Error resetting human stats: {e}")

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        if length > 1_048_576:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _send_json_error(self, status_code, message):
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ReplayHTTPServer(http.server.ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind behind an active one
    # without receiving traffic.
    allow_reuse_address = os.name != "nt"


def run_server(port=8420, logs_dir=None, open_browser=False, submissions_dir=None, db_path=None):
    from human_match import HumanSessionManager

    logs_dir = Path(logs_dir or DEFAULT_LOGS_DIR).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = Path(submissions_dir or DEFAULT_SUBMISSIONS_DIR).resolve()
    db_path = Path(db_path or DEFAULT_DB_PATH).resolve()

    session_manager = HumanSessionManager(
        submissions_dir=submissions_dir,
        logs_dir=logs_dir,
        db_path=db_path,
    )

    def handler_factory(*args, **kwargs):
        return ReplayRequestHandler(
            *args,
            logs_dir=logs_dir,
            submissions_dir=submissions_dir,
            db_path=db_path,
            session_manager=session_manager,
            **kwargs,
        )

    server_address = ("127.0.0.1", port)
    try:
        httpd = _ReplayHTTPServer(server_address, handler_factory)
    except OSError as err:
        session_manager.close_all()
        print(f"Failed to bind to 127.0.0.1:{port}: {err}", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{port}/"
    print(f"Wrongway Viewer running at: {url}")
    print(f"Serving logs from: {logs_dir}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        session_manager.close_all()
        httpd.server_close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8420, help="Port to bind (default: 8420)")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS_DIR, help="Logs directory")
    parser.add_argument("--open", action="store_true", help="Open browser on start")
    args = parser.parse_args(argv)

    return run_server(port=args.port, logs_dir=args.logs, open_browser=args.open)


if __name__ == "__main__":
    sys.exit(main())
