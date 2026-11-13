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
import urllib.parse
import webbrowser

ROOT = Path(__file__).resolve().parent
VIEWER_DIR = ROOT / "viewer"
DEFAULT_LOGS_DIR = ROOT / "logs"


class ReplayRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, logs_dir=None, **kwargs):
        self.logs_dir = Path(logs_dir or DEFAULT_LOGS_DIR).resolve()
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def end_headers(self):
        # Enable CORS for local development and prevent stale caching of logs
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

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
            self._handle_api_get_agent(agent_name)
        else:
            # Default to index.html for root path
            if path in ("", "/"):
                self.path = "/index.html"
            super().do_GET()

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
            ranking = RankingSystem()
            self._send_json(ranking.get_leaderboard())
        except Exception as e:
            self.send_error(500, f"Error reading leaderboard: {e}")

    def _handle_api_get_agent(self, agent_name):
        """Return rating profile and recent matches for an agent."""
        try:
            from ranking import RankingSystem
            ranking = RankingSystem()
            self._send_json(ranking.get_agent_history(agent_name))
        except Exception as e:
            self.send_error(500, f"Error reading agent history: {e}")

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


def run_server(port=8420, logs_dir=None, open_browser=False):
    logs_dir = Path(logs_dir or DEFAULT_LOGS_DIR).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    def handler_factory(*args, **kwargs):
        return ReplayRequestHandler(*args, logs_dir=logs_dir, **kwargs)

    server_address = ("127.0.0.1", port)
    try:
        httpd = _ReplayHTTPServer(server_address, handler_factory)
    except OSError as err:
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
