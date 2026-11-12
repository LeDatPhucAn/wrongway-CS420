#!/usr/bin/env python3
"""Incremental TrueSkill-based rating and ranking system for Wrongway.

Inspired by bombIT (Bomberland).
Each time a match is played, ratings are updated incrementally without
re-reading or recounting historical match logs.
Stores agent ratings and match histories in a lightweight SQLite database.
Zero external dependencies (uses standard library statistics.NormalDist and sqlite3).
"""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
from statistics import NormalDist
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "ranking.db"

# TrueSkill parameters matching bombIT conventions
DEFAULT_MU = 100.0
DEFAULT_SIGMA = 100.0 / 3.0  # ~33.333
DEFAULT_BETA = DEFAULT_SIGMA / 2.0  # ~16.667
DRAW_PROBABILITY = 0.1

_NORMAL = NormalDist(0.0, 1.0)
_DRAW_MARGIN = _NORMAL.inv_cdf((1.0 + DRAW_PROBABILITY) / 2.0) * math.sqrt(2.0) * DEFAULT_BETA


def score(mu: float, sigma: float) -> float:
    """Conservative skill estimate (μ - 3σ) matching bombIT ranking standard."""
    return float(mu) - 3.0 * float(sigma)


def rate_2p(mu1: float, sig1: float, mu2: float, sig2: float, outcome: int) -> Tuple[float, float, float, float]:
    """Calculate 2-player TrueSkill rating updates analytically.

    outcome:
      1 -> Player 1 won, Player 2 lost
      2 -> Player 2 won, Player 1 lost
      0 -> Draw
    Returns: (new_mu1, new_sig1, new_mu2, new_sig2)
    """
    c = math.sqrt(2.0 * (DEFAULT_BETA ** 2) + (sig1 ** 2) + (sig2 ** 2))

    if outcome == 1:
        # Player 1 won
        d = (mu1 - mu2 - _DRAW_MARGIN) / c
        v = _NORMAL.pdf(d) / max(1e-12, _NORMAL.cdf(d))
        w = v * (v + d)

        new_mu1 = mu1 + ((sig1 ** 2) / c) * v
        new_mu2 = mu2 - ((sig2 ** 2) / c) * v
        new_sig1 = math.sqrt(max(1e-4, (sig1 ** 2) * (1.0 - ((sig1 ** 2) / (c ** 2)) * w)))
        new_sig2 = math.sqrt(max(1e-4, (sig2 ** 2) * (1.0 - ((sig2 ** 2) / (c ** 2)) * w)))
        return new_mu1, new_sig1, new_mu2, new_sig2

    elif outcome == 2:
        # Player 2 won (symmetric)
        n_m2, n_s2, n_m1, n_s1 = rate_2p(mu2, sig2, mu1, sig1, outcome=1)
        return n_m1, n_s1, n_m2, n_s2

    else:
        # Draw: truncate difference inside [-draw_margin, +draw_margin]
        d1 = mu1 - mu2
        a1, b1 = (-_DRAW_MARGIN - d1) / c, (_DRAW_MARGIN - d1) / c
        denom1 = max(1e-12, _NORMAL.cdf(b1) - _NORMAL.cdf(a1))
        v1 = (_NORMAL.pdf(a1) - _NORMAL.pdf(b1)) / denom1
        w1 = (v1 ** 2) - ((a1 * _NORMAL.pdf(a1) - b1 * _NORMAL.pdf(b1)) / denom1)

        new_mu1 = mu1 + ((sig1 ** 2) / c) * v1
        new_sig1 = math.sqrt(max(1e-4, (sig1 ** 2) * (1.0 - ((sig1 ** 2) / (c ** 2)) * w1)))

        d2 = mu2 - mu1
        a2, b2 = (-_DRAW_MARGIN - d2) / c, (_DRAW_MARGIN - d2) / c
        denom2 = max(1e-12, _NORMAL.cdf(b2) - _NORMAL.cdf(a2))
        v2 = (_NORMAL.pdf(a2) - _NORMAL.pdf(b2)) / denom2
        w2 = (v2 ** 2) - ((a2 * _NORMAL.pdf(a2) - b2 * _NORMAL.pdf(b2)) / denom2)

        new_mu2 = mu2 + ((sig2 ** 2) / c) * v2
        new_sig2 = math.sqrt(max(1e-4, (sig2 ** 2) * (1.0 - ((sig2 ** 2) / (c ** 2)) * w2)))
        return new_mu1, new_sig1, new_mu2, new_sig2


class RankingSystem:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS agents (
                    name TEXT PRIMARY KEY,
                    mu REAL NOT NULL DEFAULT 100.0,
                    sigma REAL NOT NULL DEFAULT 33.333333333333336,
                    score REAL NOT NULL DEFAULT 0.0,
                    n_games INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    total_turns INTEGER NOT NULL DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    player0 TEXT NOT NULL,
                    player1 TEXT NOT NULL,
                    winner INTEGER,
                    winner_name TEXT,
                    turns INTEGER NOT NULL,
                    seed INTEGER,
                    p0_mu_before REAL,
                    p0_sigma_before REAL,
                    p0_mu_after REAL,
                    p0_sigma_after REAL,
                    p1_mu_before REAL,
                    p1_sigma_before REAL,
                    p1_mu_after REAL,
                    p1_sigma_after REAL,
                    log_path TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_agents_score ON agents(score DESC);
                CREATE INDEX IF NOT EXISTS idx_matches_played_at ON matches(played_at DESC);
            """)

    def get_agent(self, name: str) -> Dict[str, Any]:
        """Fetch current rating profile for an agent, or default if unranked."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agents WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return dict(row)

        return {
            "name": name,
            "mu": DEFAULT_MU,
            "sigma": DEFAULT_SIGMA,
            "score": score(DEFAULT_MU, DEFAULT_SIGMA),
            "n_games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "total_turns": 0,
            "last_updated": None
        }

    def update_match(self, player0: str, player1: str, winner: Optional[int],
                     turns: int = 0, log_path: Optional[str] = None,
                     seed: Optional[int] = None) -> Dict[str, Any]:
        """Incrementally update ratings based on a single match outcome.

        winner: 0 (player0 won), 1 (player1 won), or None (draw).
        Does NOT re-read or recount any previous logs.
        """
        p0_data = self.get_agent(player0)
        p1_data = self.get_agent(player1)

        mu0_before, sig0_before = float(p0_data["mu"]), float(p0_data["sigma"])
        mu1_before, sig1_before = float(p1_data["mu"]), float(p1_data["sigma"])
        score0_before = score(mu0_before, sig0_before)
        score1_before = score(mu1_before, sig1_before)

        if winner == 0:
            outcome = 1
            winner_name = player0
            p0_win, p0_draw, p0_loss = 1, 0, 0
            p1_win, p1_draw, p1_loss = 0, 0, 1
        elif winner == 1:
            outcome = 2
            winner_name = player1
            p0_win, p0_draw, p0_loss = 0, 0, 1
            p1_win, p1_draw, p1_loss = 1, 0, 0
        else:
            outcome = 0
            winner_name = "Draw"
            p0_win, p0_draw, p0_loss = 0, 1, 0
            p1_win, p1_draw, p1_loss = 0, 1, 0

        mu0_after, sig0_after, mu1_after, sig1_after = rate_2p(
            mu0_before, sig0_before, mu1_before, sig1_before, outcome
        )

        score0_after = score(mu0_after, sig0_after)
        score1_after = score(mu1_after, sig1_after)

        with self._connection() as conn:
            cursor = conn.cursor()

            # Upsert player 0
            cursor.execute("""
                INSERT INTO agents (name, mu, sigma, score, n_games, wins, draws, losses, total_turns, last_updated)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                    mu = excluded.mu,
                    sigma = excluded.sigma,
                    score = excluded.score,
                    n_games = agents.n_games + 1,
                    wins = agents.wins + excluded.wins,
                    draws = agents.draws + excluded.draws,
                    losses = agents.losses + excluded.losses,
                    total_turns = agents.total_turns + excluded.total_turns,
                    last_updated = CURRENT_TIMESTAMP
            """, (player0, mu0_after, sig0_after, score0_after, p0_win, p0_draw, p0_loss, turns))

            # Upsert player 1
            cursor.execute("""
                INSERT INTO agents (name, mu, sigma, score, n_games, wins, draws, losses, total_turns, last_updated)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                    mu = excluded.mu,
                    sigma = excluded.sigma,
                    score = excluded.score,
                    n_games = agents.n_games + 1,
                    wins = agents.wins + excluded.wins,
                    draws = agents.draws + excluded.draws,
                    losses = agents.losses + excluded.losses,
                    total_turns = agents.total_turns + excluded.total_turns,
                    last_updated = CURRENT_TIMESTAMP
            """, (player1, mu1_after, sig1_after, score1_after, p1_win, p1_draw, p1_loss, turns))

            # Record match
            cursor.execute("""
                INSERT INTO matches (
                    player0, player1, winner, winner_name, turns, seed,
                    p0_mu_before, p0_sigma_before, p0_mu_after, p0_sigma_after,
                    p1_mu_before, p1_sigma_before, p1_mu_after, p1_sigma_after,
                    log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player0, player1, winner, winner_name, turns, seed,
                mu0_before, sig0_before, mu0_after, sig0_after,
                mu1_before, sig1_before, mu1_after, sig1_after,
                log_path
            ))

            conn.commit()

        return {
            "player0": {
                "name": player0,
                "mu_before": mu0_before,
                "sigma_before": sig0_before,
                "score_before": score0_before,
                "mu_after": mu0_after,
                "sigma_after": sig0_after,
                "score_after": score0_after,
                "score_delta": score0_after - score0_before
            },
            "player1": {
                "name": player1,
                "mu_before": mu1_before,
                "sigma_before": sig1_before,
                "score_before": score1_before,
                "mu_after": mu1_after,
                "sigma_after": sig1_after,
                "score_after": score1_after,
                "score_delta": score1_after - score1_before
            },
            "winner": winner,
            "winner_name": winner_name,
            "turns": turns
        }

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Return leaderboard ordered by TrueSkill score descending (matching bombIT)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, mu, sigma, score, n_games, wins, draws, losses, total_turns, last_updated
                FROM agents
                ORDER BY score DESC, mu DESC, sigma ASC, n_games DESC
            """)
            rows = cursor.fetchall()

        leaderboard = []
        for rank, row in enumerate(rows, start=1):
            n_games = row["n_games"]
            win_rate = (row["wins"] / n_games * 100.0) if n_games > 0 else 0.0
            avg_turns = (row["total_turns"] / n_games) if n_games > 0 else 0.0
            leaderboard.append({
                "rank": rank,
                "name": row["name"],
                "score": round(row["score"], 2),
                "mu": round(row["mu"], 2),
                "sigma": round(row["sigma"], 2),
                "n_games": n_games,
                "wins": row["wins"],
                "draws": row["draws"],
                "losses": row["losses"],
                "win_rate": round(win_rate, 1),
                "avg_turns": round(avg_turns, 1),
                "last_updated": row["last_updated"]
            })
        return leaderboard

    def get_agent_history(self, name: str, limit: int = 20) -> Dict[str, Any]:
        """Fetch agent profile and recent matches."""
        agent = self.get_agent(name)
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM matches
                WHERE player0 = ? OR player1 = ?
                ORDER BY id DESC
                LIMIT ?
            """, (name, name, limit))
            matches = [dict(r) for r in cursor.fetchall()]

        return {
            "agent": agent,
            "recent_matches": matches
        }

    def reset(self):
        """Reset ratings database."""
        with self._connection() as conn:
            conn.execute("DELETE FROM matches")
            conn.execute("DELETE FROM agents")
            conn.commit()

    def rebuild_from_logs(self, logs_dir: Path) -> int:
        """Optional tool: replay all match logs in logs_dir in chronological order."""
        logs_dir = Path(logs_dir).resolve()
        if not logs_dir.exists():
            return 0

        self.reset()
        files = sorted(logs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        count = 0

        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                if data.get("type") == "series" and "matches" in data:
                    for match in data["matches"]:
                        p0 = match["players"][0]["name"]
                        p1 = match["players"][1]["name"]
                        self.update_match(
                            player0=p0,
                            player1=p1,
                            winner=match.get("winner"),
                            turns=match.get("total_turns", len(match.get("turns", []))),
                            log_path=str(file_path),
                            seed=match.get("config", {}).get("seed")
                        )
                        count += 1
                elif "initial_state" in data and "turns" in data:
                    p0 = data["players"][0]["name"]
                    p1 = data["players"][1]["name"]
                    self.update_match(
                        player0=p0,
                        player1=p1,
                        winner=data.get("winner"),
                        turns=data.get("total_turns", len(data.get("turns", []))),
                        log_path=str(file_path),
                        seed=data.get("config", {}).get("seed")
                    )
                    count += 1
            except Exception:
                continue

        return count


def print_leaderboard(db_path: Optional[Path] = None):
    system = RankingSystem(db_path=db_path)
    leaderboard = system.get_leaderboard()

    if not leaderboard:
        print("\nNo ranked agents found. Play matches with arena.py or series.py to establish rankings.")
        return

    print("\n" + "=" * 80)
    print("WRONGWAY LEADERBOARD (TrueSkill Score = μ - 3σ)")
    print("=" * 80)
    header = f"{'Rank':4s}  {'Agent':20s}  {'Score':8s}  {'Skill (μ)':10s}  {'Uncertainty (σ)':16s}  {'Record (W/D/L)':16s}  {'Win%':6s}"
    print(header)
    print("-" * 80)

    for entry in leaderboard:
        record = f"{entry['wins']}/{entry['draws']}/{entry['losses']}"
        print(f"#{entry['rank']:<3d}  {entry['name']:20s}  {entry['score']:8.2f}  {entry['mu']:10.2f}  {entry['sigma']:16.2f}  {record:16s}  {entry['win_rate']:5.1f}%")

    print("=" * 80 + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard", "-l", action="store_true", default=True, help="Display leaderboard table")
    parser.add_argument("--agent", "-a", type=str, help="View detailed stats and match history for an agent")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild leaderboard by replaying all logs in logs/ dir")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs", help="Logs directory for rebuild")
    parser.add_argument("--reset", action="store_true", help="Reset all ratings in ranking.db")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to SQLite ranking database")
    args = parser.parse_args(argv)

    system = RankingSystem(db_path=args.db)

    if args.reset:
        system.reset()
        print(f"Reset ratings database at: {system.db_path}")
        return 0

    if args.rebuild:
        print(f"Rebuilding ratings from logs in {args.logs_dir}...")
        n = system.rebuild_from_logs(args.logs_dir)
        print(f"Processed {n} matches.")
        print_leaderboard(db_path=args.db)
        return 0

    if args.agent:
        history = system.get_agent_history(args.agent)
        agent = history["agent"]
        print(f"\n--- Agent: {agent['name']} ---")
        print(f"Score (μ - 3σ): {score(agent['mu'], agent['sigma']):.2f}")
        print(f"Skill (μ):      {agent['mu']:.2f}")
        print(f"Uncertainty (σ):{agent['sigma']:.2f}")
        print(f"Games:          {agent['n_games']} ({agent['wins']}W / {agent['draws']}D / {agent['losses']}L)")
        matches = history["recent_matches"]
        if matches:
            print(f"\nRecent matches ({len(matches)}):")
            for m in matches:
                vs = m['player1'] if m['player0'] == args.agent else m['player0']
                res = "WIN" if m['winner_name'] == args.agent else ("DRAW" if m['winner'] is None else "LOSS")
                print(f"  {m['played_at'][:19]} vs {vs:15s} | {res:4s} ({m['turns']} turns)")
        print()
        return 0

    print_leaderboard(db_path=args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
