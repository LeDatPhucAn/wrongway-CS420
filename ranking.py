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

                CREATE TABLE IF NOT EXISTS human_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bot_name TEXT NOT NULL,
                    human_name TEXT NOT NULL DEFAULT 'Human',
                    human_side INTEGER NOT NULL,
                    winner INTEGER,
                    human_won INTEGER NOT NULL DEFAULT 0,
                    bot_won INTEGER NOT NULL DEFAULT 0,
                    is_draw INTEGER NOT NULL DEFAULT 0,
                    turns INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    log_path TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_agents_score ON agents(score DESC);
                CREATE INDEX IF NOT EXISTS idx_matches_played_at ON matches(played_at DESC);
                CREATE INDEX IF NOT EXISTS idx_human_matches_bot ON human_matches(bot_name, id DESC);
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

    @staticmethod
    def _to_rel_log_path(raw_path: Optional[str], logs_dir: Optional[Path] = None) -> Optional[str]:
        if not raw_path:
            return None
        base_logs = Path(logs_dir or (ROOT / "logs")).resolve()
        try:
            rel = Path(raw_path).resolve().relative_to(base_logs)
            return rel.as_posix()
        except ValueError:
            norm = str(raw_path).replace("\\", "/")
            marker = "/logs/"
            if marker in norm:
                return norm.split(marker, 1)[1]
            if norm.startswith("logs/"):
                return norm[len("logs/"):]
            return None

    def get_agent_history(
        self,
        name: str,
        limit: int = 200,
        opponent: Optional[str] = None,
        logs_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Fetch agent profile, per-opponent W/D/L breakdown, and bot-vs-bot matches."""
        agent = self.get_agent(name)
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    CASE WHEN player0 = ? THEN player1 ELSE player0 END AS opponent,
                    COUNT(*) AS games,
                    SUM(CASE WHEN (player0 = ? AND winner = 0) OR (player1 = ? AND winner = 1) THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN winner IS NULL THEN 1 ELSE 0 END) AS draws,
                    SUM(CASE WHEN (player0 = ? AND winner = 1) OR (player1 = ? AND winner = 0) THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN player0 = ? AND winner = 0 THEN 1 ELSE 0 END) AS p0_wins,
                    SUM(CASE WHEN player0 = ? AND winner = 1 THEN 1 ELSE 0 END) AS p0_losses,
                    SUM(CASE WHEN player1 = ? AND winner = 1 THEN 1 ELSE 0 END) AS p1_wins,
                    SUM(CASE WHEN player1 = ? AND winner = 0 THEN 1 ELSE 0 END) AS p1_losses,
                    COALESCE(SUM(turns), 0) AS total_turns,
                    MAX(log_path) AS log_path
                FROM matches
                WHERE player0 = ? OR player1 = ?
                GROUP BY opponent
                ORDER BY wins DESC, losses ASC, opponent ASC
            """, (name, name, name, name, name, name, name, name, name, name, name))
            h2h_rows = cursor.fetchall()

            if opponent:
                cursor.execute("""
                    SELECT * FROM matches
                    WHERE (player0 = ? AND player1 = ?) OR (player0 = ? AND player1 = ?)
                    ORDER BY id DESC
                    LIMIT ?
                """, (name, opponent, opponent, name, int(limit)))
            else:
                cursor.execute("""
                    SELECT * FROM matches
                    WHERE player0 = ? OR player1 = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (name, name, int(limit)))
            raw_matches = [dict(r) for r in cursor.fetchall()]

        head_to_head: List[Dict[str, Any]] = []
        for r in h2h_rows:
            games = int(r["games"])
            wins = int(r["wins"] or 0)
            draws = int(r["draws"] or 0)
            losses = int(r["losses"] or 0)
            total_turns = int(r["total_turns"] or 0)
            head_to_head.append({
                "opponent": r["opponent"],
                "games": games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": round((wins / games * 100.0) if games > 0 else 0.0, 1),
                "p0_wins": int(r["p0_wins"] or 0),
                "p0_losses": int(r["p0_losses"] or 0),
                "p1_wins": int(r["p1_wins"] or 0),
                "p1_losses": int(r["p1_losses"] or 0),
                "avg_turns": round((total_turns / games) if games > 0 else 0.0, 1),
                "log_rel_path": self._to_rel_log_path(r["log_path"], logs_dir=logs_dir),
            })

        matches: List[Dict[str, Any]] = []
        for m in raw_matches:
            is_p0 = (m["player0"] == name)
            opp_name = m["player1"] if is_p0 else m["player0"]
            if m["winner"] is None:
                res_label = "DRAW"
            elif (is_p0 and m["winner"] == 0) or ((not is_p0) and m["winner"] == 1):
                res_label = "WIN"
            else:
                res_label = "LOSS"

            if is_p0:
                s_before = score(m["p0_mu_before"], m["p0_sigma_before"]) if m["p0_mu_before"] is not None else 0.0
                s_after = score(m["p0_mu_after"], m["p0_sigma_after"]) if m["p0_mu_after"] is not None else 0.0
            else:
                s_before = score(m["p1_mu_before"], m["p1_sigma_before"]) if m["p1_mu_before"] is not None else 0.0
                s_after = score(m["p1_mu_after"], m["p1_sigma_after"]) if m["p1_mu_after"] is not None else 0.0

            m["opponent"] = opp_name
            m["bot_side"] = 0 if is_p0 else 1
            m["result"] = res_label
            m["score_delta"] = round(s_after - s_before, 2)
            m["log_rel_path"] = self._to_rel_log_path(m.get("log_path"), logs_dir=logs_dir)
            matches.append(m)

        return {
            "agent": agent,
            "head_to_head": head_to_head,
            "recent_matches": matches,
        }

    def reset(self, include_human: bool = False):
        """Reset bot-vs-bot ratings database (optionally also clearing human matches)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM matches")
            conn.execute("DELETE FROM agents")
            if include_human:
                conn.execute("DELETE FROM human_matches")
            conn.commit()

    def reset_human_matches(self):
        """Clear all recorded Human vs Bot matches."""
        with self._connection() as conn:
            conn.execute("DELETE FROM human_matches")
            conn.commit()

    def record_human_match(
        self,
        bot_name: str,
        human_side: int,
        winner: Optional[int],
        turns: int = 0,
        reason: Optional[str] = None,
        log_path: Optional[str] = None,
        human_name: str = "Human",
    ) -> Dict[str, Any]:
        """Record a completed Human vs Bot match outcome without altering bot TrueSkill ratings."""
        if human_side not in (0, 1):
            raise ValueError("human_side must be 0 or 1")
        if winner not in (None, 0, 1):
            raise ValueError("winner must be 0, 1, or None")

        clean_human = (human_name or "Human").strip() or "Human"
        if winner is None:
            human_won, bot_won, is_draw = 0, 0, 1
            outcome_for_human = "draw"
            outcome_for_bot = "draw"
        elif winner == human_side:
            human_won, bot_won, is_draw = 1, 0, 0
            outcome_for_human = "win"
            outcome_for_bot = "loss"
        else:
            human_won, bot_won, is_draw = 0, 1, 0
            outcome_for_human = "loss"
            outcome_for_bot = "win"

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO human_matches (
                    bot_name, human_name, human_side, winner,
                    human_won, bot_won, is_draw, turns, reason, log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bot_name, clean_human, human_side, winner,
                human_won, bot_won, is_draw, int(turns), reason, log_path
            ))
            match_id = cursor.lastrowid
            conn.commit()

        return {
            "id": match_id,
            "bot_name": bot_name,
            "human_name": clean_human,
            "human_side": human_side,
            "bot_side": 1 - human_side,
            "winner": winner,
            "outcome_for_human": outcome_for_human,
            "outcome_for_bot": outcome_for_bot,
            "turns": int(turns),
            "reason": reason,
            "log_path": log_path,
        }

    def get_human_bot_stats(self, bot_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return per-bot Human vs Bot win/loss/draw totals."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    bot_name,
                    COUNT(*) AS games,
                    COALESCE(SUM(bot_won), 0) AS bot_wins,
                    COALESCE(SUM(human_won), 0) AS bot_losses,
                    COALESCE(SUM(is_draw), 0) AS draws,
                    COALESCE(SUM(CASE WHEN human_side = 1 AND bot_won = 1 THEN 1 ELSE 0 END), 0) AS p0_bot_wins,
                    COALESCE(SUM(CASE WHEN human_side = 1 AND human_won = 1 THEN 1 ELSE 0 END), 0) AS p0_bot_losses,
                    COALESCE(SUM(CASE WHEN human_side = 0 AND bot_won = 1 THEN 1 ELSE 0 END), 0) AS p1_bot_wins,
                    COALESCE(SUM(CASE WHEN human_side = 0 AND human_won = 1 THEN 1 ELSE 0 END), 0) AS p1_bot_losses,
                    COALESCE(SUM(turns), 0) AS total_turns,
                    MAX(played_at) AS last_played_at
                FROM human_matches
                GROUP BY bot_name
            """)
            agg = {row["bot_name"]: dict(row) for row in cursor.fetchall()}

        ordered_names = list(dict.fromkeys([*(bot_names or []), *sorted(agg.keys())]))
        results: List[Dict[str, Any]] = []
        for name in ordered_names:
            row = agg.get(name)
            if row:
                games = int(row["games"])
                bot_wins = int(row["bot_wins"])
                bot_losses = int(row["bot_losses"])
                draws = int(row["draws"])
                total_turns = int(row["total_turns"])
                results.append({
                    "bot_name": name,
                    "games": games,
                    "bot_wins": bot_wins,
                    "bot_losses": bot_losses,
                    "draws": draws,
                    "human_wins": bot_losses,
                    "human_losses": bot_wins,
                    "bot_win_rate": round((bot_wins / games * 100.0) if games > 0 else 0.0, 1),
                    "human_win_rate": round((bot_losses / games * 100.0) if games > 0 else 0.0, 1),
                    "p0_bot_wins": int(row["p0_bot_wins"]),
                    "p0_bot_losses": int(row["p0_bot_losses"]),
                    "p1_bot_wins": int(row["p1_bot_wins"]),
                    "p1_bot_losses": int(row["p1_bot_losses"]),
                    "avg_turns": round((total_turns / games) if games > 0 else 0.0, 1),
                    "last_played_at": row["last_played_at"],
                })
            else:
                results.append({
                    "bot_name": name,
                    "games": 0,
                    "bot_wins": 0,
                    "bot_losses": 0,
                    "draws": 0,
                    "human_wins": 0,
                    "human_losses": 0,
                    "bot_win_rate": 0.0,
                    "human_win_rate": 0.0,
                    "p0_bot_wins": 0,
                    "p0_bot_losses": 0,
                    "p1_bot_wins": 0,
                    "p1_bot_losses": 0,
                    "avg_turns": 0.0,
                    "last_played_at": None,
                })

        results.sort(key=lambda r: (-r["games"], -r["bot_wins"], r["bot_name"]))
        return results

    def get_human_match_history(self, bot_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent Human vs Bot matches, optionally filtered by bot_name."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if bot_name:
                cursor.execute("""
                    SELECT * FROM human_matches
                    WHERE bot_name = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (bot_name, int(limit)))
            else:
                cursor.execute("""
                    SELECT * FROM human_matches
                    ORDER BY id DESC
                    LIMIT ?
                """, (int(limit),))
            rows = [dict(r) for r in cursor.fetchall()]

        for r in rows:
            if r["is_draw"]:
                r["outcome_for_human"] = "draw"
                r["outcome_for_bot"] = "draw"
            elif r["human_won"]:
                r["outcome_for_human"] = "win"
                r["outcome_for_bot"] = "loss"
            else:
                r["outcome_for_human"] = "loss"
                r["outcome_for_bot"] = "win"
        return rows

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


def print_human_stats(db_path: Optional[Path] = None, bot_names: Optional[List[str]] = None):
    system = RankingSystem(db_path=db_path)
    stats = system.get_human_bot_stats(bot_names=bot_names)

    if not stats:
        print("\nNo Human vs Bot matches recorded yet.")
        return

    print("\n" + "=" * 86)
    print("HUMAN VS BOT RECORD (Per Bot)")
    print("=" * 86)
    print(f"{'Bot':20s}  {'GP':>3s}  {'Bot W':>5s}  {'Bot L':>5s}  {'Draws':>5s}  {'Bot Win%':>8s}  {'You Win%':>8s}  {'AvgT':>5s}")
    print("-" * 86)
    for r in stats:
        print(
            f"{r['bot_name']:20s}  {r['games']:3d}  {r['bot_wins']:5d}  {r['bot_losses']:5d}  "
            f"{r['draws']:5d}  {r['bot_win_rate']:7.1f}%  {r['human_win_rate']:7.1f}%  {r['avg_turns']:5.1f}"
        )
    print("=" * 86 + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard", "-l", action="store_true", default=True, help="Display leaderboard table")
    parser.add_argument("--human", action="store_true", help="Display Human vs Bot win/loss table per bot")
    parser.add_argument("--agent", "-a", type=str, help="View detailed stats and match history for an agent")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild leaderboard by replaying all logs in logs/ dir")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs", help="Logs directory for rebuild")
    parser.add_argument("--reset", action="store_true", help="Reset all bot-vs-bot ratings in ranking.db")
    parser.add_argument("--reset-human", action="store_true", help="Reset all Human vs Bot match records")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to SQLite ranking database")
    args = parser.parse_args(argv)

    system = RankingSystem(db_path=args.db)

    if args.reset:
        system.reset()
        print(f"Reset ratings database at: {system.db_path}")
        return 0

    if args.reset_human:
        system.reset_human_matches()
        print(f"Reset Human vs Bot match history at: {system.db_path}")
        return 0

    if args.human:
        print_human_stats(db_path=args.db)
        return 0

    if args.rebuild:
        print(f"Rebuilding ratings from logs in {args.logs_dir}...")
        n = system.rebuild_from_logs(args.logs_dir)
        print(f"Processed {n} matches.")
        print_leaderboard(db_path=args.db)
        return 0

    if args.agent:
        history = system.get_agent_history(args.agent, logs_dir=args.logs_dir)
        agent = history["agent"]
        print(f"\n--- Agent: {agent['name']} ---")
        print(f"Score (μ - 3σ): {score(agent['mu'], agent['sigma']):.2f}")
        print(f"Skill (μ):      {agent['mu']:.2f}")
        print(f"Uncertainty (σ):{agent['sigma']:.2f}")
        print(f"Games:          {agent['n_games']} ({agent['wins']}W / {agent['draws']}D / {agent['losses']}L)")

        h2h = history.get("head_to_head", [])
        if h2h:
            print("\nHead-to-Head vs Other Bots:")
            print(f"  {'Opponent':22s}  {'GP':>3s}  {'W/D/L':>9s}  {'Win%':>6s}  {'P0(W-L)':>8s}  {'P1(W-L)':>8s}  {'AvgT':>5s}")
            print("  " + "-" * 72)
            for row in h2h:
                wdl = f"{row['wins']}/{row['draws']}/{row['losses']}"
                p0s = f"{row['p0_wins']}-{row['p0_losses']}"
                p1s = f"{row['p1_wins']}-{row['p1_losses']}"
                print(
                    f"  {row['opponent']:22s}  {row['games']:3d}  {wdl:>9s}  {row['win_rate']:5.1f}%  "
                    f"{p0s:>8s}  {p1s:>8s}  {row['avg_turns']:5.1f}"
                )

        matches = history["recent_matches"]
        if matches:
            print(f"\nRecorded Bot-vs-Bot Matches ({len(matches)}):")
            for m in matches:
                side = "P0" if m["bot_side"] == 0 else "P1"
                print(
                    f"  {m['played_at'][:19]} | {side} vs {m['opponent']:20s} | "
                    f"{m['result']:4s} ({m['turns']:3d} turns, Δ {m['score_delta']:+.2f})"
                )
        print()
        return 0

    print_leaderboard(db_path=args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
