#!/usr/bin/env python3
"""Human vs Bot match runner, session manager, and per-bot W/L recorder."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

if __package__:
    from .agent_runner import StudentAgent
    from .arena import resolve_player, save_json
    from .environment import Environment
    from .ranking import DEFAULT_DB_PATH, RankingSystem
    from .tournament import discover_agents
    from .visualizer import render_board
else:
    from agent_runner import StudentAgent
    from arena import resolve_player, save_json
    from environment import Environment
    from ranking import DEFAULT_DB_PATH, RankingSystem
    from tournament import discover_agents
    from visualizer import render_board

ROOT = Path(__file__).resolve().parent
DEFAULT_SUBMISSIONS_DIR = ROOT / "submissions"
DEFAULT_LOGS_DIR = ROOT / "logs"


def list_available_bots(
    submissions_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Discover all submission bots and merge Human vs Bot records + AI leaderboard scores."""
    sub_dir = Path(submissions_dir or DEFAULT_SUBMISSIONS_DIR).resolve()
    try:
        agent_paths = discover_agents(submissions_dir=sub_dir)
    except ValueError:
        agent_paths = []

    bot_names = sorted(p.stem for p in agent_paths)
    ranking = RankingSystem(db_path=db_path or DEFAULT_DB_PATH)
    human_stats = {row["bot_name"]: row for row in ranking.get_human_bot_stats(bot_names=bot_names)}
    ai_board = {row["name"]: row for row in ranking.get_leaderboard()}

    bots: List[Dict[str, Any]] = []
    for name in bot_names:
        h = human_stats[name]
        ai = ai_board.get(name)
        bots.append({
            "name": name,
            "human_stats": h,
            "ai_rank": ai["rank"] if ai else None,
            "ai_score": ai["score"] if ai else None,
            "ai_win_rate": ai["win_rate"] if ai else None,
        })

    bots.sort(
        key=lambda item: (
            item["ai_rank"] if item["ai_rank"] is not None else 9999,
            item["name"],
        )
    )
    return bots


class HumanMatchSession:
    """Thread-safe interactive match between one Human player and one StudentAgent bot."""

    def __init__(
        self,
        bot_name: str,
        *,
        human_side: int = 0,
        human_name: str = "Human",
        timeout: float = 1.0,
        startup_timeout: float = 5.0,
        max_turns: int = 500,
        seed: int = 0,
        submissions_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        if human_side not in (0, 1):
            raise ValueError("human_side must be 0 or 1")

        self.session_id = uuid.uuid4().hex[:12]
        self.sub_dir = Path(submissions_dir or DEFAULT_SUBMISSIONS_DIR).resolve()
        self.logs_dir = Path(logs_dir or DEFAULT_LOGS_DIR).resolve()
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()

        self.bot_path = resolve_player(bot_name, submissions_dir=self.sub_dir)
        self.bot_name = self.bot_path.stem
        self.human_side = int(human_side)
        self.bot_side = 1 - self.human_side
        self.human_name = (human_name or "Human").strip() or "Human"

        self.timeout = float(timeout)
        self.startup_timeout = float(startup_timeout)
        self.max_turns = int(max_turns)
        self.seed = int(seed)

        self.environment = Environment(max_turns=self.max_turns, starting_player=0)
        self.initial_state = self.environment.snapshot()
        self.turns: List[Dict[str, Any]] = []
        self.stats = [
            {"ok": 0, "pass": 0, "invalid": 0, "timeout": 0, "error": 0},
            {"ok": 0, "pass": 0, "invalid": 0, "timeout": 0, "error": 0},
        ]

        self._lock = threading.Lock()
        self._agent: Optional[StudentAgent] = StudentAgent(
            self.bot_path,
            timeout=self.timeout,
            startup_timeout=self.startup_timeout,
            seed=self.seed,
        )
        self._finalized = False
        self.log_rel_path: Optional[str] = None
        self.recorded_match: Optional[Dict[str, Any]] = None
        self.created_at = time.monotonic()
        self.updated_at = self.created_at

    def start(self) -> Dict[str, Any]:
        """If the human plays second (P1), let the bot play Turn 0 right away."""
        with self._lock:
            if not self.environment.done and self.environment.current_player == self.bot_side:
                self._run_bot_turn_locked()
            self.updated_at = time.monotonic()
            return self._snapshot_payload_locked()

    def submit_human_move(self, action: Any) -> Dict[str, Any]:
        """Validate and apply a human action, then run the bot reply if the match continues."""
        with self._lock:
            if self.environment.done:
                raise ValueError("Match has already finished")
            if self.environment.current_player != self.human_side:
                raise ValueError("It is currently not the human player's turn")

            self._validate_human_action_locked(action)

            started = time.monotonic()
            update = self.environment.step(action)
            elapsed = time.monotonic() - started
            status = update["status"]
            error = update.get("reason") if status == "invalid" else None
            self.stats[self.human_side][status] += 1
            self.turns.append({
                "turn": self.environment.turn_count,
                "player": self.human_side,
                "action": update["action"],
                "status": status,
                "reason": update.get("reason"),
                "error": error,
                "elapsed": round(elapsed, 6),
                "state": self.environment.snapshot(),
            })

            if not self.environment.done:
                self._run_bot_turn_locked()

            if self.environment.done:
                self._finalize_locked()

            self.updated_at = time.monotonic()
            return self._snapshot_payload_locked()

    def resign(self) -> Dict[str, Any]:
        """Concede the match on behalf of the human player and record a bot win."""
        with self._lock:
            if not self.environment.done:
                self.environment.done = True
                self.environment.winner = self.bot_side
                self.environment.termination_reason = "resign"
            self._finalize_locked()
            self.updated_at = time.monotonic()
            return self._snapshot_payload_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_payload_locked()

    def close(self) -> None:
        with self._lock:
            self._close_agent_locked()

    def _validate_human_action_locked(self, action: Any) -> None:
        if not isinstance(action, dict):
            raise ValueError("Action must be a JSON object")
        atype = action.get("type")
        if atype == "pass":
            return
        if atype == "move":
            to = action.get("to")
            if not isinstance(to, (list, tuple)) or len(to) != 2 or any(type(v) is not int for v in to):
                raise ValueError("Invalid move coordinates")
            target = [int(to[0]), int(to[1])]
            if target not in self.environment.legal_moves(self.human_side):
                raise ValueError(f"Illegal pawn move: {target}")
            return
        if atype == "wall":
            orientation = action.get("orientation")
            row, col = action.get("row"), action.get("col")
            if orientation not in ("H", "V"):
                raise ValueError("Wall orientation must be 'H' or 'V'")
            if type(row) is not int or type(col) is not int or not (0 <= row < 8 and 0 <= col < 8):
                raise ValueError("Wall coordinates must be integers in 0..7")
            err = self.environment._wall_error(self.human_side, orientation, row, col)
            if err is not None:
                raise ValueError(f"Illegal wall placement ({err})")
            return
        raise ValueError(f"Unsupported action type: {atype!r}")

    def _run_bot_turn_locked(self) -> None:
        if self.environment.done or self._agent is None:
            return
        obs = self.environment.get_observation(self.bot_side)
        response = self._agent.act(obs)
        if response.status == "ok":
            update = self.environment.step(response.action)
            status = update["status"]
            error = update.get("reason") if status == "invalid" else None
        else:
            update = self.environment.skip_turn(response.status)
            status = response.status
            error = response.error

        self.stats[self.bot_side][status] += 1
        self.turns.append({
            "turn": self.environment.turn_count,
            "player": self.bot_side,
            "action": update["action"],
            "status": status,
            "reason": update.get("reason"),
            "error": error,
            "elapsed": round(response.elapsed, 6),
            "state": self.environment.snapshot(),
        })

    def _close_agent_locked(self) -> None:
        if self._agent is not None:
            try:
                self._agent.close()
            finally:
                self._agent = None

    def _players_meta(self) -> List[Dict[str, Any]]:
        p0 = (
            {"id": 0, "name": self.human_name, "type": "human"}
            if self.human_side == 0
            else {"id": 0, "name": self.bot_name, "type": "bot", "path": str(self.bot_path)}
        )
        p1 = (
            {"id": 1, "name": self.human_name, "type": "human"}
            if self.human_side == 1
            else {"id": 1, "name": self.bot_name, "type": "bot", "path": str(self.bot_path)}
        )
        return [p0, p1]

    def _build_replay_payload_locked(self, include_outcome: bool) -> Dict[str, Any]:
        players = self._players_meta()
        payload: Dict[str, Any] = {
            "format_version": 1,
            "mode": "human_vs_bot",
            "players": players,
            "config": {
                "timeout": self.timeout,
                "startup_timeout": self.startup_timeout,
                "max_turns": self.max_turns,
                "seed": self.seed,
                "starting_player": 0,
                "human_side": self.human_side,
                "bot_name": self.bot_name,
            },
            "initial_state": self.initial_state,
            "turns": list(self.turns),
            "stats": [dict(s) for s in self.stats],
        }
        if include_outcome and self.environment.done:
            w = self.environment.winner
            payload.update({
                "winner": w,
                "winner_name": players[w]["name"] if w is not None else None,
                "reason": self.environment.termination_reason,
                "total_turns": self.environment.turn_count,
                "final_state": self.environment.snapshot(),
            })
        return payload

    def _finalize_locked(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._close_agent_locked()

        replay_payload = self._build_replay_payload_locked(include_outcome=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_path = Path("human") / f"{self.bot_name}_vs_human_{stamp}_{self.session_id[:6]}.json"
        out_path = self.logs_dir / rel_path
        save_json(out_path, replay_payload)
        self.log_rel_path = rel_path.as_posix()

        ranking = RankingSystem(db_path=self.db_path)
        self.recorded_match = ranking.record_human_match(
            bot_name=self.bot_name,
            human_side=self.human_side,
            winner=self.environment.winner,
            turns=self.environment.turn_count,
            reason=self.environment.termination_reason,
            log_path=self.log_rel_path,
            human_name=self.human_name,
        )

    def _snapshot_payload_locked(self) -> Dict[str, Any]:
        state = self.environment.snapshot()
        is_human_turn = (not self.environment.done) and (self.environment.current_player == self.human_side)
        legal_moves = self.environment.legal_moves(self.human_side) if is_human_turn else []
        legal_walls = self.environment.legal_wall_actions(self.human_side) if is_human_turn else []

        d0_map = self.environment.distance_map(0)
        d1_map = self.environment.distance_map(1)
        p0, p1 = self.environment.positions
        distances = [d0_map[p0[0]][p0[1]], d1_map[p1[0]][p1[1]]]

        ranking = RankingSystem(db_path=self.db_path)
        bot_record = ranking.get_human_bot_stats(bot_names=[self.bot_name])[0]

        players = self._players_meta()
        winner = self.environment.winner if self.environment.done else None
        winner_name = players[winner]["name"] if (self.environment.done and winner is not None) else None

        return {
            "session_id": self.session_id,
            "bot_name": self.bot_name,
            "human_name": self.human_name,
            "human_side": self.human_side,
            "bot_side": self.bot_side,
            "done": self.environment.done,
            "winner": winner,
            "winner_name": winner_name,
            "reason": self.environment.termination_reason,
            "state": state,
            "legal_moves": legal_moves,
            "legal_walls": legal_walls,
            "distances": distances,
            "turns": list(self.turns),
            "stats": [dict(s) for s in self.stats],
            "log_file": self.log_rel_path,
            "recorded_match": self.recorded_match,
            "bot_record": bot_record,
            "replay_data": self._build_replay_payload_locked(include_outcome=self.environment.done),
        }


class HumanSessionManager:
    """Lifecycle manager for active Human vs Bot sessions served over HTTP."""

    def __init__(
        self,
        *,
        submissions_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        max_sessions: int = 16,
    ) -> None:
        self.submissions_dir = Path(submissions_dir or DEFAULT_SUBMISSIONS_DIR).resolve()
        self.logs_dir = Path(logs_dir or DEFAULT_LOGS_DIR).resolve()
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self.max_sessions = int(max_sessions)
        self._lock = threading.Lock()
        self._sessions: Dict[str, HumanMatchSession] = {}

    def start_session(
        self,
        bot_name: str,
        *,
        human_side: int = 0,
        human_name: str = "Human",
        timeout: float = 1.0,
        startup_timeout: float = 5.0,
        max_turns: int = 500,
        seed: int = 0,
    ) -> Dict[str, Any]:
        session = HumanMatchSession(
            bot_name=bot_name,
            human_side=human_side,
            human_name=human_name,
            timeout=timeout,
            startup_timeout=startup_timeout,
            max_turns=max_turns,
            seed=seed,
            submissions_dir=self.submissions_dir,
            logs_dir=self.logs_dir,
            db_path=self.db_path,
        )
        try:
            payload = session.start()
        except Exception:
            session.close()
            raise

        with self._lock:
            self._prune_locked()
            self._sessions[session.session_id] = session
        return payload

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        session = self._require_session(session_id)
        return session.snapshot()

    def play_move(self, session_id: str, action: Any) -> Dict[str, Any]:
        session = self._require_session(session_id)
        return session.submit_human_move(action)

    def resign_session(self, session_id: str) -> Dict[str, Any]:
        session = self._require_session(session_id)
        return session.resign()

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def _require_session(self, session_id: str) -> HumanMatchSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown human match session: {session_id}")
        return session

    def _prune_locked(self) -> None:
        if len(self._sessions) < self.max_sessions:
            return
        # Evict oldest finished sessions first, then oldest overall
        ordered = sorted(
            self._sessions.items(),
            key=lambda kv: (0 if kv[1].environment.done else 1, kv[1].updated_at),
        )
        while len(self._sessions) >= self.max_sessions and ordered:
            sid, sess = ordered.pop(0)
            self._sessions.pop(sid, None)
            sess.close()


def _parse_cli_move(text: str) -> Dict[str, Any]:
    parts = text.strip().split()
    if not parts:
        raise ValueError("Empty input")
    cmd = parts[0].lower()
    if cmd in ("resign", "quit", "q"):
        return {"_resign": True}
    if cmd == "pass":
        return {"type": "pass"}
    if cmd in ("m", "move") and len(parts) == 3:
        return {"type": "move", "to": [int(parts[1]), int(parts[2])]}
    if cmd in ("h", "v") and len(parts) == 3:
        return {"type": "wall", "orientation": cmd.upper(), "row": int(parts[1]), "col": int(parts[2])}
    if cmd == "w" and len(parts) == 4 and parts[1].upper() in ("H", "V"):
        return {"type": "wall", "orientation": parts[1].upper(), "row": int(parts[2]), "col": int(parts[3])}
    raise ValueError("Commands: 'm R C' (move), 'H R C' or 'V R C' (wall), 'resign'")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play Wrongway in terminal against any bot and record W/L stats.")
    parser.add_argument("--bot", "-b", help="Bot name or path in submissions/")
    parser.add_argument("--side", "-s", type=int, choices=[0, 1], default=0, help="Your side: 0=P0 (bottom), 1=P1 (top)")
    parser.add_argument("--name", default="Human", help="Your display name")
    parser.add_argument("--stats", action="store_true", help="Show Human vs Bot win/loss table per bot")
    parser.add_argument("--list-bots", action="store_true", help="List available bots")
    args = parser.parse_args(argv)

    if args.list_bots or args.stats or not args.bot:
        bots = list_available_bots()
        print("\n=== Available Bots (Human vs Bot Record) ===")
        for item in bots:
            h = item["human_stats"]
            ai_txt = f"AI #{item['ai_rank']} ({item['ai_score']:.1f})" if item["ai_rank"] else "unranked"
            print(
                f"  {item['name']:22s} | You {h['human_wins']}W - {h['human_losses']}L - {h['draws']}D "
                f"| Bot {h['bot_wins']}W - {h['bot_losses']}L | {ai_txt}"
            )
        print()
        if not args.bot:
            return 0

    session = HumanMatchSession(bot_name=args.bot, human_side=args.side, human_name=args.name)
    try:
        data = session.start()
        names = [
            args.name if args.side == 0 else session.bot_name,
            args.name if args.side == 1 else session.bot_name,
        ]
        while not data["done"]:
            print("\n" + render_board(data["state"], names))
            print(f"Legal moves: {data['legal_moves']}")
            raw = input("Your action ('m R C', 'H R C', 'V R C', or 'resign'): ")
            try:
                cmd = _parse_cli_move(raw)
            except ValueError as exc:
                print(f"Input error: {exc}")
                continue
            if cmd.get("_resign"):
                data = session.resign()
                break
            try:
                data = session.submit_human_move(cmd)
            except ValueError as exc:
                print(f"Illegal action: {exc}")

        print("\n" + render_board(data["state"], names))
        rec = data.get("recorded_match") or {}
        print(f"Match Over! Winner: {data['winner_name'] or 'Draw'} ({data['reason']})")
        print(f"Recorded outcome for {session.bot_name}: Bot {rec.get('outcome_for_bot')} | You {rec.get('outcome_for_human')}")
        br = data["bot_record"]
        print(f"Updated record vs {session.bot_name}: You {br['human_wins']}W - {br['human_losses']}L - {br['draws']}D | Log: {data['log_file']}")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
