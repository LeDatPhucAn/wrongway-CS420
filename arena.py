#!/usr/bin/env python3
"""Run two single-file student agents, with one second per action by default."""

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path

if __package__:
    from .agent_runner import StudentAgent
    from .environment import Environment
    from .visualizer import render_board
else:
    from agent_runner import StudentAgent
    from environment import Environment
    from visualizer import render_board


ROOT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def resolve_player(value, submissions_dir=None):
    """Accept an explicit .py path, a filename, or a student ID."""
    value = Path(value).expanduser()
    folder = Path(submissions_dir) if submissions_dir else ROOT / "submissions"
    candidates = [value, folder / value]
    if not value.suffix:
        candidates.append(folder / (str(value) + ".py"))
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix == ".py":
            return candidate.resolve()
    raise ValueError("Student Python file not found: {}".format(value))


def save_json(path, payload):
    """Write a UTF-8 result, replacing the destination only after serialization."""
    path = Path(path)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_match(player1, player2, *, timeout=1.0, max_turns=500, seed=0,
              starting_player=0, startup_timeout=5.0, on_turn=None):
    """Return a complete replay. Callback receives a detached turn record.

    Player 0 always starts at the bottom; player 1 starts at the top.
    A call error, invalid action or timeout consumes exactly one turn.
    The turn timer only covers the agent call and its communication, not the
    engine's calculation of legal actions or the separate import budget.
    """
    paths = [resolve_player(player1), resolve_player(player2)]
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("startup_timeout must be a finite positive number")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    environment = Environment(max_turns=max_turns, starting_player=starting_player)
    result = {
        "format_version": 1,
        "players": [{"id": i, "name": path.stem, "path": str(path)}
                    for i, path in enumerate(paths)],
        "config": {"timeout": timeout, "startup_timeout": startup_timeout,
                   "max_turns": max_turns, "seed": seed,
                   "starting_player": starting_player},
        "initial_state": environment.snapshot(),
        "turns": [],
        "stats": [{"ok": 0, "pass": 0, "invalid": 0, "timeout": 0, "error": 0}
                  for _ in range(2)],
    }
    with ExitStack() as stack:
        agents = [stack.enter_context(StudentAgent(
            path, timeout=timeout, startup_timeout=startup_timeout,
            seed=seed * 2 + i)) for i, path in enumerate(paths)]
        while not environment.done:
            player = environment.current_player
            observation = environment.get_observation(player)
            response = agents[player].act(observation)
            if response.status == "ok":
                update = environment.step(response.action)
                status = update["status"]
                error = update.get("reason") if status == "invalid" else None
            else:
                update = environment.skip_turn(response.status)
                status = response.status
                error = response.error
            record = {
                "turn": environment.turn_count,
                "player": player,
                "action": update["action"],
                "status": status,
                "reason": update.get("reason"),
                "error": error,
                "elapsed": round(response.elapsed, 6),
                "state": environment.snapshot(),
            }
            result["stats"][player][status] += 1
            result["turns"].append(record)
            if on_turn:
                # A renderer or instructor callback cannot alter the replay.
                on_turn(json.loads(json.dumps(record)))
    result.update({
        "winner": environment.winner,
        "winner_name": (paths[environment.winner].stem
                        if environment.winner is not None else None),
        "reason": environment.termination_reason,
        "total_turns": environment.turn_count,
        "final_state": environment.snapshot(),
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player1", "-p1", required=True, help="Python file or student ID")
    parser.add_argument("--player2", "-p2", required=True, help="Python file or student ID")
    parser.add_argument("--submissions-dir", type=Path, default=ROOT / "submissions")
    parser.add_argument("--timeout", type=positive_float, default=1.0, help="seconds per action (default: 1)")
    parser.add_argument("--startup-timeout", type=positive_float, default=5.0, help="seconds to import a student file")
    parser.add_argument("--max-turns", type=positive_int, default=500, help="individual turns, including skipped turns")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--starting-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--render", action="store_true", help="print the board after each turn")
    parser.add_argument("--delay", type=nonnegative_float, default=0.0, help="display delay in seconds")
    parser.add_argument("--log", type=Path, help="save the full JSON replay")
    args = parser.parse_args(argv)
    try:
        players = [resolve_player(value, args.submissions_dir)
                   for value in (args.player1, args.player2)]
    except ValueError as exc:
        parser.error(str(exc))
    names = [path.stem for path in players]

    def show(record):
        print("\nTurn {} | P{} {} | {} | {}".format(
            record["turn"], record["player"], names[record["player"]],
            record["status"], record["action"]))
        if record["error"]:
            print("  " + record["error"])
        print(render_board(record["state"], names))
        if args.delay:
            time.sleep(args.delay)

    if args.render:
        print(render_board(Environment(args.max_turns, args.starting_player).snapshot(), names))
    try:
        result = run_match(*players, timeout=args.timeout, max_turns=args.max_turns,
                           seed=args.seed, starting_player=args.starting_player,
                           startup_timeout=args.startup_timeout,
                           on_turn=show if args.render else None)
        if args.log:
            save_json(args.log, result)
    except KeyboardInterrupt:
        print("\nMatch interrupted; student processes stopped.", file=sys.stderr)
        return 130
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if result["winner"] is None:
        print("Draw after {} turns ({})".format(result["total_turns"], result["reason"]))
    else:
        print("Winner: P{} {} after {} turns".format(
            result["winner"], result["winner_name"], result["total_turns"]))
    for i, stats in enumerate(result["stats"]):
        print("P{} {}: {}".format(i, names[i], json.dumps(stats)))
    if args.log:
        print("Replay: {}".format(args.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
