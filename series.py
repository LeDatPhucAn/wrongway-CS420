#!/usr/bin/env python3
"""Run a series of matches between two agents, tracking score and side advantage.

Outputs a multi-match series JSON compatible with the Wrongway Replay Viewer.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from arena import resolve_player, run_match, save_json, positive_float, positive_int

ROOT = Path(__file__).resolve().parent


def run_series(player1, player2, games=10, swap_sides=True, seed=0,
               timeout=1.0, max_turns=500, out_path=None):
    p1_path = resolve_player(player1)
    p2_path = resolve_player(player2)
    name1 = p1_path.stem
    name2 = p2_path.stem

    matches = []
    wins = {name1: 0, name2: 0, "draw": 0}
    side_wins = {"P0": 0, "P1": 0, "draw": 0}
    turns_list = []

    print(f"Starting Series: {name1} vs {name2} ({games} games)")
    print(f"Swap sides: {swap_sides} | Base seed: {seed} | Max turns: {max_turns}")
    print("-" * 65)

    start_time = time.time()

    for i in range(games):
        game_seed = seed + i
        # If swap_sides is enabled, alternate player roles
        if swap_sides and (i % 2 == 1):
            p0, p1 = p2_path, p1_path
            p0_name, p1_name = name2, name1
        else:
            p0, p1 = p1_path, p2_path
            p0_name, p1_name = name1, name2

        result = run_match(
            p0, p1,
            timeout=timeout,
            max_turns=max_turns,
            seed=game_seed,
            starting_player=0
        )

        matches.append(result)
        winner_id = result["winner"]
        total_turns = result["total_turns"]
        turns_list.append(total_turns)

        if winner_id is None:
            winner_name = "Draw"
            wins["draw"] += 1
            side_wins["draw"] += 1
            outcome_str = f"Draw ({result['reason']})"
        else:
            winner_name = p0_name if winner_id == 0 else p1_name
            wins[winner_name] += 1
            side_key = f"P{winner_id}"
            side_wins[side_key] += 1
            outcome_str = f"Winner: {winner_name} (P{winner_id}) in {total_turns} turns"

        print(f"Game {i + 1:2d}/{games}: P0 {p0_name:15s} vs P1 {p1_name:15s} -> {outcome_str}")

    elapsed = time.time() - start_time
    avg_turns = sum(turns_list) / max(1, len(turns_list))

    summary = {
        "games": games,
        "agents": [name1, name2],
        "wins": {
            name1: wins[name1],
            name2: wins[name2],
            "draw": wins["draw"]
        },
        "side_wins": side_wins,
        "avg_turns": round(avg_turns, 2),
        "elapsed_seconds": round(elapsed, 2)
    }

    print("\n" + "=" * 65)
    print(f"SERIES SUMMARY ({games} matches in {elapsed:.1f}s)")
    print("=" * 65)
    print(f"{name1:20s}: {wins[name1]:2d} wins ({wins[name1]/games*100:5.1f}%)")
    print(f"{name2:20s}: {wins[name2]:2d} wins ({wins[name2]/games*100:5.1f}%)")
    print(f"{'Draws':20s}: {wins['draw']:2d} draws({wins['draw']/games*100:5.1f}%)")
    print("-" * 65)
    print(f"Side advantage: P0 (bottom) = {side_wins['P0']} wins | P1 (top) = {side_wins['P1']} wins")
    print(f"Average match length: {avg_turns:.1f} turns")
    print("=" * 65)

    series_data = {
        "type": "series",
        "format_version": 1,
        "agents": [name1, name2],
        "summary": summary,
        "matches": matches
    }

    if out_path:
        out_path = Path(out_path)
        save_json(out_path, series_data)
        print(f"Saved series replay: {out_path.resolve()}")

    return series_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player1", "-p1", required=True, help="Agent 1 (.py file or name)")
    parser.add_argument("--player2", "-p2", required=True, help="Agent 2 (.py file or name)")
    parser.add_argument("--games", "-n", type=positive_int, default=10, help="Number of games (default: 10)")
    parser.add_argument("--swap-sides", action="store_true", default=True, help="Alternate P0/P1 starting positions")
    parser.add_argument("--no-swap", dest="swap_sides", action="store_false", help="Do not swap starting positions")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (default: 0)")
    parser.add_argument("--timeout", type=positive_float, default=1.0, help="Seconds per action (default: 1.0)")
    parser.add_argument("--max-turns", type=positive_int, default=500, help="Max turns per game (default: 500)")
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "series.json", help="Output series JSON log")
    args = parser.parse_args(argv)

    run_series(
        args.player1,
        args.player2,
        games=args.games,
        swap_sides=args.swap_sides,
        seed=args.seed,
        timeout=args.timeout,
        max_turns=args.max_turns,
        out_path=args.out
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
