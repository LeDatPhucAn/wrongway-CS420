#!/usr/bin/env python3
"""Run a round-robin tournament across Wrongway submission bots.

Supports parallel match execution, balanced starting-side assignment for odd
match counts, head-to-head matrices, tournament points (3/1/0), TrueSkill
leaderboard updates in ranking.db, and replayable per-pair series logs.
"""

import argparse
import concurrent.futures
import fnmatch
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__:
    from .arena import positive_float, positive_int, resolve_player, run_match, save_json
    from .ranking import RankingSystem
else:
    from arena import positive_float, positive_int, resolve_player, run_match, save_json
    from ranking import RankingSystem


ROOT = Path(__file__).resolve().parent
DEFAULT_SUBMISSIONS_DIR = ROOT / "submissions"
DEFAULT_OUT_DIR = ROOT / "logs" / "tournament"


def default_workers() -> int:
    """Choose a conservative worker count that avoids CPU starvation on timeouts."""
    cpus = os.cpu_count() or 2
    return max(1, min(6, cpus // 2))


def _is_excluded(stem: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(stem, pat) for pat in patterns)


def discover_agents(
    submissions_dir: Path = DEFAULT_SUBMISSIONS_DIR,
    include_template: bool = False,
    agents: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Resolve participating bot files from submissions_dir or explicit names."""
    folder = Path(submissions_dir).resolve()
    excluded = [Path(name).stem for name in (exclude or [])]

    if agents:
        resolved: List[Path] = []
        seen = set()
        for item in agents:
            path = resolve_player(item, folder)
            if _is_excluded(path.stem, excluded) or path in seen:
                continue
            seen.add(path)
            resolved.append(path)
    else:
        if not folder.is_dir():
            raise ValueError(f"Submissions directory not found: {folder}")
        resolved = []
        for path in sorted(folder.glob("*.py"), key=lambda p: p.stem.lower()):
            if path.stem.startswith("_"):
                continue
            if not include_template and path.stem == "TEMPLATE":
                continue
            if _is_excluded(path.stem, excluded):
                continue
            resolved.append(path.resolve())

    if len(resolved) < 2:
        raise ValueError(f"Need at least 2 agents for a tournament, found {len(resolved)}")
    return resolved


def build_round_robin_schedule(agent_paths: Sequence[Path]) -> List[Tuple[int, int, Path, Path]]:
    """Build balanced round-robin pairs using the circle method.

    Returns a list of (idx_a, idx_b, primary_p0_path, primary_p1_path) where
    primary_p0/primary_p1 alternates across opponents so every agent receives an
    equal (or near-equal for even N) number of extra first-player games when
    games-per-pair is odd.
    """
    n = len(agent_paths)
    half = n // 2

    def orient_pair(i: int, j: int) -> Tuple[int, int]:
        lo, hi = (i, j) if i < j else (j, i)
        diff = hi - lo
        if n % 2 == 1:
            primary_is_lo = diff <= half
        else:
            primary_is_lo = diff < half or (diff == half and lo % 2 == 0)
        return (lo, hi) if primary_is_lo else (hi, lo)

    # Circle method gives well-interleaved rounds rather than bunching one bot.
    ring = list(range(n))
    has_bye = n % 2 == 1
    if has_bye:
        ring.append(-1)
    m = len(ring)

    pairs: List[Tuple[int, int, Path, Path]] = []
    for _round in range(m - 1):
        for slot in range(m // 2):
            u = ring[slot]
            v = ring[m - 1 - slot]
            if u == -1 or v == -1:
                continue
            p_a, p_b = orient_pair(u, v)
            pairs.append((p_a, p_b, agent_paths[p_a], agent_paths[p_b]))
        ring = [ring[0], ring[-1]] + ring[1:-1]
    return pairs


def _run_match_task(task: Tuple[int, int, str, str, float, int, int, float]) -> Tuple[int, int, Dict[str, Any]]:
    """Worker entry point for running a single match (must be picklable on Windows)."""
    pair_idx, game_idx, p0_str, p1_str, timeout, max_turns, seed, startup_timeout = task
    result = run_match(
        p0_str,
        p1_str,
        timeout=timeout,
        max_turns=max_turns,
        seed=seed,
        starting_player=0,
        startup_timeout=startup_timeout,
    )
    return pair_idx, game_idx, result


def _summarize_pair(
    name_a: str,
    name_b: str,
    matches: List[Dict[str, Any]],
    elapsed: float,
) -> Dict[str, Any]:
    """Compute a series summary matching series.py's JSON schema."""
    wins = {name_a: 0, name_b: 0, "draw": 0}
    side_wins = {"P0": 0, "P1": 0, "draw": 0}
    total_turns = 0

    for match in matches:
        p0_name = match["players"][0]["name"]
        p1_name = match["players"][1]["name"]
        winner = match["winner"]
        total_turns += match["total_turns"]
        if winner is None:
            wins["draw"] += 1
            side_wins["draw"] += 1
        elif winner == 0:
            wins[p0_name] += 1
            side_wins["P0"] += 1
        else:
            wins[p1_name] += 1
            side_wins["P1"] += 1

    games = len(matches)
    return {
        "games": games,
        "agents": [name_a, name_b],
        "wins": wins,
        "side_wins": side_wins,
        "avg_turns": round(total_turns / max(1, games), 2),
        "elapsed_seconds": round(elapsed, 2),
    }


def _empty_agent_stats(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "p0_games": 0,
        "p0_wins": 0,
        "p0_draws": 0,
        "p0_losses": 0,
        "p1_games": 0,
        "p1_wins": 0,
        "p1_draws": 0,
        "p1_losses": 0,
        "total_turns": 0,
        "timeouts": 0,
        "errors": 0,
        "invalids": 0,
        "passes": 0,
    }


def run_tournament(
    agent_paths: Sequence[Path],
    *,
    games: int = 5,
    swap_sides: bool = True,
    seed: int = 0,
    timeout: float = 1.0,
    startup_timeout: float = 5.0,
    max_turns: int = 500,
    workers: int = 1,
    out_dir: Optional[Path] = DEFAULT_OUT_DIR,
    save_series_logs: bool = True,
    update_rank: bool = True,
    reset_rank: bool = False,
) -> Dict[str, Any]:
    """Execute a full round-robin tournament and return standings & summary."""
    names = [p.stem for p in agent_paths]
    pairs = build_round_robin_schedule(agent_paths)
    total_pairs = len(pairs)
    total_matches = total_pairs * games

    print(
        f"Starting Round-Robin Tournament: {len(names)} bots | "
        f"{total_pairs} pairs x {games} games = {total_matches} matches"
    )
    print(
        f"Swap sides: {swap_sides} | Base seed: {seed} | Timeout: {timeout}s | "
        f"Max turns: {max_turns} | Workers: {workers}"
    )
    print("Participants: " + ", ".join(names))
    print("-" * 78)

    ranking_sys: Optional[RankingSystem] = None
    if update_rank:
        ranking_sys = RankingSystem()
        if reset_rank:
            ranking_sys.reset()

    if out_dir and save_series_logs:
        series_dir = Path(out_dir) / "series"
        if series_dir.is_dir():
            for old_file in series_dir.glob("*.json"):
                old_file.unlink()

    tasks: List[Tuple[int, int, str, str, float, int, int, float]] = []
    for pair_idx, (_ia, _ib, pa_path, pb_path) in enumerate(pairs):
        for game_idx in range(games):
            if swap_sides and (game_idx % 2 == 1):
                p0_path, p1_path = pb_path, pa_path
            else:
                p0_path, p1_path = pa_path, pb_path
            match_seed = seed + pair_idx * games + game_idx
            tasks.append((
                pair_idx,
                game_idx,
                str(p0_path),
                str(p1_path),
                timeout,
                max_turns,
                match_seed,
                startup_timeout,
            ))

    start_time = time.perf_counter()
    pair_matches: List[List[Optional[Dict[str, Any]]]] = [
        [None] * games for _ in range(total_pairs)
    ]
    pair_remaining = [games] * total_pairs
    pair_results_summary: List[Optional[Dict[str, Any]]] = [None] * total_pairs

    agent_stats: Dict[str, Dict[str, Any]] = {n: _empty_agent_stats(n) for n in names}
    h2h: Dict[str, Dict[str, Dict[str, int]]] = {
        a: {b: {"wins": 0, "draws": 0, "losses": 0} for b in names if b != a}
        for a in names
    }
    overall_side_wins = {"P0": 0, "P1": 0, "draw": 0}
    completed_count = 0

    def handle_completed_match(pair_idx: int, game_idx: int, result: Dict[str, Any]) -> None:
        nonlocal completed_count
        pair_matches[pair_idx][game_idx] = result
        pair_remaining[pair_idx] -= 1
        completed_count += 1

        if pair_remaining[pair_idx] == 0:
            _ia, _ib, pa_path, pb_path = pairs[pair_idx]
            name_a, name_b = pa_path.stem, pb_path.stem
            matches = [m for m in pair_matches[pair_idx] if m is not None]
            summary = _summarize_pair(name_a, name_b, matches, 0.0)
            pair_results_summary[pair_idx] = summary

            series_path: Optional[Path] = None
            if out_dir and save_series_logs:
                series_path = Path(out_dir) / "series" / f"{name_a}_vs_{name_b}.json"
                save_json(series_path, {
                    "type": "series",
                    "format_version": 1,
                    "agents": [name_a, name_b],
                    "summary": summary,
                    "matches": matches,
                })
                # Free turn-by-turn arrays from memory once saved to disk.
                pair_matches[pair_idx] = []

            wa = summary["wins"][name_a]
            wb = summary["wins"][name_b]
            wd = summary["wins"]["draw"]
            print(
                f"[{completed_count:3d}/{total_matches}] "
                f"{name_a:16s} vs {name_b:16s} -> {wa}-{wb}-{wd} "
                f"(avg {summary['avg_turns']:.1f} turns)"
            )

    if workers <= 1:
        for task in tasks:
            p_idx, g_idx, res = _run_match_task(task)
            handle_completed_match(p_idx, g_idx, res)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_match_task, task) for task in tasks]
            for fut in concurrent.futures.as_completed(futures):
                p_idx, g_idx, res = fut.result()
                handle_completed_match(p_idx, g_idx, res)

    # Aggregate statistics and apply TrueSkill updates in deterministic order.
    # If series logs were flushed to disk to save RAM, re-read lightweight fields
    # from disk; otherwise use in-memory match dicts.
    compact_matches: List[Dict[str, Any]] = []
    for pair_idx, (_ia, _ib, pa_path, pb_path) in enumerate(pairs):
        name_a, name_b = pa_path.stem, pb_path.stem
        if out_dir and save_series_logs and not pair_matches[pair_idx]:
            series_path = Path(out_dir) / "series" / f"{name_a}_vs_{name_b}.json"
            matches = json.loads(series_path.read_text(encoding="utf-8"))["matches"]
            log_ref = str(series_path)
        else:
            matches = [m for m in pair_matches[pair_idx] if m is not None]
            log_ref = None

        for match in matches:
            p0_name = match["players"][0]["name"]
            p1_name = match["players"][1]["name"]
            winner = match["winner"]
            turns = match["total_turns"]
            match_seed = match["config"]["seed"]
            stats0, stats1 = match["stats"][0], match["stats"][1]

            s0 = agent_stats[p0_name]
            s1 = agent_stats[p1_name]
            s0["games"] += 1
            s1["games"] += 1
            s0["p0_games"] += 1
            s1["p1_games"] += 1
            s0["total_turns"] += turns
            s1["total_turns"] += turns

            for s, st in ((s0, stats0), (s1, stats1)):
                s["timeouts"] += st.get("timeout", 0)
                s["errors"] += st.get("error", 0)
                s["invalids"] += st.get("invalid", 0)
                s["passes"] += st.get("pass", 0)

            if winner is None:
                s0["draws"] += 1
                s1["draws"] += 1
                s0["points"] += 1
                s1["points"] += 1
                s0["p0_draws"] += 1
                s1["p1_draws"] += 1
                h2h[p0_name][p1_name]["draws"] += 1
                h2h[p1_name][p0_name]["draws"] += 1
                overall_side_wins["draw"] += 1
            elif winner == 0:
                s0["wins"] += 1
                s1["losses"] += 1
                s0["points"] += 3
                s0["p0_wins"] += 1
                s1["p1_losses"] += 1
                h2h[p0_name][p1_name]["wins"] += 1
                h2h[p1_name][p0_name]["losses"] += 1
                overall_side_wins["P0"] += 1
            else:
                s1["wins"] += 1
                s0["losses"] += 1
                s1["points"] += 3
                s1["p1_wins"] += 1
                s0["p0_losses"] += 1
                h2h[p1_name][p0_name]["wins"] += 1
                h2h[p0_name][p1_name]["losses"] += 1
                overall_side_wins["P1"] += 1

            compact_matches.append({
                "p0": p0_name,
                "p1": p1_name,
                "winner": winner,
                "winner_name": match["winner_name"],
                "turns": turns,
                "reason": match["reason"],
                "seed": match_seed,
                "stats": match["stats"],
            })

            if ranking_sys is not None:
                ranking_sys.update_match(
                    player0=p0_name,
                    player1=p1_name,
                    winner=winner,
                    turns=turns,
                    log_path=log_ref,
                    seed=match_seed,
                )

    elapsed = time.perf_counter() - start_time

    standings = []
    for name, s in agent_stats.items():
        gp = max(1, s["games"])
        entry = {
            **s,
            "win_rate": round(s["wins"] / gp * 100.0, 1),
            "point_pct": round(s["points"] / (3 * gp) * 100.0, 1),
            "avg_turns": round(s["total_turns"] / gp, 1),
        }
        if ranking_sys is not None:
            r = ranking_sys.get_agent(name)
            entry["trueskill_score"] = round(r["score"], 2)
            entry["trueskill_mu"] = round(r["mu"], 2)
            entry["trueskill_sigma"] = round(r["sigma"], 2)
        standings.append(entry)

    # Sort by tournament points DESC, then wins DESC, then win_rate DESC, then avg_turns ASC.
    standings.sort(
        key=lambda e: (-e["points"], -e["wins"], -e["win_rate"], e["avg_turns"], e["name"])
    )
    for rank, entry in enumerate(standings, start=1):
        entry["rank"] = rank

    report = {
        "type": "tournament",
        "format_version": 1,
        "config": {
            "games_per_pair": games,
            "total_pairs": total_pairs,
            "total_matches": total_matches,
            "swap_sides": swap_sides,
            "seed": seed,
            "timeout": timeout,
            "startup_timeout": startup_timeout,
            "max_turns": max_turns,
            "workers": workers,
        },
        "elapsed_seconds": round(elapsed, 2),
        "agents": names,
        "side_wins": overall_side_wins,
        "standings": standings,
        "head_to_head": h2h,
        "pair_summaries": [p for p in pair_results_summary if p is not None],
        "matches": compact_matches,
    }

    if out_dir:
        out_path = Path(out_dir) / "summary.json"
        save_json(out_path, report)
        report["summary_path"] = str(out_path.resolve())

    print_tournament_report(report, show_trueskill=ranking_sys is not None)
    return report


def print_tournament_report(report: Dict[str, Any], show_trueskill: bool = True) -> None:
    """Print formatted standings and head-to-head matrix to stdout."""
    cfg = report["config"]
    side = report["side_wins"]
    total = cfg["total_matches"]
    print("\n" + "=" * 102)
    print(
        f"ROUND-ROBIN TOURNAMENT STANDINGS "
        f"({total} matches, {cfg['games_per_pair']} per pair, {report['elapsed_seconds']:.1f}s)"
    )
    print("=" * 102)

    if show_trueskill:
        header = (
            f"{'Rank':4s}  {'Agent':16s}  {'GP':>3s}  {'W/D/L':>10s}  "
            f"{'Pts':>4s}  {'Win%':>6s}  {'P0(W-L)':>8s}  {'P1(W-L)':>8s}  "
            f"{'AvgT':>5s}  {'TO/Err':>6s}  {'TS Score':>8s}"
        )
    else:
        header = (
            f"{'Rank':4s}  {'Agent':16s}  {'GP':>3s}  {'W/D/L':>10s}  "
            f"{'Pts':>4s}  {'Win%':>6s}  {'P0(W-L)':>8s}  {'P1(W-L)':>8s}  "
            f"{'AvgT':>5s}  {'TO/Err':>6s}"
        )
    print(header)
    print("-" * 102)

    for row in report["standings"]:
        wdl = f"{row['wins']}/{row['draws']}/{row['losses']}"
        p0_wl = f"{row['p0_wins']}-{row['p0_losses']}"
        p1_wl = f"{row['p1_wins']}-{row['p1_losses']}"
        issues = f"{row['timeouts']}/{row['errors'] + row['invalids']}"
        line = (
            f"#{row['rank']:<3d}  {row['name']:16s}  {row['games']:3d}  {wdl:>10s}  "
            f"{row['points']:4d}  {row['win_rate']:5.1f}%  {p0_wl:>8s}  {p1_wl:>8s}  "
            f"{row['avg_turns']:5.1f}  {issues:>6s}"
        )
        if show_trueskill:
            line += f"  {row.get('trueskill_score', 0.0):8.2f}"
        print(line)

    print("-" * 102)
    p0_pct = side["P0"] / max(1, total) * 100.0
    p1_pct = side["P1"] / max(1, total) * 100.0
    d_pct = side["draw"] / max(1, total) * 100.0
    print(
        f"Side Advantage: P0 (bottom/first) = {side['P0']} ({p0_pct:.1f}%) | "
        f"P1 (top/second) = {side['P1']} ({p1_pct:.1f}%) | "
        f"Draws = {side['draw']} ({d_pct:.1f}%)"
    )

    ordered = [r["name"] for r in report["standings"]]
    h2h = report["head_to_head"]
    print("\nHEAD-TO-HEAD MATRIX (row vs col: W-L, or W-L-D when draws occurred)")
    col_w = 6 if all(
        h2h[a][b]["draws"] == 0 for a in ordered for b in ordered if a != b
    ) else 8
    label_w = max(16, max((len(n) for n in ordered), default=16)) + 5
    top = " " * (label_w + 2) + "".join(f"{f'#{idx}':>{col_w}s}" for idx in range(1, len(ordered) + 1))
    print(top)
    for idx, a in enumerate(ordered, start=1):
        cells = []
        for b in ordered:
            if a == b:
                cells.append(f"{'--':>{col_w}s}")
            else:
                rec = h2h[a][b]
                txt = f"{rec['wins']}-{rec['losses']}" if rec["draws"] == 0 else f"{rec['wins']}-{rec['losses']}-{rec['draws']}"
                cells.append(f"{txt:>{col_w}s}")
        row_label = f"#{idx:<2d} {a}"
        print(f"{row_label:{label_w}s}  " + "".join(cells))

    if "summary_path" in report:
        print(f"\nSaved tournament summary: {report['summary_path']}")
    print("=" * 102 + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions-dir", type=Path, default=DEFAULT_SUBMISSIONS_DIR)
    parser.add_argument("--agents", nargs="+", help="Explicit list of agents (.py or names)")
    parser.add_argument("--exclude", nargs="+", default=[], help="Agent names to exclude")
    parser.add_argument("--include-template", action="store_true", help="Include TEMPLATE.py")
    parser.add_argument("--games", "-n", type=positive_int, default=5, help="Matches per pair (default: 5)")
    parser.add_argument("--no-swap", dest="swap_sides", action="store_false", help="Do not swap P0/P1 sides")
    parser.set_defaults(swap_sides=True)
    parser.add_argument("--seed", type=int, default=0, help="Base seed (default: 0)")
    parser.add_argument("--timeout", type=positive_float, default=1.0, help="Seconds per action (default: 1.0)")
    parser.add_argument("--startup-timeout", type=positive_float, default=5.0, help="Seconds per agent startup (default: 5.0)")
    parser.add_argument("--max-turns", type=positive_int, default=500, help="Max turns per match (default: 500)")
    parser.add_argument("--workers", "-w", type=positive_int, default=default_workers(), help="Parallel match workers")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for summary & series logs")
    parser.add_argument("--no-series-logs", action="store_true", help="Do not save per-pair series JSON files")
    parser.add_argument("--no-rank", action="store_true", help="Do not update ranking.db")
    parser.add_argument("--reset-rank", action="store_true", help="Reset ranking.db before starting tournament")
    args = parser.parse_args(argv)

    try:
        agent_paths = discover_agents(
            submissions_dir=args.submissions_dir,
            include_template=args.include_template,
            agents=args.agents,
            exclude=args.exclude,
        )
    except ValueError as exc:
        parser.error(str(exc))

    run_tournament(
        agent_paths,
        games=args.games,
        swap_sides=args.swap_sides,
        seed=args.seed,
        timeout=args.timeout,
        startup_timeout=args.startup_timeout,
        max_turns=args.max_turns,
        workers=args.workers,
        out_dir=args.out_dir,
        save_series_logs=not args.no_series_logs,
        update_rank=not args.no_rank,
        reset_rank=args.reset_rank,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
