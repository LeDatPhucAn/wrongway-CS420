#!/usr/bin/env python3
"""Tests for the round-robin tournament runner."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tournament import (
    build_round_robin_schedule,
    discover_agents,
    run_tournament,
)

ROOT = Path(__file__).resolve().parent.parent


class TestTournament(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ww_tourney_test_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discover_agents_excludes_template_by_default(self):
        default_agents = discover_agents(ROOT / "submissions")
        names = {p.stem for p in default_agents}
        self.assertNotIn("TEMPLATE", names)
        self.assertIn("greedy", names)

        with_template = discover_agents(ROOT / "submissions", include_template=True)
        self.assertIn("TEMPLATE", {p.stem for p in with_template})

    def test_schedule_odd_side_balance(self):
        dummy = [Path(f"bot_{i}.py") for i in range(13)]
        pairs = build_round_robin_schedule(dummy)
        self.assertEqual(len(pairs), 13 * 12 // 2)

        seen_pairs = set()
        primary_counts = {i: 0 for i in range(13)}
        for idx_a, idx_b, _pa, _pb in pairs:
            key = tuple(sorted((idx_a, idx_b)))
            self.assertNotIn(key, seen_pairs)
            seen_pairs.add(key)
            primary_counts[idx_a] += 1

        self.assertTrue(all(count == 6 for count in primary_counts.values()))

    def test_schedule_even_side_balance(self):
        dummy = [Path(f"bot_{i}.py") for i in range(4)]
        pairs = build_round_robin_schedule(dummy)
        self.assertEqual(len(pairs), 6)

        primary_counts = {i: 0 for i in range(4)}
        for idx_a, _idx_b, _pa, _pb in pairs:
            primary_counts[idx_a] += 1

        self.assertTrue(all(count in (1, 2) for count in primary_counts.values()))

    def test_run_tournament_end_to_end(self):
        agents = discover_agents(
            ROOT / "submissions",
            agents=["greedy", "random_agent", "wall_builder"],
        )
        report = run_tournament(
            agents,
            games=3,
            swap_sides=True,
            seed=7,
            max_turns=60,
            workers=2,
            out_dir=self.temp_dir,
            save_series_logs=True,
            update_rank=False,
        )
        self.assertEqual(report["config"]["total_matches"], 9)
        self.assertEqual(len(report["standings"]), 3)
        for row in report["standings"]:
            self.assertEqual(row["games"], 6)
            self.assertEqual(row["p0_games"], 3)
            self.assertEqual(row["p1_games"], 3)
            self.assertEqual(row["points"], 3 * row["wins"] + row["draws"])

        summary_file = self.temp_dir / "summary.json"
        self.assertTrue(summary_file.is_file())
        loaded = json.loads(summary_file.read_text(encoding="utf-8"))
        self.assertEqual(len(loaded["matches"]), 9)
        series_files = list((self.temp_dir / "series").glob("*.json"))
        self.assertEqual(len(series_files), 3)


if __name__ == "__main__":
    unittest.main()
