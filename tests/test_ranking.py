#!/usr/bin/env python3
"""Tests for the incremental TrueSkill ranking system."""

import math
from pathlib import Path
import shutil
import tempfile
import unittest

from ranking import RankingSystem, score, rate_2p, DEFAULT_MU, DEFAULT_SIGMA


class TestRankingSystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ww_rank_test_"))
        self.db_path = self.temp_dir / "test_ranking.db"
        self.ranking = RankingSystem(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_score_calculation(self):
        self.assertAlmostEqual(score(100.0, 33.333333333333336), 0.0, places=4)
        self.assertAlmostEqual(score(120.0, 20.0), 60.0, places=4)

    def test_rate_2p_win(self):
        m1, s1, m2, s2 = rate_2p(100.0, 100.0 / 3.0, 100.0, 100.0 / 3.0, outcome=1)
        # Winner's skill increases
        self.assertGreater(m1, 100.0)
        # Loser's skill decreases
        self.assertLess(m2, 100.0)
        # Both uncertainties decrease
        self.assertLess(s1, 100.0 / 3.0)
        self.assertLess(s2, 100.0 / 3.0)
        # Winner's conservative score increases
        self.assertGreater(score(m1, s1), 0.0)

    def test_rate_2p_draw_equal(self):
        m1, s1, m2, s2 = rate_2p(100.0, 25.0, 100.0, 25.0, outcome=0)
        # Mus remain symmetric
        self.assertAlmostEqual(m1, 100.0, places=4)
        self.assertAlmostEqual(m2, 100.0, places=4)
        # Sigmas decrease
        self.assertLess(s1, 25.0)
        self.assertLess(s2, 25.0)

    def test_rate_2p_draw_unequal(self):
        m1, s1, m2, s2 = rate_2p(130.0, 20.0, 90.0, 20.0, outcome=0)
        # Higher rated agent drops, lower rated agent rises
        self.assertLess(m1, 130.0)
        self.assertGreater(m2, 90.0)

    def test_incremental_updates(self):
        # Match 1: agentA beats agentB
        u1 = self.ranking.update_match("agentA", "agentB", winner=0, turns=25)
        self.assertGreater(u1["player0"]["score_delta"], 0)
        self.assertLess(u1["player1"]["score_delta"], 0)

        a_data = self.ranking.get_agent("agentA")
        b_data = self.ranking.get_agent("agentB")
        self.assertEqual(a_data["n_games"], 1)
        self.assertEqual(a_data["wins"], 1)
        self.assertEqual(b_data["losses"], 1)

        # Match 2: agentA beats agentB again (incremental update)
        u2 = self.ranking.update_match("agentA", "agentB", winner=0, turns=20)
        # Before score of match 2 must equal after score of match 1
        self.assertAlmostEqual(u2["player0"]["score_before"], u1["player0"]["score_after"], places=4)

        a_data2 = self.ranking.get_agent("agentA")
        self.assertEqual(a_data2["n_games"], 2)
        self.assertEqual(a_data2["wins"], 2)

        # Match 3: agentC draws with agentA
        u3 = self.ranking.update_match("agentC", "agentA", winner=None, turns=50)
        c_data = self.ranking.get_agent("agentC")
        self.assertEqual(c_data["draws"], 1)

        # Leaderboard test
        lb = self.ranking.get_leaderboard()
        self.assertEqual(len(lb), 3)
        # agentA should be rank #1
        self.assertEqual(lb[0]["name"], "agentA")
        self.assertEqual(lb[0]["rank"], 1)

        # History test
        hist = self.ranking.get_agent_history("agentA")
        self.assertEqual(len(hist["recent_matches"]), 3)

    def test_human_match_recording_and_isolation(self):
        # Record matches across sides against two bots
        r1 = self.ranking.record_human_match("botA", human_side=0, winner=0, turns=22, reason="goal")
        self.assertEqual(r1["outcome_for_human"], "win")
        self.assertEqual(r1["outcome_for_bot"], "loss")

        r2 = self.ranking.record_human_match("botA", human_side=1, winner=0, turns=30, reason="goal")
        self.assertEqual(r2["outcome_for_human"], "loss")
        self.assertEqual(r2["outcome_for_bot"], "win")

        self.ranking.record_human_match("botB", human_side=0, winner=None, turns=500, reason="max_turns")

        stats = {s["bot_name"]: s for s in self.ranking.get_human_bot_stats(["botA", "botB", "botC"])}
        self.assertEqual(stats["botA"]["games"], 2)
        self.assertEqual(stats["botA"]["bot_wins"], 1)
        self.assertEqual(stats["botA"]["bot_losses"], 1)
        self.assertEqual(stats["botA"]["human_wins"], 1)
        self.assertEqual(stats["botA"]["human_losses"], 1)
        self.assertEqual(stats["botA"]["p0_bot_wins"], 1)
        self.assertEqual(stats["botA"]["p1_bot_losses"], 1)

        self.assertEqual(stats["botB"]["draws"], 1)
        self.assertEqual(stats["botC"]["games"], 0)

        # Human matches must not pollute AI TrueSkill leaderboard
        self.assertEqual(self.ranking.get_leaderboard(), [])

        # Default tournament reset preserves human records
        self.ranking.reset()
        self.assertEqual(len(self.ranking.get_human_match_history()), 3)

        # Explicit human reset clears them
        self.ranking.reset_human_matches()
        self.assertEqual(len(self.ranking.get_human_match_history()), 0)

    def test_agent_head_to_head_and_match_records(self):
        """Per-opponent bot-vs-bot W/D/L breakdown, side splits, and filtering."""
        logs_root = Path(self.temp_dir.name) / "logs"
        logs_root.mkdir(parents=True, exist_ok=True)
        series_ab = str(logs_root / "tournament" / "series" / "BotA_vs_BotB.json")

        # BotA vs BotB: 2 wins (as P0, as P1), 1 loss (as P1)
        self.ranking.update_match("BotA", "BotB", winner=0, turns=30, log_path=series_ab)
        self.ranking.update_match("BotB", "BotA", winner=1, turns=34, log_path=series_ab)
        self.ranking.update_match("BotB", "BotA", winner=0, turns=40, log_path=series_ab)
        # BotA vs BotC: 1 draw (as P0)
        self.ranking.update_match("BotA", "BotC", winner=None, turns=150)

        hist = self.ranking.get_agent_history("BotA", logs_dir=logs_root)
        h2h = {row["opponent"]: row for row in hist["head_to_head"]}

        self.assertEqual(set(h2h.keys()), {"BotB", "BotC"})
        self.assertEqual(h2h["BotB"]["games"], 3)
        self.assertEqual(h2h["BotB"]["wins"], 2)
        self.assertEqual(h2h["BotB"]["losses"], 1)
        self.assertEqual(h2h["BotB"]["draws"], 0)
        self.assertEqual(h2h["BotB"]["p0_wins"], 1)
        self.assertEqual(h2h["BotB"]["p1_wins"], 1)
        self.assertEqual(h2h["BotB"]["p1_losses"], 1)
        self.assertEqual(h2h["BotB"]["log_rel_path"], "tournament/series/BotA_vs_BotB.json")

        self.assertEqual(h2h["BotC"]["games"], 1)
        self.assertEqual(h2h["BotC"]["draws"], 1)
        self.assertIsNone(h2h["BotC"]["log_rel_path"])

        self.assertEqual(len(hist["recent_matches"]), 4)
        self.assertEqual(hist["recent_matches"][0]["result"], "DRAW")
        self.assertEqual(hist["recent_matches"][0]["opponent"], "BotC")

        filtered = self.ranking.get_agent_history("BotA", opponent="BotB", logs_dir=logs_root)
        self.assertEqual(len(filtered["recent_matches"]), 3)
        self.assertTrue(all(m["opponent"] == "BotB" for m in filtered["recent_matches"]))


if __name__ == "__main__":
    unittest.main()
