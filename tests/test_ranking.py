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


if __name__ == "__main__":
    unittest.main()
