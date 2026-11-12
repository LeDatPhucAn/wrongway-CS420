"""Wrongway Bot - Waller (Mauerer) (Easy Difficulty).

Personality: Waller (Mauerer) (Aggressively builds wall traps to disrupt opponent)
Difficulty: Easy (greedy shortest path BFS with 25% random tactical walls, 0-ply search)
Ported from https://wrongway.app/ (ai.js & bot.js).
"""

from collections import deque
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

BOARD_SIZE = 9
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL_CHANCE = 0.25


def _inside(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _edge_open(r1: int, c1: int, r2: int, c2: int,
               h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> bool:
    if not _inside(r1, c1) or not _inside(r2, c2):
        return False
    if abs(r1 - r2) + abs(c1 - c2) != 1:
        return False
    if r1 != r2:
        edge_row = min(r1, r2)
        return (edge_row, c1) not in h_walls and (edge_row, c1 - 1) not in h_walls
    else:
        edge_col = min(c1, c2)
        return (r1, edge_col) not in v_walls and (r1 - 1, edge_col) not in v_walls


def _compute_distance_map(goal_row: int,
                          h_walls: Set[Tuple[int, int]],
                          v_walls: Set[Tuple[int, int]]) -> List[List[int]]:
    dist = [[999] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    q = deque()
    for c in range(BOARD_SIZE):
        dist[goal_row][c] = 0
        q.append((goal_row, c))

    while q:
        r, c = q.popleft()
        d = dist[r][c] + 1
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if _inside(nr, nc) and dist[nr][nc] == 999:
                if _edge_open(r, c, nr, nc, h_walls, v_walls):
                    dist[nr][nc] = d
                    q.append((nr, nc))
    return dist


def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    legal_moves = [tuple(m) for m in observation.get("legal_moves", [])]
    legal_walls = observation.get("legal_walls", [])
    if not legal_moves and not legal_walls:
        return {"type": "pass"}

    my_goal = observation["you"]["goal_row"]
    opp_goal = observation["opponent"]["goal_row"]
    my_walls = observation["you"]["walls_remaining"]
    my_pos = tuple(observation["you"]["position"])
    opp_pos = tuple(observation["opponent"]["position"])

    # Immediate win check
    for m in legal_moves:
        if m[0] == my_goal:
            return {"type": "move", "to": list(m)}

    h_walls = {tuple(w) for w in observation["walls"].get("horizontal", [])}
    v_walls = {tuple(w) for w in observation["walls"].get("vertical", [])}

    my_dist_map = observation.get("distances")
    if not my_dist_map:
        my_dist_map = _compute_distance_map(my_goal, h_walls, v_walls)

    # Occasional wall attempt based on personality
    if my_walls > 0 and legal_walls and random.random() < WALL_CHANCE:
        opp_dist_map = _compute_distance_map(opp_goal, h_walls, v_walls)
        sample_walls = random.sample(legal_walls, min(12, len(legal_walls)))
        for w in sample_walls:
            orient, wr, wc = w["orientation"], w["row"], w["col"]
            # Prefer wall close to opponent
            if abs(wr - opp_pos[0]) <= 2 and abs(wc - opp_pos[1]) <= 2:
                return {"type": "wall", "orientation": orient, "row": wr, "col": wc}

    # Greedy shortest path move
    best_m = legal_moves[0]
    best_d = float("inf")
    for m in legal_moves:
        d = my_dist_map[m[0]][m[1]]
        if 0 <= d < best_d:
            best_d = d
            best_m = m

    return {"type": "move", "to": list(best_m)}
