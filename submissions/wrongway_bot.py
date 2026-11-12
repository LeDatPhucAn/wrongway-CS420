"""Wrongway App Bot: ported from https://wrongway.app/ (bot.js & ai.js).

Features:
- Alpha-Beta / Negamax search with iterative deepening and time-budget guard.
- 6-feature evaluation model from wrongway.app duel9 hard mode:
  botWert(dMe, dOp, wMe, wOp) = g[0] + g[1]*(dOp - dMe) + g[2]*(wMe - wOp) + ...
- Downhill wall candidate generator (botKandidaten): evaluates walls obstructing opponent's shortest path.
- Anti-oscillation penalty (bot.js 'vorher') & forward drive bonus ('vd').
- Jump handling identical to official Quoridor rules and wrongway.app.
- Endgame sprint optimization (rushes to goal when out of walls or close to finish).
- Standard library only; strictly compliant with arena rules.
"""

from collections import deque
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

BOARD_SIZE = 9
INFINITY = 100_000
BOT_WIN = 100_000
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Weights from wrongway.app duel9 hard mode (Mauerer / Waller style):
# [tempo, path_lead, wall_lead, effective_walls, race_endgame, wall_exhaustion]
WEIGHTS = [42.8, 100.0, 141.2, 30.4, 646.2, -21.9]
WALL_BIAS = 20.0
FORWARD_DRIVE = 40.0
BACKTRACK_PENALTY = 60.0

# Track recent positions to penalize oscillation/backtracking (bot.js 'vorher')
_recent_positions: List[Tuple[int, int]] = []
_last_match_reset: float = 0.0


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


def _get_legal_moves(pos: Tuple[int, int],
                     opp_pos: Tuple[int, int],
                     h_walls: Set[Tuple[int, int]],
                     v_walls: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    r, c = pos
    moves = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if not _edge_open(r, c, nr, nc, h_walls, v_walls):
            continue
        if (nr, nc) != opp_pos:
            moves.append((nr, nc))
            continue
        # Opponent adjacent: check jump
        beyond_r, beyond_c = nr + dr, nc + dc
        if _edge_open(nr, nc, beyond_r, beyond_c, h_walls, v_walls):
            moves.append((beyond_r, beyond_c))
        else:
            # Diagonal jump around opponent
            for sdr, sdc in ((dc, dr), (-dc, -dr)):
                side_r, side_c = nr + sdr, nc + sdc
                if _edge_open(nr, nc, side_r, side_c, h_walls, v_walls):
                    moves.append((side_r, side_c))
    return moves


def _evaluate(d_me: int, d_op: int, w_me: int, w_op: int) -> float:
    """Evaluation function ported from bot.js botWert."""
    return (
        WEIGHTS[0]
        + WEIGHTS[1] * (d_op - d_me)
        + WEIGHTS[2] * (w_me - w_op)
        + WEIGHTS[3] * (min(w_me, d_op) - min(w_op, d_me))
        + WEIGHTS[4] * (1.0 if (w_me == 0 and w_op == 0 and d_me <= d_op) else (-1.0 if (w_me == 0 and w_op == 0) else 0.0))
        + WEIGHTS[5] * ((1.0 if w_op == 0 else 0.0) - (1.0 if w_me == 0 else 0.0))
    )


def _wall_conflicts(orient: str, r: int, c: int,
                    h_walls: Set[Tuple[int, int]],
                    v_walls: Set[Tuple[int, int]]) -> bool:
    """Fast check for wall overlap or crossing (botWandFrei)."""
    if orient == "H":
        return ((r, c) in h_walls or
                (r, c - 1) in h_walls or
                (r, c + 1) in h_walls or
                (r, c) in v_walls)
    else:
        return ((r, c) in v_walls or
                (r - 1, c) in v_walls or
                (r + 1, c) in v_walls or
                (r, c) in h_walls)


def _find_candidate_walls(opp_pos: Tuple[int, int],
                          my_pos: Tuple[int, int],
                          opp_goal: int,
                          my_goal: int,
                          opp_dist_map: List[List[int]],
                          my_dist_map: List[List[int]],
                          h_walls: Set[Tuple[int, int]],
                          v_walls: Set[Tuple[int, int]],
                          lookahead: int = 5,
                          max_k: int = 6) -> List[Tuple[Tuple[str, int, int], List[List[int]], List[List[int]]]]:
    """Select candidate walls that intersect opponent's shortest path downhill (botKandidaten)."""
    cur_opp_d = opp_dist_map[opp_pos[0]][opp_pos[1]]
    cur_my_d = my_dist_map[my_pos[0]][my_pos[1]]
    if cur_opp_d <= 0 or cur_opp_d >= 999:
        return []

    # Downhill BFS along shortest path
    proposed_walls: Set[Tuple[str, int, int]] = set()
    q = deque([opp_pos])
    visited = {opp_pos}
    target_dist_limit = max(0, cur_opp_d - lookahead)

    while q:
        cr, cc = q.popleft()
        cd = opp_dist_map[cr][cc]
        if cd <= target_dist_limit or cd == 0:
            continue

        for dr, dc in DIRECTIONS:
            nr, nc = cr + dr, cc + dc
            if _inside(nr, nc) and opp_dist_map[nr][nc] == cd - 1:
                if _edge_open(cr, cc, nr, nc, h_walls, v_walls):
                    if dr == -1:  # Up
                        proposed_walls.add(("H", cr - 1, cc))
                        proposed_walls.add(("H", cr - 1, cc - 1))
                    elif dr == 1:  # Down
                        proposed_walls.add(("H", cr, cc))
                        proposed_walls.add(("H", cr, cc - 1))
                    elif dc == -1:  # Left
                        proposed_walls.add(("V", cr, cc - 1))
                        proposed_walls.add(("V", cr - 1, cc - 1))
                    elif dc == 1:  # Right
                        proposed_walls.add(("V", cr, cc))
                        proposed_walls.add(("V", cr - 1, cc))

                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))

    scored: List[Tuple[int, Tuple[str, int, int], List[List[int]], List[List[int]]]] = []
    for orient, wr, wc in proposed_walls:
        if not (0 <= wr < BOARD_SIZE - 1 and 0 <= wc < BOARD_SIZE - 1):
            continue
        if _wall_conflicts(orient, wr, wc, h_walls, v_walls):
            continue

        if orient == "H":
            h_walls.add((wr, wc))
            n_opp_map = _compute_distance_map(opp_goal, h_walls, v_walls)
            n_opp_d = n_opp_map[opp_pos[0]][opp_pos[1]]
            if n_opp_d >= 999 or n_opp_d - cur_opp_d < 1:
                h_walls.remove((wr, wc))
                continue
            n_my_map = _compute_distance_map(my_goal, h_walls, v_walls)
            n_my_d = n_my_map[my_pos[0]][my_pos[1]]
            h_walls.remove((wr, wc))
        else:
            v_walls.add((wr, wc))
            n_opp_map = _compute_distance_map(opp_goal, h_walls, v_walls)
            n_opp_d = n_opp_map[opp_pos[0]][opp_pos[1]]
            if n_opp_d >= 999 or n_opp_d - cur_opp_d < 1:
                v_walls.remove((wr, wc))
                continue
            n_my_map = _compute_distance_map(my_goal, h_walls, v_walls)
            n_my_d = n_my_map[my_pos[0]][my_pos[1]]
            v_walls.remove((wr, wc))

        if n_my_d >= 999:
            continue

        gain = (n_opp_d - cur_opp_d) - (n_my_d - cur_my_d)
        scored.append((gain, (orient, wr, wc), n_opp_map, n_my_map))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(item[1], item[2], item[3]) for item in scored[:max_k]]


def _negamax(pos_me: Tuple[int, int],
             pos_op: Tuple[int, int],
             goal_me: int,
             goal_op: int,
             walls_me: int,
             walls_op: int,
             dist_me_map: List[List[int]],
             dist_op_map: List[List[int]],
             h_walls: Set[Tuple[int, int]],
             v_walls: Set[Tuple[int, int]],
             depth: int,
             alpha: float,
             beta: float,
             ply: int,
             deadline: float) -> float:
    """Negamax search with Alpha-Beta pruning (botSuche)."""
    if time.perf_counter() > deadline:
        raise TimeoutError()

    # Immediate win check for current player
    legal_moves = _get_legal_moves(pos_me, pos_op, h_walls, v_walls)
    for m in legal_moves:
        if m[0] == goal_me:
            return BOT_WIN - ply

    d_me = dist_me_map[pos_me[0]][pos_me[1]]
    d_op = dist_op_map[pos_op[0]][pos_op[1]]

    if depth <= 0:
        return _evaluate(d_me, d_op, walls_me, walls_op)

    best_score = -INFINITY

    # 1. Pawn moves (sorted closest to goal first)
    legal_moves.sort(key=lambda m: dist_me_map[m[0]][m[1]])
    for m in legal_moves:
        score = -_negamax(pos_op, m, goal_op, goal_me, walls_op, walls_me,
                          dist_op_map, dist_me_map, h_walls, v_walls,
                          depth - 1, -beta, -alpha, ply + 1, deadline)
        if score > best_score:
            best_score = score
        if score > alpha:
            alpha = score
            if alpha >= beta:
                return best_score

    # 2. Candidate walls
    if walls_me > 0 and alpha < beta:
        max_k = 4 if depth <= 1 else 6
        candidate_walls = _find_candidate_walls(
            pos_op, pos_me, goal_op, goal_me,
            dist_op_map, dist_me_map, h_walls, v_walls,
            lookahead=5, max_k=max_k
        )
        for (orient, wr, wc), new_opp_map, new_my_map in candidate_walls:
            if orient == "H":
                h_walls.add((wr, wc))
            else:
                v_walls.add((wr, wc))

            score = -_negamax(pos_op, pos_me, goal_op, goal_me, walls_op, walls_me - 1,
                              new_opp_map, new_my_map, h_walls, v_walls,
                              depth - 1, -beta, -alpha, ply + 1, deadline)

            if orient == "H":
                h_walls.remove((wr, wc))
            else:
                v_walls.remove((wr, wc))

            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    return best_score

    return best_score if best_score != -INFINITY else _evaluate(d_me, d_op, walls_me, walls_op)


def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    global _recent_positions, _last_match_reset
    start_time = time.perf_counter()
    deadline = start_time + 0.85  # safety margin for 1.0s limit

    # Detect new match and reset history
    turn = observation.get("turn", 0)
    now = time.time()
    if turn <= 1 or (now - _last_match_reset > 10.0):
        _recent_positions = []
        _last_match_reset = now

    my_pos = tuple(observation["you"]["position"])
    opp_pos = tuple(observation["opponent"]["position"])
    my_goal = observation["you"]["goal_row"]
    opp_goal = observation["opponent"]["goal_row"]
    my_walls = observation["you"]["walls_remaining"]
    opp_walls = observation["opponent"]["walls_remaining"]

    h_walls = {tuple(w) for w in observation["walls"].get("horizontal", [])}
    v_walls = {tuple(w) for w in observation["walls"].get("vertical", [])}

    legal_moves = [tuple(m) for m in observation.get("legal_moves", [])]
    if not legal_moves and not observation.get("legal_walls"):
        return {"type": "pass"}

    # Immediate win check
    for m in legal_moves:
        if m[0] == my_goal:
            _recent_positions.append(m)
            return {"type": "move", "to": list(m)}

    # Distance maps
    my_dist_map = observation.get("distances")
    if not my_dist_map:
        my_dist_map = _compute_distance_map(my_goal, h_walls, v_walls)
    opp_dist_map = _compute_distance_map(opp_goal, h_walls, v_walls)

    my_dist = my_dist_map[my_pos[0]][my_pos[1]]
    opp_dist = opp_dist_map[opp_pos[0]][opp_pos[1]]

    # Update recent positions (keep last 4)
    if not _recent_positions or _recent_positions[-1] != my_pos:
        _recent_positions.append(my_pos)
    if len(_recent_positions) > 6:
        _recent_positions = _recent_positions[-6:]

    # Endgame sprint optimization: if out of walls or 2 steps from goal and ahead
    if my_walls == 0 or (my_dist <= 2 and my_dist <= opp_dist):
        best_m = legal_moves[0]
        best_d = float("inf")
        for m in legal_moves:
            d = my_dist_map[m[0]][m[1]]
            if 0 <= d < best_d:
                best_d = d
                best_m = m
        _recent_positions.append(best_m)
        return {"type": "move", "to": list(best_m)}

    # Build root candidate moves
    root_moves = []
    for m in legal_moves:
        d_m = my_dist_map[m[0]][m[1]]
        bonus = 0.0
        # Forward drive
        if d_m < my_dist:
            bonus += FORWARD_DRIVE
        # Anti-oscillation penalty (bot.js 'vorher')
        if m in _recent_positions[:-1]:
            idx_from_end = len(_recent_positions) - 1 - _recent_positions.index(m)
            if idx_from_end < 4:
                bonus -= BACKTRACK_PENALTY * (4 - idx_from_end) / 4.0
        root_moves.append({"type": "move", "to": m, "bonus": bonus, "score": 0.0})

    # Build root candidate walls
    root_walls = []
    if my_walls > 0:
        cands = _find_candidate_walls(
            opp_pos, my_pos, opp_goal, my_goal,
            opp_dist_map, my_dist_map, h_walls, v_walls,
            lookahead=6, max_k=8
        )
        for (orient, wr, wc), n_opp_map, n_my_map in cands:
            root_walls.append({
                "type": "wall",
                "orient": orient,
                "row": wr,
                "col": wc,
                "opp_map": n_opp_map,
                "my_map": n_my_map,
                "bonus": WALL_BIAS,
                "score": 0.0
            })

    # Iterative deepening search (Depth 1, 2, 3...)
    best_candidate = None
    all_candidates = root_moves + root_walls

    for target_depth in range(1, 4):
        try:
            current_best = None
            current_best_score = -INFINITY

            for cand in all_candidates:
                if time.perf_counter() > deadline:
                    raise TimeoutError()

                if cand["type"] == "move":
                    m = cand["to"]
                    score = -_negamax(opp_pos, m, opp_goal, my_goal, opp_walls, my_walls,
                                      opp_dist_map, my_dist_map, h_walls, v_walls,
                                      target_depth - 1, -INFINITY, INFINITY, 1, deadline)
                else:
                    orient, wr, wc = cand["orient"], cand["row"], cand["col"]
                    if orient == "H":
                        h_walls.add((wr, wc))
                    else:
                        v_walls.add((wr, wc))

                    score = -_negamax(opp_pos, my_pos, opp_goal, my_goal, opp_walls, my_walls - 1,
                                      cand["opp_map"], cand["my_map"], h_walls, v_walls,
                                      target_depth - 1, -INFINITY, INFINITY, 1, deadline)

                    if orient == "H":
                        h_walls.remove((wr, wc))
                    else:
                        v_walls.remove((wr, wc))

                total_score = score + cand["bonus"]
                cand["score"] = total_score
                if total_score > current_best_score:
                    current_best_score = total_score
                    current_best = cand

            # Completed this depth
            best_candidate = current_best
            all_candidates.sort(key=lambda c: c["score"], reverse=True)

            # If forced win found, stop early
            if current_best_score >= BOT_WIN - 50:
                break

        except TimeoutError:
            # Time limit reached, use best candidate from last completed depth
            break

    # If no candidate scored, fallback to best forward move
    if not best_candidate:
        best_m = legal_moves[0]
        best_d = float("inf")
        for m in legal_moves:
            d = my_dist_map[m[0]][m[1]]
            if 0 <= d < best_d:
                best_d = d
                best_m = m
        _recent_positions.append(best_m)
        return {"type": "move", "to": list(best_m)}

    if best_candidate["type"] == "move":
        _recent_positions.append(best_candidate["to"])
        return {"type": "move", "to": list(best_candidate["to"])}
    else:
        return {
            "type": "wall",
            "orientation": best_candidate["orient"],
            "row": best_candidate["row"],
            "col": best_candidate["col"]
        }
