"""Layer 0 Safeguard Component for Wrongway / Quoridor Arena.

Implements ONLY the highest-priority anti-loss safeguard layer designed to run
standalone or wrap/filter any downstream search agent via
``SafeguardLayer.select_or_filter(observation, candidate_action=None)``.

Pipeline stages:
  S0. Runtime, time-budget & state hygiene guard + authoritative output gate
  S1. Immediate 1-ply win short-circuit (normal steps and jumps)
  S2. Pawn-aware 1-ply loss defense & jump-bridge blunder filter
  S3. 2-ply anti-fork / un-wallable threat horizon guard
  S4. Zero-wall & winning-race sprint guard
  S5. Shortest-path DAG cut, net-gain, conservation & 1-wall self-trap filter
  S6. State-history anti-shuffle & safe pawn move selection

Standard library only; single-file Arena compliant.
"""

from collections import deque
import time
from typing import Any, Dict, List, Optional, Set, Tuple

BOARD_SIZE = 9
WALL_GRID = BOARD_SIZE - 1
INF_DIST = 999
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Tight deadlines inside Arena's 1.0s turn cap
SOFT_DEADLINE_SEC = 0.45
HARD_DEADLINE_SEC = 0.70

Cell = Tuple[int, int]
WallKey = Tuple[str, int, int]
ActionDict = Dict[str, Any]


class _SafeguardTimeout(Exception):
    """Raised when bitwise-periodic deadline checks exceed the hard budget."""


class _Deadline:
    """Fast periodic timer check modeled after quoridors_ace_brfast's _should_stop."""

    __slots__ = ("start", "soft", "hard", "ticks")

    def __init__(self, soft_sec: float = SOFT_DEADLINE_SEC, hard_sec: float = HARD_DEADLINE_SEC) -> None:
        now = time.perf_counter()
        self.start = now
        self.soft = now + soft_sec
        self.hard = now + hard_sec
        self.ticks = 0

    def check(self) -> None:
        self.ticks += 1
        if (self.ticks & 7) == 0 and time.perf_counter() >= self.hard:
            raise _SafeguardTimeout()

    def soft_exceeded(self) -> bool:
        return time.perf_counter() >= self.soft

    def hard_exceeded(self) -> bool:
        return time.perf_counter() >= self.hard


def _inside(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _edge_open(
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    h_walls: Set[Cell],
    v_walls: Set[Cell],
) -> bool:
    """Return True if adjacent squares (r1, c1) and (r2, c2) share an open edge."""
    if not _inside(r1, c1) or not _inside(r2, c2):
        return False
    if abs(r1 - r2) + abs(c1 - c2) != 1:
        return False
    if r1 != r2:
        edge_row = min(r1, r2)
        return (edge_row, c1) not in h_walls and (edge_row, c1 - 1) not in h_walls
    edge_col = min(c1, c2)
    return (r1, edge_col) not in v_walls and (r1 - 1, edge_col) not in v_walls


def _wall_conflicts(
    orient: str,
    wr: int,
    wc: int,
    h_walls: Set[Cell],
    v_walls: Set[Cell],
) -> bool:
    """Return True if wall (orient, wr, wc) overlaps or crosses an existing wall."""
    if not (0 <= wr < WALL_GRID and 0 <= wc < WALL_GRID):
        return True
    if orient == "H":
        return (
            (wr, wc) in h_walls
            or (wr, wc - 1) in h_walls
            or (wr, wc + 1) in h_walls
            or (wr, wc) in v_walls
        )
    return (
        (wr, wc) in v_walls
        or (wr - 1, wc) in v_walls
        or (wr + 1, wc) in v_walls
        or (wr, wc) in h_walls
    )


def _wall_needs_path_check(
    orient: str,
    wr: int,
    wc: int,
    h_walls: Set[Cell],
    v_walls: Set[Cell],
) -> bool:
    """Conservative 5x5 / edge-anchor precheck from quoridors.com engines."""
    if orient == "H" and (wc == 0 or wc == WALL_GRID - 1):
        return True
    if orient == "V" and (wr == 0 or wr == WALL_GRID - 1):
        return True
    nearby = 0
    for dr in range(-2, 3):
        rr = wr + dr
        if not (0 <= rr < WALL_GRID):
            continue
        for dc in range(-2, 3):
            cc = wc + dc
            if not (0 <= cc < WALL_GRID):
                continue
            if (rr, cc) in h_walls or (rr, cc) in v_walls:
                nearby += 1
                if nearby >= 2:
                    return True
    return False


def _has_path(
    start: Cell,
    goal_row: int,
    h_walls: Set[Cell],
    v_walls: Set[Cell],
) -> bool:
    """Fast BFS reachability check from start to goal_row."""
    if start[0] == goal_row:
        return True
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if nxt not in seen and _edge_open(r, c, nr, nc, h_walls, v_walls):
                if nr == goal_row:
                    return True
                seen.add(nxt)
                q.append(nxt)
    return False


def _wall_legal_hypothetical(
    orient: str,
    wr: int,
    wc: int,
    p0_pos: Cell,
    p0_goal: int,
    p1_pos: Cell,
    p1_goal: int,
    h_walls: Set[Cell],
    v_walls: Set[Cell],
) -> bool:
    """Validate wall legality for a hypothetical board state."""
    if _wall_conflicts(orient, wr, wc, h_walls, v_walls):
        return False
    if not _wall_needs_path_check(orient, wr, wc, h_walls, v_walls):
        return True
    target = h_walls if orient == "H" else v_walls
    target.add((wr, wc))
    try:
        return _has_path(p0_pos, p0_goal, h_walls, v_walls) and _has_path(
            p1_pos, p1_goal, h_walls, v_walls
        )
    finally:
        target.remove((wr, wc))


def _compute_distance_map(
    goal_row: int,
    h_walls: Set[Cell],
    v_walls: Set[Cell],
    deadline: Optional[_Deadline] = None,
) -> List[List[int]]:
    """Multi-source BFS distance map from goal_row; unreachable cells hold INF_DIST."""
    dist = [[INF_DIST] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    q = deque()
    for c in range(BOARD_SIZE):
        dist[goal_row][c] = 0
        q.append((goal_row, c))

    while q:
        if deadline is not None:
            deadline.check()
        r, c = q.popleft()
        nd = dist[r][c] + 1
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if _inside(nr, nc) and dist[nr][nc] > nd:
                if _edge_open(r, c, nr, nc, h_walls, v_walls):
                    dist[nr][nc] = nd
                    q.append((nr, nc))
    return dist


def _normalize_distance_map(raw_map: Optional[List[List[int]]]) -> Optional[List[List[int]]]:
    """Convert Arena's -1 unreachable markers to INF_DIST."""
    if not raw_map:
        return None
    return [[INF_DIST if v < 0 else v for v in row] for row in raw_map]


def _get_legal_moves(
    pos: Cell,
    opp_pos: Cell,
    h_walls: Set[Cell],
    v_walls: Set[Cell],
) -> List[Cell]:
    """Generate exact legal pawn moves including straight and diagonal jumps."""
    r, c = pos
    dests: Set[Cell] = set()
    for dr, dc in DIRECTIONS:
        adj = (r + dr, c + dc)
        if not _edge_open(r, c, adj[0], adj[1], h_walls, v_walls):
            continue
        if adj != opp_pos:
            dests.add(adj)
            continue
        beyond = (adj[0] + dr, adj[1] + dc)
        if _edge_open(adj[0], adj[1], beyond[0], beyond[1], h_walls, v_walls):
            dests.add(beyond)
        else:
            for sdr, sdc in ((dc, dr), (-dc, -dr)):
                beside = (adj[0] + sdr, adj[1] + sdc)
                if _edge_open(adj[0], adj[1], beside[0], beside[1], h_walls, v_walls):
                    dests.add(beside)
    return sorted(dests)


def _dag_cut_wall_set(
    start: Cell,
    dist_map: List[List[int]],
    h_walls: Set[Cell],
    v_walls: Set[Cell],
) -> Set[WallKey]:
    """Return every wall placement (orient, r, c) cutting at least one shortest-path DAG edge."""
    d0 = dist_map[start[0]][start[1]]
    if d0 <= 0 or d0 >= INF_DIST:
        return set()

    cuts: Set[WallKey] = set()
    seen = {start}
    q = deque([start])

    while q:
        r, c = q.popleft()
        cur_d = dist_map[r][c]
        if cur_d <= 0:
            continue
        target = cur_d - 1
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not _inside(nr, nc) or dist_map[nr][nc] != target:
                continue
            if not _edge_open(r, c, nr, nc, h_walls, v_walls):
                continue
            if dr != 0:
                edge_r = min(r, nr)
                for wc in (c, c - 1):
                    if 0 <= wc < WALL_GRID:
                        cuts.add(("H", edge_r, wc))
            else:
                edge_c = min(c, nc)
                for wr in (r, r - 1):
                    if 0 <= wr < WALL_GRID:
                        cuts.add(("V", wr, edge_c))
            nxt = (nr, nc)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return cuts


def _local_wall_slots(cell_a: Cell, cell_b: Optional[Cell] = None) -> Set[Cell]:
    """Return wall anchor coordinates (wr, wc) touching edges adjacent to cell_a (and cell_b)."""
    anchors: Set[Cell] = set()
    cells = (cell_a,) if cell_b is None else (cell_a, cell_b)
    for r, c in cells:
        for wr in (r - 1, r):
            if 0 <= wr < WALL_GRID:
                for wc in (c - 1, c):
                    if 0 <= wc < WALL_GRID:
                        anchors.add((wr, wc))
    return anchors


def _wall_to_action(wall: WallKey) -> ActionDict:
    return {"type": "wall", "orientation": wall[0], "row": wall[1], "col": wall[2]}


def _move_to_action(cell: Cell) -> ActionDict:
    return {"type": "move", "to": [cell[0], cell[1]]}


def _action_to_wall_key(action: Optional[ActionDict]) -> Optional[WallKey]:
    if not isinstance(action, dict) or action.get("type") != "wall":
        return None
    ori = action.get("orientation")
    r, c = action.get("row"), action.get("col")
    if ori in ("H", "V") and type(r) is int and type(c) is int:
        return (ori, r, c)
    return None


def _action_to_move_cell(action: Optional[ActionDict]) -> Optional[Cell]:
    if not isinstance(action, dict) or action.get("type") != "move":
        return None
    to = action.get("to")
    if isinstance(to, (list, tuple)) and len(to) == 2 and type(to[0]) is int and type(to[1]) is int:
        return (to[0], to[1])
    return None


class SafeguardLayer:
    """Highest-priority anti-loss layer (Layer 0) for Wrongway / Quoridor agents."""

    def __init__(self) -> None:
        self._visited_states: Set[Tuple[Any, ...]] = set()
        self._recent_positions: List[Cell] = []
        self._last_wall_sum: int = 20
        self._last_call_time: float = 0.0

    def _reset_if_new_game(
        self,
        turn: int,
        my_walls: int,
        opp_walls: int,
        my_pos: Cell,
        opp_pos: Cell,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
    ) -> None:
        now = time.perf_counter()
        wall_sum = my_walls + opp_walls
        if (
            turn <= 1
            or wall_sum > self._last_wall_sum
            or (self._last_call_time > 0.0 and (now - self._last_call_time) > 10.0)
        ):
            self._visited_states.clear()
            self._recent_positions.clear()
        self._last_wall_sum = wall_sum
        self._last_call_time = now

        state_key = (my_pos, opp_pos, tuple(sorted(h_walls)), tuple(sorted(v_walls)))
        self._visited_states.add(state_key)
        if not self._recent_positions or self._recent_positions[-1] != my_pos:
            self._recent_positions.append(my_pos)
            if len(self._recent_positions) > 6:
                self._recent_positions = self._recent_positions[-6:]

    @staticmethod
    def _emergency_fallback(
        legal_moves: List[Cell],
        legal_walls: List[WallKey],
        my_dist_map: List[List[int]],
    ) -> ActionDict:
        if legal_moves:
            best = min(
                legal_moves,
                key=lambda m: (my_dist_map[m[0]][m[1]], abs(m[1] - 4), m[0], m[1]),
            )
            return _move_to_action(best)
        if legal_walls:
            return _wall_to_action(legal_walls[0])
        return {"type": "pass"}

    @staticmethod
    def _sanitize_action(
        action: Optional[ActionDict],
        legal_move_set: Set[Cell],
        legal_wall_set: Set[WallKey],
        fallback: ActionDict,
    ) -> ActionDict:
        if not isinstance(action, dict):
            return fallback
        atype = action.get("type")
        if atype == "move":
            cell = _action_to_move_cell(action)
            if cell is not None and cell in legal_move_set:
                return _move_to_action(cell)
            return fallback
        if atype == "wall":
            wkey = _action_to_wall_key(action)
            if wkey is not None and wkey in legal_wall_set:
                return _wall_to_action(wkey)
            return fallback
        if atype == "pass" and not legal_move_set and not legal_wall_set:
            return {"type": "pass"}
        return fallback

    @staticmethod
    def _opp_immediate_wins(
        opp_pos: Cell,
        opp_goal: int,
        my_pos: Cell,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
    ) -> List[Cell]:
        return [
            om
            for om in _get_legal_moves(opp_pos, my_pos, h_walls, v_walls)
            if om[0] == opp_goal
        ]

    def _is_unwallable_at_dist_1(
        self,
        opp_cell: Cell,
        opp_goal: int,
        my_pos: Cell,
        my_goal: int,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
    ) -> bool:
        """Return True if no single legal wall can prevent opp_cell from reaching opp_goal."""
        if not self._opp_immediate_wins(opp_cell, opp_goal, my_pos, h_walls, v_walls):
            return False
        for wr, wc in _local_wall_slots(opp_cell, my_pos):
            for orient in ("H", "V"):
                deadline.check()
                if not _wall_legal_hypothetical(
                    orient, wr, wc, my_pos, my_goal, opp_cell, opp_goal, h_walls, v_walls
                ):
                    continue
                target = h_walls if orient == "H" else v_walls
                target.add((wr, wc))
                try:
                    still_wins = bool(
                        self._opp_immediate_wins(opp_cell, opp_goal, my_pos, h_walls, v_walls)
                    )
                finally:
                    target.remove((wr, wc))
                if not still_wins:
                    return False
        return True

    def _max_opp_1wall_delay(
        self,
        my_pos: Cell,
        my_goal: int,
        my_dist_map: List[List[int]],
        opp_pos: Cell,
        opp_goal: int,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
        limit: int = 24,
    ) -> int:
        """Maximum shortest-path delay one opponent wall cutting our DAG can inflict."""
        cur_d = my_dist_map[my_pos[0]][my_pos[1]]
        if cur_d <= 0 or cur_d >= INF_DIST:
            return 0
        dag_walls = _dag_cut_wall_set(my_pos, my_dist_map, h_walls, v_walls)
        if not dag_walls:
            return 0
        worst_delay = 0
        checked = 0
        for orient, wr, wc in dag_walls:
            if checked >= limit or deadline.soft_exceeded():
                break
            deadline.check()
            if not _wall_legal_hypothetical(
                orient, wr, wc, my_pos, my_goal, opp_pos, opp_goal, h_walls, v_walls
            ):
                continue
            checked += 1
            target = h_walls if orient == "H" else v_walls
            target.add((wr, wc))
            try:
                nd_map = _compute_distance_map(my_goal, h_walls, v_walls, deadline)
                nd = nd_map[my_pos[0]][my_pos[1]]
                if nd < INF_DIST:
                    delay = nd - cur_d
                    if delay > worst_delay:
                        worst_delay = delay
            finally:
                target.remove((wr, wc))
        return worst_delay

    def _evaluate_wall_candidate(
        self,
        wkey: WallKey,
        my_pos: Cell,
        my_goal: int,
        my_dist: int,
        opp_pos: Cell,
        opp_goal: int,
        opp_dist: int,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
    ) -> Tuple[int, int, int, int, List[List[int]]]:
        """Place wall temporarily and compute (gain, op_delta, my_delta, new_opp_dist, new_my_map)."""
        orient, wr, wc = wkey
        target = h_walls if orient == "H" else v_walls
        target.add((wr, wc))
        try:
            new_opp_map = _compute_distance_map(opp_goal, h_walls, v_walls, deadline)
            new_my_map = _compute_distance_map(my_goal, h_walls, v_walls, deadline)
            new_opp_d = new_opp_map[opp_pos[0]][opp_pos[1]]
            new_my_d = new_my_map[my_pos[0]][my_pos[1]]
        finally:
            target.remove((wr, wc))
        op_delta = new_opp_d - opp_dist
        my_delta = new_my_d - my_dist
        gain = op_delta - my_delta
        return gain, op_delta, my_delta, new_opp_d, new_my_map

    def _find_emergency_1ply_block(
        self,
        my_pos: Cell,
        my_goal: int,
        my_dist: int,
        opp_pos: Cell,
        opp_goal: int,
        opp_dist: int,
        legal_wall_set: Set[WallKey],
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
    ) -> Optional[WallKey]:
        """Stage S2: find best legal wall eliminating all 1-ply opponent wins."""
        anchors = _local_wall_slots(opp_pos, my_pos)
        best_wall: Optional[WallKey] = None
        best_rank: Optional[Tuple[int, int, int, int]] = None

        for wr, wc in anchors:
            for orient in ("H", "V"):
                deadline.check()
                wkey = (orient, wr, wc)
                if wkey not in legal_wall_set:
                    continue
                target = h_walls if orient == "H" else v_walls
                target.add((wr, wc))
                try:
                    still_wins = bool(
                        self._opp_immediate_wins(opp_pos, opp_goal, my_pos, h_walls, v_walls)
                    )
                finally:
                    target.remove((wr, wc))
                if still_wins:
                    continue
                gain, op_delta, my_delta, new_opp_d, _ = self._evaluate_wall_candidate(
                    wkey, my_pos, my_goal, my_dist, opp_pos, opp_goal, opp_dist, h_walls, v_walls, deadline
                )
                if new_opp_d >= INF_DIST or my_dist + my_delta >= INF_DIST:
                    continue
                rank = (new_opp_d - (my_dist + my_delta), gain, op_delta, -abs(wc - 4))
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_wall = wkey
        return best_wall

    def _find_2ply_unwallable_threats(
        self,
        my_pos: Cell,
        my_goal: int,
        opp_pos: Cell,
        opp_goal: int,
        opp_dist_map: List[List[int]],
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
    ) -> List[Cell]:
        """Stage S3: opponent next moves reaching dist==1 that would then be un-wallable."""
        threats: List[Cell] = []
        for u in _get_legal_moves(opp_pos, my_pos, h_walls, v_walls):
            deadline.check()
            if opp_dist_map[u[0]][u[1]] != 1:
                continue
            if self._is_unwallable_at_dist_1(u, opp_goal, my_pos, my_goal, h_walls, v_walls, deadline):
                threats.append(u)
        return threats

    def _find_antifork_block(
        self,
        my_pos: Cell,
        my_goal: int,
        my_dist: int,
        opp_pos: Cell,
        opp_goal: int,
        opp_dist: int,
        legal_wall_set: Set[WallKey],
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
    ) -> Optional[WallKey]:
        """Stage S3: pre-emptive wall at opp_dist==2 eliminating all un-wallable dist==1 threats."""
        anchors = _local_wall_slots(opp_pos, my_pos)
        best_wall: Optional[WallKey] = None
        best_rank: Optional[Tuple[int, int, int, int]] = None

        for wr, wc in anchors:
            for orient in ("H", "V"):
                deadline.check()
                wkey = (orient, wr, wc)
                if wkey not in legal_wall_set:
                    continue
                target = h_walls if orient == "H" else v_walls
                target.add((wr, wc))
                try:
                    new_opp_map = _compute_distance_map(opp_goal, h_walls, v_walls, deadline)
                    new_my_map = _compute_distance_map(my_goal, h_walls, v_walls, deadline)
                    new_opp_d = new_opp_map[opp_pos[0]][opp_pos[1]]
                    new_my_d = new_my_map[my_pos[0]][my_pos[1]]
                    if new_opp_d >= INF_DIST or new_my_d >= INF_DIST:
                        continue
                    if self._opp_immediate_wins(opp_pos, opp_goal, my_pos, h_walls, v_walls):
                        continue
                    rem_threats = self._find_2ply_unwallable_threats(
                        my_pos, my_goal, opp_pos, opp_goal, new_opp_map, h_walls, v_walls, deadline
                    )
                finally:
                    target.remove((wr, wc))
                if rem_threats:
                    continue
                op_delta = new_opp_d - opp_dist
                my_delta = new_my_d - my_dist
                gain = op_delta - my_delta
                if gain < 0 or op_delta < 0:
                    continue
                rank = (new_opp_d - new_my_d, gain, op_delta, -abs(wc - 4))
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_wall = wkey
        return best_wall

    def _is_safe_s5_wall(
        self,
        wkey: WallKey,
        my_pos: Cell,
        my_goal: int,
        my_dist: int,
        my_walls: int,
        opp_pos: Cell,
        opp_goal: int,
        opp_dist: int,
        opp_walls: int,
        opp_dag_walls: Set[WallKey],
        base_self_trap: int,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Stage S5 validation returning (race_lead, gain, op_delta, my_delta) or None."""
        if wkey not in opp_dag_walls:
            return None
        gain, op_delta, my_delta, new_opp_d, new_my_map = self._evaluate_wall_candidate(
            wkey, my_pos, my_goal, my_dist, opp_pos, opp_goal, opp_dist, h_walls, v_walls, deadline
        )
        new_my_d = my_dist + my_delta
        if new_opp_d >= INF_DIST or new_my_d >= INF_DIST:
            return None
        if op_delta < 1 or gain < 1:
            return None
        if my_walls <= 2 and opp_dist > 3 and gain < 2:
            return None
        if my_walls == 1 and new_my_d > new_opp_d and opp_dist > 2:
            return None
        if opp_walls > 0 and not deadline.soft_exceeded():
            orient, wr, wc = wkey
            target = h_walls if orient == "H" else v_walls
            target.add((wr, wc))
            try:
                trap_delay = self._max_opp_1wall_delay(
                    my_pos, my_goal, new_my_map, opp_pos, opp_goal, h_walls, v_walls, deadline, limit=16
                )
            finally:
                target.remove((wr, wc))
            if trap_delay >= 4 and trap_delay > base_self_trap:
                return None
        return (new_opp_d - new_my_d, gain, op_delta, -my_delta)

    def _choose_best_safe_wall(
        self,
        my_pos: Cell,
        my_goal: int,
        my_dist: int,
        my_walls: int,
        opp_pos: Cell,
        opp_goal: int,
        opp_dist: int,
        opp_walls: int,
        opp_dag_walls: Set[WallKey],
        legal_wall_set: Set[WallKey],
        base_self_trap: int,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
    ) -> Optional[WallKey]:
        """Select highest-value Stage S5 safe defensive wall from opponent's shortest-path DAG."""
        candidates = [w for w in opp_dag_walls if w in legal_wall_set]
        # Evaluate slots closer to opponent first so soft deadline keeps highest-leverage walls
        candidates.sort(key=lambda w: (abs(w[1] - opp_pos[0]) + abs(w[2] - opp_pos[1]), w[0], w[1], w[2]))

        best_wall: Optional[WallKey] = None
        best_rank: Optional[Tuple[int, int, int, int, int]] = None
        for wkey in candidates:
            if deadline.soft_exceeded():
                break
            deadline.check()
            metrics = self._is_safe_s5_wall(
                wkey,
                my_pos,
                my_goal,
                my_dist,
                my_walls,
                opp_pos,
                opp_goal,
                opp_dist,
                opp_walls,
                opp_dag_walls,
                base_self_trap,
                h_walls,
                v_walls,
                deadline,
            )
            if metrics is None:
                continue
            race_lead, gain, op_delta, neg_my_delta = metrics
            rank = (race_lead, gain, op_delta, neg_my_delta, -abs(wkey[2] - 4))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_wall = wkey
        return best_wall

    def _choose_safe_pawn_move(
        self,
        pool: List[Cell],
        my_pos: Cell,
        my_goal: int,
        my_dist: int,
        my_dist_map: List[List[int]],
        opp_pos: Cell,
        opp_goal: int,
        opp_walls: int,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        deadline: _Deadline,
        base_self_trap: int = 0,
    ) -> Cell:
        """Stage S6: choose shortest-path safe move avoiding repeats, traps & oscillation."""
        h_tuple = tuple(sorted(h_walls))
        v_tuple = tuple(sorted(v_walls))

        scored: List[Tuple[Tuple[int, int, int, int, int, int], Cell]] = []
        for m in pool:
            deadline.check()
            dm = my_dist_map[m[0]][m[1]]
            state_key = (m, opp_pos, h_tuple, v_tuple)
            repeats_state = 1 if state_key in self._visited_states else 0
            repeats_recent = 1 if m in self._recent_positions[:-1] else 0

            trap_penalty = 0
            if opp_walls > 0 and len(pool) > 1 and not deadline.soft_exceeded():
                trap_delay = self._max_opp_1wall_delay(
                    m, my_goal, my_dist_map, opp_pos, opp_goal, h_walls, v_walls, deadline, limit=12
                )
                if trap_delay >= 4 and trap_delay > base_self_trap and dm >= my_dist - 1:
                    trap_penalty = 1

            # Reward jumps that skip over the opponent toward goal
            step_span = abs(m[0] - my_pos[0]) + abs(m[1] - my_pos[1])
            jump_bonus = -1 if (step_span == 2 and dm < my_dist) else 0
            rank = (
                dm + 2 * repeats_state + repeats_recent + trap_penalty,
                repeats_state,
                repeats_recent,
                jump_bonus,
                abs(m[1] - 4),
                m[0],
            )
            scored.append((rank, m))

        scored.sort(key=lambda item: item[0])
        best_move = scored[0][1]

        # Strict anti-shuffle guard (like quoridors_expert._anti_shuffle):
        # if best_move repeats a visited state, substitute best non-repeating move
        # that does not worsen distance beyond current my_dist.
        best_state = (best_move, opp_pos, h_tuple, v_tuple)
        if best_state in self._visited_states and len(pool) > 1:
            non_repeating = [
                (my_dist_map[m[0]][m[1]], abs(m[1] - 4), m)
                for m in pool
                if (m, opp_pos, h_tuple, v_tuple) not in self._visited_states
                and my_dist_map[m[0]][m[1]] <= max(my_dist, my_dist_map[best_move[0]][best_move[1]] + 1)
            ]
            if non_repeating:
                non_repeating.sort(key=lambda x: (x[0], x[1], x[2]))
                return non_repeating[0][2]

        return best_move

    def select_or_filter(
        self,
        observation: Dict[str, Any],
        candidate_action: Optional[ActionDict] = None,
    ) -> ActionDict:
        """Execute Safeguard Layer S0-S6 standalone or filter a candidate_action."""
        deadline = _Deadline()

        if observation.get("done"):
            return {"type": "pass"}

        raw_moves = observation.get("legal_moves") or []
        raw_walls = observation.get("legal_walls") or []
        if not raw_moves and not raw_walls:
            return {"type": "pass"}

        legal_moves: List[Cell] = [(int(m[0]), int(m[1])) for m in raw_moves]
        legal_move_set = set(legal_moves)
        legal_walls: List[WallKey] = [
            (w["orientation"], int(w["row"]), int(w["col"])) for w in raw_walls
        ]
        legal_wall_set = set(legal_walls)

        you = observation["you"]
        opp = observation["opponent"]
        my_pos: Cell = (int(you["position"][0]), int(you["position"][1]))
        opp_pos: Cell = (int(opp["position"][0]), int(opp["position"][1]))
        my_goal = int(you["goal_row"])
        opp_goal = int(opp["goal_row"])
        my_walls = int(you["walls_remaining"])
        opp_walls = int(opp["walls_remaining"])

        h_walls: Set[Cell] = {(int(w[0]), int(w[1])) for w in observation["walls"].get("horizontal", [])}
        v_walls: Set[Cell] = {(int(w[0]), int(w[1])) for w in observation["walls"].get("vertical", [])}

        # Stage S0: Reset state hygiene & precompute emergency fallback
        self._reset_if_new_game(
            int(observation.get("turn", 0)),
            my_walls,
            opp_walls,
            my_pos,
            opp_pos,
            h_walls,
            v_walls,
        )

        my_dist_map = _normalize_distance_map(observation.get("distances"))
        if my_dist_map is None:
            my_dist_map = _compute_distance_map(my_goal, h_walls, v_walls)
        fallback = self._emergency_fallback(legal_moves, legal_walls, my_dist_map)

        # Stage S1: Immediate 1-ply win short-circuit
        for m in legal_moves:
            if m[0] == my_goal:
                return _move_to_action(m)

        try:
            chosen = self._select_raw(
                deadline=deadline,
                legal_moves=legal_moves,
                legal_move_set=legal_move_set,
                legal_wall_set=legal_wall_set,
                my_pos=my_pos,
                opp_pos=opp_pos,
                my_goal=my_goal,
                opp_goal=opp_goal,
                my_walls=my_walls,
                opp_walls=opp_walls,
                h_walls=h_walls,
                v_walls=v_walls,
                my_dist_map=my_dist_map,
                candidate_action=candidate_action,
            )
            return self._sanitize_action(chosen, legal_move_set, legal_wall_set, fallback)
        except _SafeguardTimeout:
            return fallback

    def _select_raw(
        self,
        *,
        deadline: _Deadline,
        legal_moves: List[Cell],
        legal_move_set: Set[Cell],
        legal_wall_set: Set[WallKey],
        my_pos: Cell,
        opp_pos: Cell,
        my_goal: int,
        opp_goal: int,
        my_walls: int,
        opp_walls: int,
        h_walls: Set[Cell],
        v_walls: Set[Cell],
        my_dist_map: List[List[int]],
        candidate_action: Optional[ActionDict],
    ) -> ActionDict:
        """Internal S2-S6 decision pipeline guarded by S0 deadline & sanitizer."""
        opp_dist_map = _compute_distance_map(opp_goal, h_walls, v_walls, deadline)
        my_dist = my_dist_map[my_pos[0]][my_pos[1]]
        opp_dist = opp_dist_map[opp_pos[0]][opp_pos[1]]

        # Stage S2: Pawn-aware 1-ply opponent win & jump-bridge blunder filter
        opp_wins_now = self._opp_immediate_wins(opp_pos, opp_goal, my_pos, h_walls, v_walls)
        safe_pawn_moves: List[Cell] = [
            m
            for m in legal_moves
            if not self._opp_immediate_wins(opp_pos, opp_goal, m, h_walls, v_walls)
        ]
        pawn_pool = safe_pawn_moves if safe_pawn_moves else legal_moves

        cand_move = _action_to_move_cell(candidate_action)
        cand_wall = _action_to_wall_key(candidate_action)
        if cand_move is not None and cand_move not in legal_move_set:
            cand_move = None
        if cand_wall is not None and cand_wall not in legal_wall_set:
            cand_wall = None

        base_self_trap = 0
        if opp_walls > 0 and not deadline.soft_exceeded():
            base_self_trap = self._max_opp_1wall_delay(
                my_pos, my_goal, my_dist_map, opp_pos, opp_goal, h_walls, v_walls, deadline, limit=12
            )

        if opp_wins_now:
            if cand_move is not None and cand_move in safe_pawn_moves and opp_dist >= 2:
                return _move_to_action(cand_move)
            if cand_wall is not None:
                target = h_walls if cand_wall[0] == "H" else v_walls
                target.add((cand_wall[1], cand_wall[2]))
                try:
                    cand_stops_loss = not self._opp_immediate_wins(
                        opp_pos, opp_goal, my_pos, h_walls, v_walls
                    )
                finally:
                    target.remove((cand_wall[1], cand_wall[2]))
                if cand_stops_loss:
                    return _wall_to_action(cand_wall)

            # If threat was solely a jump over our pawn, step safely out of the bridge for 0 walls
            if opp_dist >= 2 and safe_pawn_moves:
                best_safe = self._choose_safe_pawn_move(
                    safe_pawn_moves,
                    my_pos,
                    my_goal,
                    my_dist,
                    my_dist_map,
                    opp_pos,
                    opp_goal,
                    opp_walls,
                    h_walls,
                    v_walls,
                    deadline,
                    base_self_trap,
                )
                return _move_to_action(best_safe)

            if my_walls > 0:
                em_wall = self._find_emergency_1ply_block(
                    my_pos,
                    my_goal,
                    my_dist,
                    opp_pos,
                    opp_goal,
                    opp_dist,
                    legal_wall_set,
                    h_walls,
                    v_walls,
                    deadline,
                )
                if em_wall is not None:
                    return _wall_to_action(em_wall)

            if safe_pawn_moves:
                best_safe = self._choose_safe_pawn_move(
                    safe_pawn_moves,
                    my_pos,
                    my_goal,
                    my_dist,
                    my_dist_map,
                    opp_pos,
                    opp_goal,
                    opp_walls,
                    h_walls,
                    v_walls,
                    deadline,
                    base_self_trap,
                )
                return _move_to_action(best_safe)

        # Stage S3: 2-ply anti-fork horizon guard (un-wallable dist==1 threats)
        if opp_dist == 2 and my_dist > 1 and my_walls > 0 and not deadline.soft_exceeded():
            unwallable_threats = self._find_2ply_unwallable_threats(
                my_pos, my_goal, opp_pos, opp_goal, opp_dist_map, h_walls, v_walls, deadline
            )
            if unwallable_threats:
                if cand_wall is not None:
                    target = h_walls if cand_wall[0] == "H" else v_walls
                    target.add((cand_wall[1], cand_wall[2]))
                    try:
                        n_opp_map = _compute_distance_map(opp_goal, h_walls, v_walls, deadline)
                        still_threat = bool(
                            self._find_2ply_unwallable_threats(
                                my_pos, my_goal, opp_pos, opp_goal, n_opp_map, h_walls, v_walls, deadline
                            )
                        )
                    finally:
                        target.remove((cand_wall[1], cand_wall[2]))
                    if not still_threat:
                        return _wall_to_action(cand_wall)

                preemptive = self._find_antifork_block(
                    my_pos,
                    my_goal,
                    my_dist,
                    opp_pos,
                    opp_goal,
                    opp_dist,
                    legal_wall_set,
                    h_walls,
                    v_walls,
                    deadline,
                )
                if preemptive is not None:
                    return _wall_to_action(preemptive)

        # Stage S4: Zero-wall & winning-race sprint guard
        must_sprint = (
            my_walls == 0
            or (opp_walls == 0 and my_dist <= opp_dist)
            or (my_dist <= 2 and my_dist <= opp_dist)
        )
        if must_sprint:
            if cand_move is not None and cand_move in pawn_pool:
                cand_d = my_dist_map[cand_move[0]][cand_move[1]]
                best_d = min(my_dist_map[m[0]][m[1]] for m in pawn_pool)
                if cand_d == best_d:
                    state_key = (cand_move, opp_pos, tuple(sorted(h_walls)), tuple(sorted(v_walls)))
                    if state_key not in self._visited_states:
                        return _move_to_action(cand_move)
            best_sprint = self._choose_safe_pawn_move(
                pawn_pool,
                my_pos,
                my_goal,
                my_dist,
                my_dist_map,
                opp_pos,
                opp_goal,
                opp_walls,
                h_walls,
                v_walls,
                deadline,
                base_self_trap,
            )
            return _move_to_action(best_sprint)

        # Stage S5: Shortest-path DAG & self-harm wall safeguard
        opp_dag_walls = _dag_cut_wall_set(opp_pos, opp_dist_map, h_walls, v_walls)

        if cand_wall is not None:
            metrics = self._is_safe_s5_wall(
                cand_wall,
                my_pos,
                my_goal,
                my_dist,
                my_walls,
                opp_pos,
                opp_goal,
                opp_dist,
                opp_walls,
                opp_dag_walls,
                base_self_trap,
                h_walls,
                v_walls,
                deadline,
            )
            if metrics is not None:
                return _wall_to_action(cand_wall)
            if opp_dist < my_dist:
                alt_wall = self._choose_best_safe_wall(
                    my_pos,
                    my_goal,
                    my_dist,
                    my_walls,
                    opp_pos,
                    opp_goal,
                    opp_dist,
                    opp_walls,
                    opp_dag_walls,
                    legal_wall_set,
                    base_self_trap,
                    h_walls,
                    v_walls,
                    deadline,
                )
                if alt_wall is not None:
                    return _wall_to_action(alt_wall)

        elif cand_move is not None:
            if cand_move in pawn_pool:
                h_tuple = tuple(sorted(h_walls))
                v_tuple = tuple(sorted(v_walls))
                if (cand_move, opp_pos, h_tuple, v_tuple) not in self._visited_states:
                    return _move_to_action(cand_move)

        else:
            # Standalone Layer 0: wall defensively when strictly behind, or when tied near goal with >=2 net gain
            if opp_dist < my_dist or (opp_dist <= 3 and opp_dist == my_dist):
                defensive_wall = self._choose_best_safe_wall(
                    my_pos,
                    my_goal,
                    my_dist,
                    my_walls,
                    opp_pos,
                    opp_goal,
                    opp_dist,
                    opp_walls,
                    opp_dag_walls,
                    legal_wall_set,
                    base_self_trap,
                    h_walls,
                    v_walls,
                    deadline,
                )
                if defensive_wall is not None:
                    if opp_dist < my_dist:
                        return _wall_to_action(defensive_wall)
                    gain, _, _, _, _ = self._evaluate_wall_candidate(
                        defensive_wall, my_pos, my_goal, my_dist, opp_pos, opp_goal, opp_dist, h_walls, v_walls, deadline
                    )
                    if gain >= 2:
                        return _wall_to_action(defensive_wall)

        # Stage S6: Safe, anti-shuffle pawn selection
        best_move = self._choose_safe_pawn_move(
            pawn_pool,
            my_pos,
            my_goal,
            my_dist,
            my_dist_map,
            opp_pos,
            opp_goal,
            opp_walls,
            h_walls,
            v_walls,
            deadline,
            base_self_trap,
        )
        return _move_to_action(best_move)


_SAFEGUARD = SafeguardLayer()


def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Arena entry point running standalone Layer 0 Safeguard decisions."""
    return _SAFEGUARD.select_or_filter(observation, candidate_action=None)
