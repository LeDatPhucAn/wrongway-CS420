"""Pure-Python port of quoridors.com's QSeriesBot Expert ("Strategist").

Faithfully implements the client-side engine from ``quoridors.com``:
- Rules/board/Zobrist core (``js/ace/engine-core.js``): blocked-edge 9x9 board,
  incremental 64-bit Zobrist hash, wall-fit & conservative anchor precheck,
  full jump generation, and multi-source goal BFS distance fields.
- Search engine (``js/qseries/qseries-search.js``, ``LEVELS.expert``): iterative
  deepening alpha-beta minimax (up to depth 10, LMR at depth >= 4 after move 4,
  2-slot killer table, history heuristic, 300k transposition table),
  opponent-path focus + wall-fight template candidate generation (capped at 24),
  shortest-path DAG cut walls when opponent is out of walls, phase-weighted
  shortest-path heuristic, exact zero-wall race verdict, and anti-shuffle guard.

Adapted for Wrongway Arena constraints: stateless-per-call reconstruction from
``observation`` (with persistent module-level anti-shuffle position history reset
on new games) and an internal 0.72s budget inside the 1.0s turn cap.
"""

import time
from typing import Dict, List, Optional, Set, Tuple

BOARD_SIZE = 9
CELL_COUNT = 81
WALL_SLOTS = 64
H_BASE = 100
V_BASE = 200
INF = 1e9

# Direction indices: 0=N (-9), 1=S (+9), 2=W (-1), 3=E (+1)
DELTA = (-9, 9, -1, 1)
DIRBIT = (1, 2, 4, 8)

# Expert config (qseries-search.js LEVELS.expert, budgeted for 1.0s Arena cap)
MAX_DEPTH = 10
WALL_CAP = 24
WALL_FLOOR = 2
LMR_MIN_DEPTH = 4
LMR_AFTER = 4
W_WALLS_BASE = 0.05
W_WALLS_PHASE = 0.15
NODE_BUDGET = 1_200_000
TIME_BUDGET_SEC = 0.72

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
TT_MAX = 300_000
DIST_CACHE_MAX = 16_384

# ---------- Static Precomputed Tables ----------

CELL_ROW = tuple(c // 9 for c in range(CELL_COUNT))
CELL_COL = tuple(c % 9 for c in range(CELL_COUNT))
SLOT_ROW = tuple(s // 8 for s in range(WALL_SLOTS))
SLOT_COL = tuple(s % 8 for s in range(WALL_SLOTS))

BORDER = tuple(
    (1 if (c // 9) == 0 else 0)
    | (2 if (c // 9) == 8 else 0)
    | (4 if (c % 9) == 0 else 0)
    | (8 if (c % 9) == 8 else 0)
    for c in range(CELL_COUNT)
)

GOAL_ROW0_CELLS = tuple(range(0, 9))
GOAL_ROW8_CELLS = tuple(range(72, 81))


def _build_wall_masks():
    h_overlap = [0] * WALL_SLOTS
    v_overlap = [0] * WALL_SLOTS
    anchor_5x5 = [0] * WALL_SLOTS
    h_edge_anchor = [0] * WALL_SLOTS
    v_edge_anchor = [0] * WALL_SLOTS
    hw_tpl_h = [0] * WALL_SLOTS
    hw_tpl_v = [0] * WALL_SLOTS
    vw_tpl_h = [0] * WALL_SLOTS
    vw_tpl_v = [0] * WALL_SLOTS

    def bit(r: int, c: int) -> int:
        if 0 <= r <= 7 and 0 <= c <= 7:
            return 1 << (r * 8 + c)
        return 0

    for s in range(WALL_SLOTS):
        r, c = s // 8, s % 8
        hm = 1 << s
        if c > 0:
            hm |= 1 << (s - 1)
        if c < 7:
            hm |= 1 << (s + 1)
        h_overlap[s] = hm

        vm = 1 << s
        if r > 0:
            vm |= 1 << (s - 8)
        if r < 7:
            vm |= 1 << (s + 8)
        v_overlap[s] = vm

        am = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                am |= bit(r + dr, c + dc)
        anchor_5x5[s] = am

        h_edge_anchor[s] = 1 if (c == 0 or c == 7) else 0
        v_edge_anchor[s] = 1 if (r == 0 or r == 7) else 0

        # Template neighbors for placed H wall
        th = bit(r, c - 2) | bit(r, c + 2) | bit(r, c - 3) | bit(r, c + 3)
        tv = bit(r, c - 2) | bit(r, c + 2)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                tv |= bit(r + dr, c + dc)
        hw_tpl_h[s] = th
        hw_tpl_v[s] = tv

        # Template neighbors for placed V wall
        vh = bit(r - 2, c) | bit(r + 2, c)
        vv = bit(r - 2, c) | bit(r + 2, c) | bit(r - 3, c) | bit(r + 3, c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                vh |= bit(r + dr, c + dc)
        vw_tpl_h[s] = vh
        vw_tpl_v[s] = vv

    edge_h_cols = 0
    for r in range(8):
        edge_h_cols |= bit(r, 0) | bit(r, 7)

    cell_slots = [0] * CELL_COUNT
    cell_focus = [0] * CELL_COUNT
    for cell in range(CELL_COUNT):
        r, c = cell // 9, cell % 9
        sm = 0
        for sr in (r - 1, r):
            for sc in (c - 1, c):
                sm |= bit(sr, sc)
        cell_slots[cell] = sm

        # Precompute wall slots touching any cell in 3x3 around `cell`
        fslots = 0
        for rr in range(max(0, r - 1), min(8, r + 1) + 1):
            for cc in range(max(0, c - 1), min(8, c + 1) + 1):
                for sr in (rr - 1, rr):
                    for sc in (cc - 1, cc):
                        fslots |= bit(sr, sc)
        cell_focus[cell] = fslots

    return (
        tuple(h_overlap),
        tuple(v_overlap),
        tuple(anchor_5x5),
        tuple(h_edge_anchor),
        tuple(v_edge_anchor),
        tuple(hw_tpl_h),
        tuple(hw_tpl_v),
        tuple(vw_tpl_h),
        tuple(vw_tpl_v),
        edge_h_cols,
        tuple(cell_slots),
        tuple(cell_focus),
    )


(
    H_OVERLAP_MASK,
    V_OVERLAP_MASK,
    ANCHOR_MASK_5X5,
    H_EDGE_ANCHOR,
    V_EDGE_ANCHOR,
    HW_TPL_H,
    HW_TPL_V,
    VW_TPL_H,
    VW_TPL_V,
    EDGE_H_COLS_MASK,
    CELL_SLOTS_MASK,
    CELL_FOCUS_SLOTS_MASK,
) = _build_wall_masks()

# ---------- Zobrist Tables (exact 32-bit xorshift32 sequence from engine-core.js) ----------

def _init_zobrist():
    seed = 0x9E3779B9

    def zrand() -> int:
        nonlocal seed
        seed ^= (seed << 13) & 0xFFFFFFFF
        seed ^= (seed >> 17) & 0xFFFFFFFF
        seed ^= (seed << 5) & 0xFFFFFFFF
        return seed & 0xFFFFFFFF

    pawn_lo = [[0] * CELL_COUNT for _ in range(2)]
    pawn_hi = [[0] * CELL_COUNT for _ in range(2)]
    for p in range(2):
        for c in range(CELL_COUNT):
            pawn_lo[p][c] = zrand()
            pawn_hi[p][c] = zrand()

    hw_lo = [0] * WALL_SLOTS
    hw_hi = [0] * WALL_SLOTS
    vw_lo = [0] * WALL_SLOTS
    vw_hi = [0] * WALL_SLOTS
    for s in range(WALL_SLOTS):
        hw_lo[s] = zrand()
        hw_hi[s] = zrand()
        vw_lo[s] = zrand()
        vw_hi[s] = zrand()

    turn_lo = zrand()
    turn_hi = zrand()
    return (
        (tuple(pawn_lo[0]), tuple(pawn_lo[1])),
        (tuple(pawn_hi[0]), tuple(pawn_hi[1])),
        tuple(hw_lo),
        tuple(hw_hi),
        tuple(vw_lo),
        tuple(vw_hi),
        turn_lo,
        turn_hi,
    )


Z_PAWN_LO, Z_PAWN_HI, Z_HW_LO, Z_HW_HI, Z_VW_LO, Z_VW_HI, Z_TURN_LO, Z_TURN_HI = _init_zobrist()


class _SearchAbort(Exception):
    """Unwinds search cleanly when node or time budget expires."""


class _Board:
    """Fast make/unmake 9x9 Quoridor board matching js/ace/engine-core.js."""

    __slots__ = (
        "pawn",
        "wl",
        "turn",
        "hw_mask",
        "vw_mask",
        "blocked",
        "hash_lo",
        "hash_hi",
        "hist_len",
        "hist_m",
        "hist_from",
        "dist_cache",
    )

    def __init__(self) -> None:
        self.pawn = [76, 4]
        self.wl = [10, 10]
        self.turn = 0
        self.hw_mask = 0
        self.vw_mask = 0
        self.blocked = [0] * CELL_COUNT
        self.hash_lo = Z_PAWN_LO[0][76] ^ Z_PAWN_LO[1][4]
        self.hash_hi = Z_PAWN_HI[0][76] ^ Z_PAWN_HI[1][4]
        self.hist_len = 0
        self.hist_m: List[int] = []
        self.hist_from: List[int] = []
        self.dist_cache: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}

    @classmethod
    def from_observation(cls, obs: dict) -> "_Board":
        b = cls()
        me = obs["player_id"]
        opp = 1 - me
        you = obs["you"]
        them = obs["opponent"]
        b.pawn[me] = you["position"][0] * 9 + you["position"][1]
        b.pawn[opp] = them["position"][0] * 9 + them["position"][1]
        b.wl[me] = you["walls_remaining"]
        b.wl[opp] = them["walls_remaining"]
        b.turn = obs["current_player"]
        b.hist_len = int(obs.get("turn", 0))

        h_lo = Z_PAWN_LO[0][b.pawn[0]] ^ Z_PAWN_LO[1][b.pawn[1]]
        h_hi = Z_PAWN_HI[0][b.pawn[0]] ^ Z_PAWN_HI[1][b.pawn[1]]
        if b.turn == 1:
            h_lo ^= Z_TURN_LO
            h_hi ^= Z_TURN_HI

        for r, c in obs["walls"]["horizontal"]:
            s = r * 8 + c
            b.hw_mask |= 1 << s
            b._set_wall_bits(0, s, True)
            h_lo ^= Z_HW_LO[s]
            h_hi ^= Z_HW_HI[s]
        for r, c in obs["walls"]["vertical"]:
            s = r * 8 + c
            b.vw_mask |= 1 << s
            b._set_wall_bits(1, s, True)
            h_lo ^= Z_VW_LO[s]
            h_hi ^= Z_VW_HI[s]

        b.hash_lo = h_lo
        b.hash_hi = h_hi
        return b

    def _set_wall_bits(self, wtype: int, slot: int, on: bool) -> None:
        r = SLOT_ROW[slot]
        c = SLOT_COL[slot]
        a = r * 9 + c
        blk = self.blocked
        if wtype == 0:
            b = a + 1
            cc = a + 9
            dd = b + 9
            if on:
                blk[a] |= 2
                blk[b] |= 2
                blk[cc] |= 1
                blk[dd] |= 1
            else:
                blk[a] &= ~2
                blk[b] &= ~2
                blk[cc] &= ~1
                blk[dd] &= ~1
        else:
            b = a + 9
            cc = a + 1
            dd = b + 1
            if on:
                blk[a] |= 8
                blk[b] |= 8
                blk[cc] |= 4
                blk[dd] |= 4
            else:
                blk[a] &= ~8
                blk[b] &= ~8
                blk[cc] &= ~4
                blk[dd] &= ~4

    def wall_fits(self, wtype: int, slot: int) -> bool:
        if wtype == 0:
            return not ((self.hw_mask & H_OVERLAP_MASK[slot]) or (self.vw_mask & (1 << slot)))
        return not ((self.vw_mask & V_OVERLAP_MASK[slot]) or (self.hw_mask & (1 << slot)))

    def wall_needs_path_check(self, wtype: int, slot: int) -> bool:
        hit = (self.hw_mask | self.vw_mask) & ANCHOR_MASK_5X5[slot]
        if not hit:
            return False
        edge = H_EDGE_ANCHOR[slot] if wtype == 0 else V_EDGE_ANCHOR[slot]
        if edge:
            return True
        return (hit & (hit - 1)) != 0

    def has_path(self, player: int) -> bool:
        start = self.pawn[player]
        goal = 0 if player == 0 else 8
        row_of = CELL_ROW
        if row_of[start] == goal:
            return True
        seen = bytearray(CELL_COUNT)
        seen[start] = 1
        q = [start]
        idx = 0
        blk = self.blocked
        border = BORDER
        while idx < len(q):
            u = q[idx]
            idx += 1
            bm = blk[u] | border[u]
            if not (bm & 1):
                v = u - 9
                if not seen[v]:
                    if row_of[v] == goal:
                        return True
                    seen[v] = 1
                    q.append(v)
            if not (bm & 2):
                v = u + 9
                if not seen[v]:
                    if row_of[v] == goal:
                        return True
                    seen[v] = 1
                    q.append(v)
            if not (bm & 4):
                v = u - 1
                if not seen[v]:
                    if row_of[v] == goal:
                        return True
                    seen[v] = 1
                    q.append(v)
            if not (bm & 8):
                v = u + 1
                if not seen[v]:
                    if row_of[v] == goal:
                        return True
                    seen[v] = 1
                    q.append(v)
        return False

    def wall_legal(self, wtype: int, slot: int) -> bool:
        if self.wl[self.turn] <= 0:
            return False
        if not self.wall_fits(wtype, slot):
            return False
        if not self.wall_needs_path_check(wtype, slot):
            return True
        self._set_wall_bits(wtype, slot, True)
        ok = self.has_path(0) and self.has_path(1)
        self._set_wall_bits(wtype, slot, False)
        return ok

    def gen_pawn_moves(self) -> List[int]:
        me = self.turn
        s = self.pawn[me]
        o = self.pawn[1 - me]
        blk = self.blocked
        border = BORDER
        bm_s = blk[s] | border[s]
        out: List[int] = []
        for d in range(4):
            bit = DIRBIT[d]
            if bm_s & bit:
                continue
            delta = DELTA[d]
            t = s + delta
            if t != o:
                out.append(t)
                continue
            bm_o = blk[o] | border[o]
            if not (bm_o & bit):
                out.append(o + delta)
                continue
            p1 = 2 if d < 2 else 0
            p2 = 3 if d < 2 else 1
            if not (bm_o & DIRBIT[p1]):
                w1 = o + DELTA[p1]
                if w1 != s:
                    out.append(w1)
            if not (bm_o & DIRBIT[p2]):
                w2 = o + DELTA[p2]
                if w2 != s:
                    out.append(w2)
        if len(out) > 1:
            out = sorted(set(out))
        return out

    def make_move(self, m: int) -> None:
        self.hist_m.append(m)
        p = self.turn
        if m < H_BASE:
            frm = self.pawn[p]
            self.hist_from.append(frm)
            self.hash_lo ^= Z_PAWN_LO[p][frm] ^ Z_PAWN_LO[p][m]
            self.hash_hi ^= Z_PAWN_HI[p][frm] ^ Z_PAWN_HI[p][m]
            self.pawn[p] = m
        elif m < V_BASE:
            s0 = m - H_BASE
            self.hist_from.append(-1)
            self.hw_mask |= 1 << s0
            self._set_wall_bits(0, s0, True)
            self.wl[p] -= 1
            self.hash_lo ^= Z_HW_LO[s0]
            self.hash_hi ^= Z_HW_HI[s0]
        else:
            s1 = m - V_BASE
            self.hist_from.append(-1)
            self.vw_mask |= 1 << s1
            self._set_wall_bits(1, s1, True)
            self.wl[p] -= 1
            self.hash_lo ^= Z_VW_LO[s1]
            self.hash_hi ^= Z_VW_HI[s1]
        self.turn = 1 - p
        self.hash_lo ^= Z_TURN_LO
        self.hash_hi ^= Z_TURN_HI
        self.hist_len += 1

    def unmake_move(self) -> None:
        self.hist_len -= 1
        m = self.hist_m.pop()
        frm = self.hist_from.pop()
        p = 1 - self.turn
        self.turn = p
        self.hash_lo ^= Z_TURN_LO
        self.hash_hi ^= Z_TURN_HI
        if m < H_BASE:
            self.hash_lo ^= Z_PAWN_LO[p][frm] ^ Z_PAWN_LO[p][m]
            self.hash_hi ^= Z_PAWN_HI[p][frm] ^ Z_PAWN_HI[p][m]
            self.pawn[p] = frm
        elif m < V_BASE:
            s0 = m - H_BASE
            self.hw_mask &= ~(1 << s0)
            self._set_wall_bits(0, s0, False)
            self.wl[p] += 1
            self.hash_lo ^= Z_HW_LO[s0]
            self.hash_hi ^= Z_HW_HI[s0]
        else:
            s1 = m - V_BASE
            self.vw_mask &= ~(1 << s1)
            self._set_wall_bits(1, s1, False)
            self.wl[p] += 1
            self.hash_lo ^= Z_VW_LO[s1]
            self.hash_hi ^= Z_VW_HI[s1]

    def _compute_dist_pair(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Multi-source BFS from both goal rows, cached by placed-wall bitmask."""
        wkey = self.hw_mask | (self.vw_mask << 64)
        cached = self.dist_cache.get(wkey)
        if cached is not None:
            return cached

        blk = self.blocked
        border = BORDER

        def bfs_goal(goal_cells: Tuple[int, ...]) -> Tuple[int, ...]:
            dist = [255] * CELL_COUNT
            q = list(goal_cells)
            for c in goal_cells:
                dist[c] = 0
            idx = 0
            while idx < len(q):
                u = q[idx]
                idx += 1
                du = dist[u] + 1
                bm = blk[u] | border[u]
                if not (bm & 1):
                    v = u - 9
                    if dist[v] > du:
                        dist[v] = du
                        q.append(v)
                if not (bm & 2):
                    v = u + 9
                    if dist[v] > du:
                        dist[v] = du
                        q.append(v)
                if not (bm & 4):
                    v = u - 1
                    if dist[v] > du:
                        dist[v] = du
                        q.append(v)
                if not (bm & 8):
                    v = u + 1
                    if dist[v] > du:
                        dist[v] = du
                        q.append(v)
            return tuple(dist)

        pair = (bfs_goal(GOAL_ROW0_CELLS), bfs_goal(GOAL_ROW8_CELLS))
        if len(self.dist_cache) >= DIST_CACHE_MAX:
            self.dist_cache.clear()
        self.dist_cache[wkey] = pair
        return pair

    def dist_pair(self) -> Tuple[int, int]:
        d0, d1 = self._compute_dist_pair()
        return d0[self.pawn[0]], d1[self.pawn[1]]

    def winner(self) -> int:
        if self.pawn[0] < 9:
            return 0
        if self.pawn[1] >= 72:
            return 1
        return -1


# ---------- Evaluation ----------

def _heuristic(b: _Board, ai: int) -> float:
    d0, d1 = b.dist_pair()
    ai_d, opp_d = (d0, d1) if ai == 0 else (d1, d0)
    phase = ai_d if ai_d < opp_d else opp_d
    if phase > 8:
        phase = 8
    w = W_WALLS_BASE + W_WALLS_PHASE * phase / 8.0
    return (opp_d - ai_d) + w * (b.wl[ai] - b.wl[1 - ai])


def _race_verdict(b: _Board, ai: int) -> Optional[int]:
    if b.wl[0] != 0 or b.wl[1] != 0:
        return None
    d0, d1 = b.dist_pair()
    ai_d, opp_d = (d0, d1) if ai == 0 else (d1, d0)
    gap = ai_d - opp_d
    if -2 < gap < 2:
        return None
    return (900 - ai_d) if gap < 0 else -(900 - opp_d)


# ---------- Candidate Generation ----------

def _shortest_path_slots_mask(b: _Board, player: int) -> int:
    start = b.pawn[player]
    goal = 0 if player == 0 else 8
    row_of = CELL_ROW
    cslots = CELL_SLOTS_MASK
    if row_of[start] == goal:
        return cslots[start]

    prev = [-1] * CELL_COUNT
    seen = bytearray(CELL_COUNT)
    seen[start] = 1
    q = [start]
    idx = 0
    blk = b.blocked
    border = BORDER
    end = -1
    while idx < len(q):
        u = q[idx]
        idx += 1
        if row_of[u] == goal:
            end = u
            break
        bm = blk[u] | border[u]
        if not (bm & 1):
            v = u - 9
            if not seen[v]:
                seen[v] = 1
                prev[v] = u
                if row_of[v] == goal:
                    end = v
                    break
                q.append(v)
        if not (bm & 2):
            v = u + 9
            if not seen[v]:
                seen[v] = 1
                prev[v] = u
                if row_of[v] == goal:
                    end = v
                    break
                q.append(v)
        if not (bm & 4):
            v = u - 1
            if not seen[v]:
                seen[v] = 1
                prev[v] = u
                if row_of[v] == goal:
                    end = v
                    break
                q.append(v)
        if not (bm & 8):
            v = u + 1
            if not seen[v]:
                seen[v] = 1
                prev[v] = u
                if row_of[v] == goal:
                    end = v
                    break
                q.append(v)

    mask = cslots[start]
    while end >= 0:
        mask |= cslots[end]
        end = prev[end]
    return mask


def _dag_cut_walls(b: _Board, player: int) -> Tuple[int, int]:
    start = b.pawn[player]
    goal = 0 if player == 0 else 8
    row_of = CELL_ROW
    if row_of[start] == goal:
        return 0, 0

    dist_s = [255] * CELL_COUNT
    dist_s[start] = 0
    q = [start]
    idx = 0
    blk = b.blocked
    border = BORDER
    while idx < len(q):
        u = q[idx]
        idx += 1
        if row_of[u] == goal:
            continue
        du = dist_s[u] + 1
        bm = blk[u] | border[u]
        if not (bm & 1):
            v = u - 9
            if dist_s[v] == 255:
                dist_s[v] = du
                q.append(v)
        if not (bm & 2):
            v = u + 9
            if dist_s[v] == 255:
                dist_s[v] = du
                q.append(v)
        if not (bm & 4):
            v = u - 1
            if dist_s[v] == 255:
                dist_s[v] = du
                q.append(v)
        if not (bm & 8):
            v = u + 1
            if dist_s[v] == 255:
                dist_s[v] = du
                q.append(v)

    goal_base = goal * 9
    dmin = min(dist_s[goal_base : goal_base + 9])
    if dmin == 255:
        return 0, 0

    seen = bytearray(CELL_COUNT)
    back: List[int] = []
    for gc in range(goal_base, goal_base + 9):
        if dist_s[gc] == dmin:
            back.append(gc)
            seen[gc] = 1

    h_mask = 0
    v_mask = 0
    bidx = 0
    while bidx < len(back):
        cur = back[bidx]
        bidx += 1
        r = row_of[cur]
        cc = CELL_COL[cur]
        dd = dist_s[cur] - 1
        bm = blk[cur] | border[cur]
        for d in range(4):
            if bm & DIRBIT[d]:
                continue
            p = cur + DELTA[d]
            if dist_s[p] != dd:
                continue
            pr = row_of[p]
            pc = CELL_COL[p]
            if r == pr - 1:
                if pr - 1 >= 0:
                    if pc <= 7:
                        h_mask |= 1 << ((pr - 1) * 8 + pc)
                    if pc - 1 >= 0:
                        h_mask |= 1 << ((pr - 1) * 8 + pc - 1)
            elif r == pr + 1:
                if pr <= 7:
                    if pc <= 7:
                        h_mask |= 1 << (pr * 8 + pc)
                    if pc - 1 >= 0:
                        h_mask |= 1 << (pr * 8 + pc - 1)
            elif cc == pc - 1:
                if pc - 1 >= 0:
                    if pr <= 7:
                        v_mask |= 1 << (pr * 8 + pc - 1)
                    if pr - 1 >= 0:
                        v_mask |= 1 << ((pr - 1) * 8 + pc - 1)
            else:
                if pc <= 7:
                    if pr <= 7:
                        v_mask |= 1 << (pr * 8 + pc)
                    if pr - 1 >= 0:
                        v_mask |= 1 << ((pr - 1) * 8 + pc)
            if not seen[p]:
                seen[p] = 1
                back.append(p)
    return h_mask, v_mask


def _emit_mask_walls(b: _Board, wtype: int, base: int, mask: int, out: List[int]) -> None:
    m = mask
    while m:
        lsb = m & -m
        s = lsb.bit_length() - 1
        m ^= lsb
        if b.wall_legal(wtype, s):
            out.append(base + s)


def _candidates(b: _Board, use_walls: bool, cap: int) -> List[int]:
    out = b.gen_pawn_moves()
    me = b.turn
    opp = 1 - me
    if not use_walls or b.wl[me] <= 0:
        return out

    if b.wl[opp] == 0:
        h_mask, v_mask = _dag_cut_walls(b, opp)
        _emit_mask_walls(b, 0, H_BASE, h_mask, out)
        _emit_mask_walls(b, 1, V_BASE, v_mask, out)
        return out

    opp_cell = b.pawn[opp]
    me_cell = b.pawn[me]
    slots_mask = (
        CELL_FOCUS_SLOTS_MASK[me_cell]
        | CELL_FOCUS_SLOTS_MASK[opp_cell]
        | _shortest_path_slots_mask(b, opp)
    )

    opr = CELL_ROW[opp_cell]
    opc = CELL_COL[opp_cell]
    mpr = CELL_ROW[me_cell]
    mpc = CELL_COL[me_cell]

    if cap > 0:
        keys: List[Tuple[int, int]] = []
        m = slots_mask
        while m:
            lsb = m & -m
            s = lsb.bit_length() - 1
            m ^= lsb
            keys.append((abs(SLOT_ROW[s] - opr) + abs(SLOT_COL[s] - opc), s))
        if len(keys) > cap:
            keys.sort()
            slots_mask = 0
            for _, s in keys[:cap]:
                slots_mask |= 1 << s

    pairs_h = slots_mask
    pairs_v = slots_mask

    tpl_h = EDGE_H_COLS_MASK if b.hist_len >= 6 else 0
    tpl_v = 0
    m = b.hw_mask
    while m:
        lsb = m & -m
        s = lsb.bit_length() - 1
        m ^= lsb
        tpl_h |= HW_TPL_H[s]
        tpl_v |= HW_TPL_V[s]
    m = b.vw_mask
    while m:
        lsb = m & -m
        s = lsb.bit_length() - 1
        m ^= lsb
        tpl_h |= VW_TPL_H[s]
        tpl_v |= VW_TPL_V[s]

    tpl_h &= ~pairs_h
    tpl_v &= ~pairs_v

    if cap > 0 and (tpl_h or tpl_v):
        tkeys: List[Tuple[int, int]] = []
        mh = tpl_h
        while mh:
            lsb = mh & -mh
            s = lsb.bit_length() - 1
            mh ^= lsb
            sr = SLOT_ROW[s]
            sc = SLOT_COL[s]
            da = abs(sr - mpr) + abs(sc - mpc)
            db = abs(sr - opr) + abs(sc - opc)
            tkeys.append((da if da < db else db, s))
        mv = tpl_v
        while mv:
            lsb = mv & -mv
            s = lsb.bit_length() - 1
            mv ^= lsb
            sr = SLOT_ROW[s]
            sc = SLOT_COL[s]
            da = abs(sr - mpr) + abs(sc - mpc)
            db = abs(sr - opr) + abs(sc - opc)
            tkeys.append((da if da < db else db, 64 + s))
        if len(tkeys) > cap:
            tkeys.sort()
            tpl_h = 0
            tpl_v = 0
            for _, k in tkeys[:cap]:
                if k < 64:
                    tpl_h |= 1 << k
                else:
                    tpl_v |= 1 << (k - 64)

    pairs_h |= tpl_h
    pairs_v |= tpl_v

    _emit_mask_walls(b, 0, H_BASE, pairs_h, out)
    _emit_mask_walls(b, 1, V_BASE, pairs_v, out)
    return out


# ---------- Search ----------

class _Ctl:
    __slots__ = ("nodes", "limit", "deadline")

    def __init__(self, limit: int, deadline: float) -> None:
        self.nodes = 0
        self.limit = limit
        self.deadline = deadline

    def tick(self) -> None:
        n = self.nodes + 1
        self.nodes = n
        if self.limit and n > self.limit:
            raise _SearchAbort()
        if self.deadline and (n & 31) == 0 and time.perf_counter() > self.deadline:
            raise _SearchAbort()


def _move_legal_here(b: _Board, m: int) -> bool:
    if m < 0:
        return False
    if m < H_BASE:
        return m in b.gen_pawn_moves()
    if m < V_BASE:
        return b.wall_legal(0, m - H_BASE)
    return b.wall_legal(1, m - V_BASE)


def _order_moves(
    b: _Board,
    moves: List[int],
    tt_move: int,
    ply: int,
    killers: List[List[int]],
    history: Dict[int, int],
) -> None:
    while len(killers) <= ply:
        killers.append([])
    klist = killers[ply]
    k0 = klist[0] if len(klist) >= 1 else -1
    k1 = klist[1] if len(klist) >= 2 else -1

    def sort_key(m: int) -> Tuple[int, int, int]:
        if m == tt_move:
            return (-1, 0, 0)
        k_rank = 0 if m == k0 else (1 if m == k1 else 2)
        h_score = history.get(m, 0)
        is_wall = 0 if m < H_BASE else 1
        return (k_rank, -h_score, is_wall)

    moves.sort(key=sort_key)
    if tt_move >= 0 and (not moves or moves[0] != tt_move):
        if tt_move not in moves and _move_legal_here(b, tt_move):
            moves.insert(0, tt_move)


def _record_cutoff(
    m: int,
    depth: int,
    ply: int,
    killers: List[List[int]],
    history: Dict[int, int],
    tt_move: int,
) -> None:
    history[m] = history.get(m, 0) + depth * depth
    if m == tt_move:
        return
    lst = killers[ply]
    if m in lst:
        lst.remove(m)
    lst.insert(0, m)
    if len(lst) > 2:
        del lst[2:]


def _minimax(
    b: _Board,
    depth: int,
    alpha: float,
    beta: float,
    ai: int,
    tt: Dict[int, Tuple[int, int, float, int, int]],
    ply: int,
    killers: List[List[int]],
    history: Dict[int, int],
    ctl: _Ctl,
) -> Tuple[float, int]:
    w = b.winner()
    if w != -1:
        return (1000.0 + depth if w == ai else -1000.0 - depth), -1

    ctl.tick()

    if depth <= 0:
        race = _race_verdict(b, ai)
        if race is not None:
            return float(race), -1
        return _heuristic(b, ai), -1

    alpha_orig = alpha
    beta_orig = beta
    key = b.hash_lo * 121 + b.wl[0] * 11 + b.wl[1]
    hit = tt.get(key)
    tt_move = -1
    if hit is not None and hit[0] == b.hash_hi:
        tt_move = hit[4]
        if hit[1] >= depth:
            h_flag = hit[3]
            h_val = hit[2]
            if h_flag == TT_EXACT:
                return h_val, tt_move
            if h_flag == TT_LOWER and h_val > alpha:
                alpha = h_val
            elif h_flag == TT_UPPER and h_val < beta:
                beta = h_val
            if alpha >= beta:
                return h_val, tt_move

    maximizing = b.turn == ai
    use_walls = depth >= WALL_FLOOR
    moves = _candidates(b, use_walls, WALL_CAP)
    if not moves:
        return _heuristic(b, ai), -1

    _order_moves(b, moves, tt_move, ply, killers, history)

    value = -INF if maximizing else INF
    best_move = -1

    for i, m in enumerate(moves):
        r = 1 if (depth >= LMR_MIN_DEPTH and i >= LMR_AFTER) else 0
        b.make_move(m)
        child, _ = _minimax(
            b, depth - 1 - r, alpha, beta, ai, tt, ply + 1, killers, history, ctl
        )
        if r and ((child > alpha) if maximizing else (child < beta)):
            child, _ = _minimax(
                b, depth - 1, alpha, beta, ai, tt, ply + 1, killers, history, ctl
            )
        b.unmake_move()

        if maximizing:
            if child > value:
                value = child
                best_move = m
            if value > alpha:
                alpha = value
        else:
            if child < value:
                value = child
                best_move = m
            if value < beta:
                beta = value
        if alpha >= beta:
            _record_cutoff(m, depth, ply, killers, history, tt_move)
            break

    flag = TT_UPPER if value <= alpha_orig else (TT_LOWER if value >= beta_orig else TT_EXACT)
    if len(tt) >= TT_MAX:
        tt.clear()
    tt[key] = (b.hash_hi, depth, value, flag, best_move)
    return value, best_move


def _greedy_fallback(b: _Board, ai: int) -> int:
    best = -1
    best_d = INF
    for m in b.gen_pawn_moves():
        b.make_move(m)
        d0, d1 = b.dist_pair()
        d = d0 if ai == 0 else d1
        b.unmake_move()
        if d < best_d:
            best_d = d
            best = m
    if best >= 0:
        return best
    for s in range(WALL_SLOTS):
        if b.wall_legal(0, s):
            return H_BASE + s
        if b.wall_legal(1, s):
            return V_BASE + s
    return -1


def _pos_key(b: _Board) -> Tuple[int, int]:
    return (b.hash_lo, b.hash_hi)


def _repeats(b: _Board, move: int, visited: Set[Tuple[int, int]]) -> bool:
    b.make_move(move)
    seen = _pos_key(b) in visited
    b.unmake_move()
    return seen


def _anti_shuffle(b: _Board, move: int, ai: int, visited: Set[Tuple[int, int]]) -> int:
    if move < 0 or move >= H_BASE:
        return move
    if not _repeats(b, move, visited):
        return move
    best = -1
    best_d = INF
    for m in b.gen_pawn_moves():
        if _repeats(b, m, visited):
            continue
        b.make_move(m)
        d0, d1 = b.dist_pair()
        d = d0 if ai == 0 else d1
        b.unmake_move()
        if d < best_d:
            best_d = d
            best = m
    return best if best >= 0 else move


# Module-level history for anti-shuffle across turns within a single game.
_VISITED_POSITIONS: Set[Tuple[int, int]] = set()
_LAST_WALL_SUM: int = 20


def _move_to_action(m: int) -> dict:
    if m < H_BASE:
        return {"type": "move", "to": [CELL_ROW[m], CELL_COL[m]]}
    if m < V_BASE:
        s = m - H_BASE
        return {"type": "wall", "orientation": "H", "row": SLOT_ROW[s], "col": SLOT_COL[s]}
    s = m - V_BASE
    return {"type": "wall", "orientation": "V", "row": SLOT_ROW[s], "col": SLOT_COL[s]}


def choose_action(observation: dict) -> dict:
    """Choose an action using QSeriesBot Expert ("Strategist") search."""
    global _LAST_WALL_SUM
    started = time.perf_counter()

    if observation.get("done"):
        return {"type": "pass"}

    legal_moves = observation.get("legal_moves") or []
    legal_walls = observation.get("legal_walls") or []
    if not legal_moves and not legal_walls:
        return {"type": "pass"}

    b = _Board.from_observation(observation)
    turn_idx = observation.get("turn", 0)
    wl_sum = b.wl[0] + b.wl[1]
    if turn_idx <= 1 or wl_sum > _LAST_WALL_SUM:
        _VISITED_POSITIONS.clear()
    _LAST_WALL_SUM = wl_sum
    _VISITED_POSITIONS.add(_pos_key(b))

    ai = b.turn
    deadline = started + TIME_BUDGET_SEC
    ctl = _Ctl(NODE_BUDGET, deadline)
    tt: Dict[int, Tuple[int, int, float, int, int]] = {}
    killers: List[List[int]] = []
    history: Dict[int, int] = {}

    chosen = -1
    for d in range(1, MAX_DEPTH + 1):
        try:
            _, best_m = _minimax(b, d, -INF, INF, ai, tt, 0, killers, history, ctl)
        except _SearchAbort:
            break
        if best_m >= 0:
            chosen = best_m
        if time.perf_counter() >= deadline:
            break

    if chosen < 0:
        chosen = _greedy_fallback(b, ai)
    if chosen >= 0:
        chosen = _anti_shuffle(b, chosen, ai, _VISITED_POSITIONS)

    action = _move_to_action(chosen) if chosen >= 0 else {"type": "pass"}

    # Validate output against observation's authoritative legal sets
    if action["type"] == "move":
        if action["to"] in legal_moves:
            return action
    elif action["type"] == "wall":
        target_w = {
            "orientation": action["orientation"],
            "row": action["row"],
            "col": action["col"],
        }
        if target_w in legal_walls:
            return action

    if legal_moves:
        distances = observation["distances"]
        best_to = min(legal_moves, key=lambda cell: distances[cell[0]][cell[1]])
        return {"type": "move", "to": list(best_to)}
    if legal_walls:
        w = legal_walls[0]
        return {
            "type": "wall",
            "orientation": w["orientation"],
            "row": w["row"],
            "col": w["col"],
        }
    return {"type": "pass"}
