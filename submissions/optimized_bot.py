"""Full rule-based optimized Wrongway / Quoridor agent (no MLP).

Combines:
- Bitboard representation, exact edge-cut witness/DAG masks, and Zobrist hashing
- Exact 0-wall retrograde BFS Oracle (13,122 states, cached by wall layout)
- Iterative-deepening PVS Negamax with TT, 2-slot Killers, Butterfly + CMH,
  Reverse Futility Pruning, conservative LMR, and predictive soft/hard time budget
- Focused candidate generation (positive-gain DAG walls + template/focus walls)
- Pure rule-based centipawn evaluation (distance lead, phase-scaled wall value,
  effective walls, race parity, wall exhaustion, jump congestion, corridor exits)
- Layer 0 Safeguards (S0-S6): cross-match reset, 1-ply win/block/jump-escape,
  2-ply anti-fork threat block, exact 0-wall endgame play, root self-harm/waste
  filtering, anti-shuffle cycle breaking, and authoritative action sanitization.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Dict, List, Optional, Tuple

BOARD_SIZE = 9
CELL_COUNT = 81
WALL_SLOTS = 64
MAX_WALLS = 10

H_BASE = 100
V_BASE = 200
HIST_ACTIONS = 256  # _action_code maps 0..80, 100..163 -> 81..144, 200..263 -> 145..208

DELTA = (-9, 9, -1, 1)  # U, D, L, R
MATE = 30000
MATE_BAND = 29000
INF = 31000
EVAL_CLAMP = 9999
CMH_MAX = 32767
RFP_MARGIN = 100
MAX_ID_DEPTH = 64

TIME_BUDGET_MS = 780.0
SOFT_PCT = 0.55
HARD_PCT = 0.84

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

ALL_CELLS = (1 << CELL_COUNT) - 1
GOAL0_MASK = (1 << BOARD_SIZE) - 1
GOAL1_MASK = GOAL0_MASK << (8 * BOARD_SIZE)
GOAL_MASKS = (GOAL0_MASK, GOAL1_MASK)

# ---------------------------------------------------------------------------
# Precomputed Bitboard, Wall-Overlap, DAG-Cut, and Template Focus Tables
# ---------------------------------------------------------------------------

def _build_init_edges() -> Tuple[int, int, int, int]:
    u_ok = ALL_CELLS ^ GOAL0_MASK
    d_ok = ALL_CELLS ^ GOAL1_MASK
    l_ok = 0
    r_ok = 0
    for r in range(BOARD_SIZE):
        base = r * BOARD_SIZE
        l_ok |= (0x1FF ^ 1) << base
        r_ok |= (0x1FF ^ 0x100) << base
    return u_ok, d_ok, l_ok, r_ok


INIT_U_OK, INIT_D_OK, INIT_L_OK, INIT_R_OK = _build_init_edges()


def _build_wall_tables():
    h_s = [0] * WALL_SLOTS
    h_n = [0] * WALL_SLOTS
    v_e = [0] * WALL_SLOTS
    v_w = [0] * WALL_SLOTS
    h_ov = [0] * WALL_SLOTS
    v_ov = [0] * WALL_SLOTS
    anchor5 = [0] * WALL_SLOTS
    hw_h = [0] * WALL_SLOTS
    hw_v = [0] * WALL_SLOTS
    vw_h = [0] * WALL_SLOTS
    vw_v = [0] * WALL_SLOTS
    edge_h = 0

    for wr in range(8):
        for wc in range(8):
            wi = wr * 8 + wc
            a = wr * 9 + wc
            hs = (1 << a) | (1 << (a + 1))
            h_s[wi] = hs
            h_n[wi] = hs << 9

            ve = (1 << a) | (1 << (a + 9))
            v_e[wi] = ve
            v_w[wi] = ve << 1

            ho = 1 << wi
            if wc > 0:
                ho |= 1 << (wi - 1)
            if wc < 7:
                ho |= 1 << (wi + 1)
            h_ov[wi] = ho

            vo = 1 << wi
            if wr > 0:
                vo |= 1 << (wi - 8)
            if wr < 7:
                vo |= 1 << (wi + 8)
            v_ov[wi] = vo

            am = 0
            for dr in range(-2, 3):
                nr = wr + dr
                if 0 <= nr < 8:
                    for dc in range(-2, 3):
                        nc = wc + dc
                        if 0 <= nc < 8:
                            am |= 1 << (nr * 8 + nc)
            anchor5[wi] = am

            if wc == 0 or wc == 7:
                edge_h |= 1 << wi

            # Templates around an existing H wall at (wr, wc)
            th = 0
            if wc >= 2:
                th |= 1 << (wr * 8 + wc - 2)
            if wc <= 5:
                th |= 1 << (wr * 8 + wc + 2)
            for dr in (-1, 1):
                nr = wr + dr
                if 0 <= nr < 8:
                    for dc in (-1, 0, 1):
                        nc = wc + dc
                        if 0 <= nc < 8:
                            th |= 1 << (nr * 8 + nc)
            hw_h[wi] = th

            tv = 0
            for nr in (wr - 1, wr, wr + 1):
                if 0 <= nr < 8:
                    if wc >= 1:
                        tv |= 1 << (nr * 8 + wc - 1)
                    if wc <= 6:
                        tv |= 1 << (nr * 8 + wc + 1)
            for nr in (wr - 1, wr + 1):
                if 0 <= nr < 8:
                    tv |= 1 << (nr * 8 + wc)
            hw_v[wi] = tv

            # Templates around an existing V wall at (wr, wc)
            tv2 = 0
            if wr >= 2:
                tv2 |= 1 << ((wr - 2) * 8 + wc)
            if wr <= 5:
                tv2 |= 1 << ((wr + 2) * 8 + wc)
            for dc in (-1, 1):
                nc = wc + dc
                if 0 <= nc < 8:
                    for dr in (-1, 0, 1):
                        nr = wr + dr
                        if 0 <= nr < 8:
                            tv2 |= 1 << (nr * 8 + nc)
            vw_v[wi] = tv2

            th2 = 0
            for nc in (wc - 1, wc, wc + 1):
                if 0 <= nc < 8:
                    if wr >= 1:
                        th2 |= 1 << ((wr - 1) * 8 + nc)
                    if wr <= 6:
                        th2 |= 1 << ((wr + 1) * 8 + nc)
            for nc in (wc - 1, wc + 1):
                if 0 <= nc < 8:
                    th2 |= 1 << (wr * 8 + nc)
            vw_h[wi] = th2

    step_h = [0] * CELL_COUNT
    step_v = [0] * CELL_COUNT
    cell_focus = [0] * CELL_COUNT
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            sq = r * 9 + c
            if r < 8:
                sh = 0
                if c < 8:
                    sh |= 1 << (r * 8 + c)
                if c > 0:
                    sh |= 1 << (r * 8 + c - 1)
                step_h[sq] = sh
            if c < 8:
                sv = 0
                if r < 8:
                    sv |= 1 << (r * 8 + c)
                if r > 0:
                    sv |= 1 << ((r - 1) * 8 + c)
                step_v[sq] = sv

            cf = 0
            for wr in range(max(0, r - 2), min(8, r + 2)):
                for wc in range(max(0, c - 2), min(8, c + 2)):
                    cf |= 1 << (wr * 8 + wc)
            cell_focus[sq] = cf

    return (
        tuple(h_s),
        tuple(h_n),
        tuple(v_e),
        tuple(v_w),
        tuple(h_ov),
        tuple(v_ov),
        tuple(anchor5),
        tuple(hw_h),
        tuple(hw_v),
        tuple(vw_h),
        tuple(vw_v),
        edge_h,
        tuple(step_h),
        tuple(step_v),
        tuple(cell_focus),
    )


(
    H_SOUTH_BLK,
    H_NORTH_BLK,
    V_EAST_BLK,
    V_WEST_BLK,
    H_OVERLAP_MASK,
    V_OVERLAP_MASK,
    ANCHOR_MASK_5X5,
    HW_TPL_H,
    HW_TPL_V,
    VW_TPL_H,
    VW_TPL_V,
    EDGE_H_COLS_MASK,
    STEP_H_MASK,
    STEP_V_MASK,
    CELL_FOCUS_SLOTS_MASK,
) = _build_wall_tables()


def _build_zobrist():
    s = 0x9E3779B9

    def _next64() -> int:
        nonlocal s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        hi = s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        return (hi << 32) | s

    zp0 = tuple(_next64() for _ in range(CELL_COUNT))
    zp1 = tuple(_next64() for _ in range(CELL_COUNT))
    zwh = tuple(_next64() for _ in range(WALL_SLOTS))
    zwv = tuple(_next64() for _ in range(WALL_SLOTS))
    zwl0 = tuple(_next64() for _ in range(MAX_WALLS + 1))
    zwl1 = tuple(_next64() for _ in range(MAX_WALLS + 1))
    zturn = _next64()
    return zp0, zp1, zwh, zwv, zwl0, zwl1, zturn


Z_PAWN0, Z_PAWN1, Z_WALL_H, Z_WALL_V, Z_WL0, Z_WL1, Z_TURN = _build_zobrist()

CELL_R = tuple(i // 9 for i in range(CELL_COUNT))
CELL_C = tuple(i % 9 for i in range(CELL_COUNT))
CELL_CENTER_DIST = tuple(abs(i % 9 - 4) for i in range(CELL_COUNT))


def _action_code(m: int) -> int:
    if m < 100:
        return m
    if m < 200:
        return m - 19  # 81..144
    return m - 55      # 145..208


def _action_to_dict(m: int) -> Dict[str, Any]:
    if m < 100:
        return {"type": "move", "to": [m // 9, m % 9]}
    if m < 200:
        wi = m - 100
        return {"type": "wall", "orientation": "H", "row": wi // 8, "col": wi % 8}
    wi = m - 200
    return {"type": "wall", "orientation": "V", "row": wi // 8, "col": wi % 8}


def _dict_to_action(d: Dict[str, Any]) -> int:
    if d["type"] == "move":
        r, c = d["to"]
        return r * 9 + c
    wi = d["row"] * 8 + d["col"]
    return (H_BASE if d["orientation"] == "H" else V_BASE) + wi


# ---------------------------------------------------------------------------
# Bitboard Graph Primitives
# ---------------------------------------------------------------------------

def _has_path_bb(start: int, gmask: int, u_ok: int, d_ok: int, l_ok: int, r_ok: int) -> bool:
    seen = 1 << start
    fr = seen
    while fr:
        if fr & gmask:
            return True
        nxt = (
            ((fr & u_ok) >> 9)
            | ((fr & d_ok) << 9)
            | ((fr & l_ok) >> 1)
            | ((fr & r_ok) << 1)
        ) & ~seen
        seen |= nxt
        fr = nxt
    return False


def _bb_dist(start: int, gmask: int, u_ok: int, d_ok: int, l_ok: int, r_ok: int) -> int:
    seen = 1 << start
    fr = seen
    dist = 0
    while fr:
        if fr & gmask:
            return dist
        nxt = (
            ((fr & u_ok) >> 9)
            | ((fr & d_ok) << 9)
            | ((fr & l_ok) >> 1)
            | ((fr & r_ok) << 1)
        ) & ~seen
        seen |= nxt
        fr = nxt
        dist += 1
    return -1


def _pawn_moves_raw(me: int, opp: int, u_ok: int, d_ok: int, l_ok: int, r_ok: int) -> List[int]:
    moves: List[int] = []
    # UP (0), DOWN (1), LEFT (2), RIGHT (3)
    b = 1 << me
    if u_ok & b:
        n1 = me - 9
        if n1 != opp:
            moves.append(n1)
        else:
            ob = 1 << opp
            if u_ok & ob:
                moves.append(opp - 9)
            else:
                if l_ok & ob:
                    moves.append(opp - 1)
                if r_ok & ob:
                    moves.append(opp + 1)
    if d_ok & b:
        n1 = me + 9
        if n1 != opp:
            moves.append(n1)
        else:
            ob = 1 << opp
            if d_ok & ob:
                moves.append(opp + 9)
            else:
                if l_ok & ob:
                    moves.append(opp - 1)
                if r_ok & ob:
                    moves.append(opp + 1)
    if l_ok & b:
        n1 = me - 1
        if n1 != opp:
            moves.append(n1)
        else:
            ob = 1 << opp
            if l_ok & ob:
                moves.append(opp - 1)
            else:
                if u_ok & ob:
                    moves.append(opp - 9)
                if d_ok & ob:
                    moves.append(opp + 9)
    if r_ok & b:
        n1 = me + 1
        if n1 != opp:
            moves.append(n1)
        else:
            ob = 1 << opp
            if r_ok & ob:
                moves.append(opp + 1)
            else:
                if u_ok & ob:
                    moves.append(opp - 9)
                if d_ok & ob:
                    moves.append(opp + 9)
    return moves


# ---------------------------------------------------------------------------
# Fast Mutable Board State
# ---------------------------------------------------------------------------

class _Board:
    __slots__ = (
        "pawn",
        "wl",
        "turn",
        "h_occ",
        "v_occ",
        "u_ok",
        "d_ok",
        "l_ok",
        "r_ok",
        "zob",
        "_undo",
    )

    def __init__(self) -> None:
        self.pawn = [76, 4]
        self.wl = [10, 10]
        self.turn = 0
        self.h_occ = 0
        self.v_occ = 0
        self.u_ok = INIT_U_OK
        self.d_ok = INIT_D_OK
        self.l_ok = INIT_L_OK
        self.r_ok = INIT_R_OK
        self.zob = Z_PAWN0[76] ^ Z_PAWN1[4] ^ Z_WL0[10] ^ Z_WL1[10]
        self._undo: List[Tuple[int, int, int]] = []

    @classmethod
    def from_observation(cls, obs: Dict[str, Any]) -> "_Board":
        b = cls()
        pid = int(obs.get("player_id", obs.get("current_player", 0)))
        you = obs["you"]
        opp = obs["opponent"]
        yr, yc = int(you["position"][0]), int(you["position"][1])
        opr, opc = int(opp["position"][0]), int(opp["position"][1])
        yw = int(you["walls_remaining"])
        ow = int(opp["walls_remaining"])

        if pid == 0:
            p0, p1 = yr * 9 + yc, opr * 9 + opc
            w0, w1 = yw, ow
        else:
            p0, p1 = opr * 9 + opc, yr * 9 + yc
            w0, w1 = ow, yw

        b.pawn = [p0, p1]
        b.wl = [w0, w1]
        b.turn = int(obs.get("current_player", pid))

        h_occ = 0
        v_occ = 0
        u_ok = INIT_U_OK
        d_ok = INIT_D_OK
        l_ok = INIT_L_OK
        r_ok = INIT_R_OK
        zob = Z_PAWN0[p0] ^ Z_PAWN1[p1] ^ Z_WL0[w0] ^ Z_WL1[w1]
        if b.turn == 1:
            zob ^= Z_TURN

        walls = obs.get("walls", {})
        for wr, wc in walls.get("horizontal", ()):
            wi = int(wr) * 8 + int(wc)
            bit = 1 << wi
            if not (h_occ & bit):
                h_occ |= bit
                d_ok &= ~H_SOUTH_BLK[wi]
                u_ok &= ~H_NORTH_BLK[wi]
                zob ^= Z_WALL_H[wi]
        for wr, wc in walls.get("vertical", ()):
            wi = int(wr) * 8 + int(wc)
            bit = 1 << wi
            if not (v_occ & bit):
                v_occ |= bit
                r_ok &= ~V_EAST_BLK[wi]
                l_ok &= ~V_WEST_BLK[wi]
                zob ^= Z_WALL_V[wi]

        b.h_occ = h_occ
        b.v_occ = v_occ
        b.u_ok = u_ok
        b.d_ok = d_ok
        b.l_ok = l_ok
        b.r_ok = r_ok
        b.zob = zob
        b._undo = []
        return b

    def make_move(self, m: int) -> None:
        stm = self.turn
        if m < 100:
            prev_p = self.pawn[stm]
            self.pawn[stm] = m
            if stm == 0:
                self.zob ^= Z_PAWN0[prev_p] ^ Z_PAWN0[m] ^ Z_TURN
            else:
                self.zob ^= Z_PAWN1[prev_p] ^ Z_PAWN1[m] ^ Z_TURN
            self.turn = 1 - stm
            self._undo.append((m, prev_p, stm))
            return

        prev_w = self.wl[stm]
        self.wl[stm] = prev_w - 1
        zwl = Z_WL0 if stm == 0 else Z_WL1
        if m < 200:
            wi = m - 100
            self.h_occ |= 1 << wi
            self.u_ok ^= H_NORTH_BLK[wi]
            self.d_ok ^= H_SOUTH_BLK[wi]
            self.zob ^= Z_WALL_H[wi] ^ zwl[prev_w] ^ zwl[prev_w - 1] ^ Z_TURN
        else:
            wi = m - 200
            self.v_occ |= 1 << wi
            self.l_ok ^= V_WEST_BLK[wi]
            self.r_ok ^= V_EAST_BLK[wi]
            self.zob ^= Z_WALL_V[wi] ^ zwl[prev_w] ^ zwl[prev_w - 1] ^ Z_TURN
        self.turn = 1 - stm
        self._undo.append((m, prev_w, stm))

    def unmake_move(self) -> None:
        m, prev_val, stm = self._undo.pop()
        self.turn = stm
        if m < 100:
            cur_p = self.pawn[stm]
            self.pawn[stm] = prev_val
            if stm == 0:
                self.zob ^= Z_PAWN0[prev_val] ^ Z_PAWN0[cur_p] ^ Z_TURN
            else:
                self.zob ^= Z_PAWN1[prev_val] ^ Z_PAWN1[cur_p] ^ Z_TURN
            return

        self.wl[stm] = prev_val
        zwl = Z_WL0 if stm == 0 else Z_WL1
        if m < 200:
            wi = m - 100
            self.h_occ ^= 1 << wi
            self.u_ok ^= H_NORTH_BLK[wi]
            self.d_ok ^= H_SOUTH_BLK[wi]
            self.zob ^= Z_WALL_H[wi] ^ zwl[prev_val] ^ zwl[prev_val - 1] ^ Z_TURN
        else:
            wi = m - 200
            self.v_occ ^= 1 << wi
            self.l_ok ^= V_WEST_BLK[wi]
            self.r_ok ^= V_EAST_BLK[wi]
            self.zob ^= Z_WALL_V[wi] ^ zwl[prev_val] ^ zwl[prev_val - 1] ^ Z_TURN

    def gen_pawn_moves(self, player: Optional[int] = None) -> List[int]:
        p = self.turn if player is None else player
        return _pawn_moves_raw(
            self.pawn[p], self.pawn[1 - p], self.u_ok, self.d_ok, self.l_ok, self.r_ok
        )

    def wall_legal_flags(
        self,
        is_v: bool,
        wi: int,
        w0_h: int,
        w0_v: int,
        w1_h: int,
        w1_v: int,
    ) -> bool:
        bit = 1 << wi
        if is_v:
            if (self.v_occ & V_OVERLAP_MASK[wi]) or (self.h_occ & bit):
                return False
        else:
            if (self.h_occ & H_OVERLAP_MASK[wi]) or (self.v_occ & bit):
                return False

        if ((self.h_occ | self.v_occ) & ANCHOR_MASK_5X5[wi]) == 0:
            return True

        need0 = bool(bit & (w0_v if is_v else w0_h))
        need1 = bool(bit & (w1_v if is_v else w1_h))
        if not need0 and not need1:
            return True

        if is_v:
            l2 = self.l_ok ^ V_WEST_BLK[wi]
            r2 = self.r_ok ^ V_EAST_BLK[wi]
            u2 = self.u_ok
            d2 = self.d_ok
        else:
            u2 = self.u_ok ^ H_NORTH_BLK[wi]
            d2 = self.d_ok ^ H_SOUTH_BLK[wi]
            l2 = self.l_ok
            r2 = self.r_ok

        if need0 and not _has_path_bb(self.pawn[0], GOAL0_MASK, u2, d2, l2, r2):
            return False
        if need1 and not _has_path_bb(self.pawn[1], GOAL1_MASK, u2, d2, l2, r2):
            return False
        return True

    def compute_dist_tables(self, dist_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]]):
        key = (self.h_occ, self.v_occ)
        cached = dist_cache.get(key)
        if cached is not None:
            return cached

        u_ok, d_ok, l_ok, r_ok = self.u_ok, self.d_ok, self.l_ok, self.r_ok

        def _fill_from_goal(gmask: int) -> Tuple[int, ...]:
            tbl = [999] * CELL_COUNT
            seen = gmask
            fr = seen
            step = 0
            while fr:
                m = fr
                while m:
                    lsb = m & -m
                    tbl[lsb.bit_length() - 1] = step
                    m ^= lsb
                step += 1
                fr = (
                    ((fr & u_ok) >> 9)
                    | ((fr & d_ok) << 9)
                    | ((fr & l_ok) >> 1)
                    | ((fr & r_ok) << 1)
                ) & ~seen
                seen |= fr
            return tuple(tbl)

        res = (_fill_from_goal(GOAL0_MASK), _fill_from_goal(GOAL1_MASK))
        if len(dist_cache) > 65536:
            dist_cache.clear()
        dist_cache[key] = res
        return res

    def witness_cut_masks(self, d0_tbl: Tuple[int, ...], d1_tbl: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        u_ok, d_ok, l_ok, r_ok = self.u_ok, self.d_ok, self.l_ok, self.r_ok

        def _trace(start: int, d_tbl: Tuple[int, ...]) -> Tuple[int, int]:
            wh = 0
            wv = 0
            cur = start
            cur_d = d_tbl[cur]
            if cur_d >= 999:
                return 0, 0
            while cur_d > 0:
                b = 1 << cur
                nd = cur_d - 1
                if (u_ok & b) and d_tbl[cur - 9] == nd:
                    nxt = cur - 9
                    wh |= STEP_H_MASK[nxt]
                elif (d_ok & b) and d_tbl[cur + 9] == nd:
                    nxt = cur + 9
                    wh |= STEP_H_MASK[cur]
                elif (l_ok & b) and d_tbl[cur - 1] == nd:
                    nxt = cur - 1
                    wv |= STEP_V_MASK[nxt]
                elif (r_ok & b) and d_tbl[cur + 1] == nd:
                    nxt = cur + 1
                    wv |= STEP_V_MASK[cur]
                else:
                    break
                cur = nxt
                cur_d = nd
            return wh, wv

        w0_h, w0_v = _trace(self.pawn[0], d0_tbl)
        w1_h, w1_v = _trace(self.pawn[1], d1_tbl)
        return w0_h, w0_v, w1_h, w1_v

    def dag_cut_masks(self, player: int, d_tbl: Tuple[int, ...]) -> Tuple[int, int]:
        start = self.pawn[player]
        start_d = d_tbl[start]
        if start_d <= 0 or start_d >= 999:
            return 0, 0

        u_ok, d_ok, l_ok, r_ok = self.u_ok, self.d_ok, self.l_ok, self.r_ok
        cut_h = 0
        cut_v = 0
        vis = 1 << start
        q = [start]
        qi = 0
        while qi < len(q):
            cur = q[qi]
            qi += 1
            nd = d_tbl[cur] - 1
            if nd < 0:
                continue
            b = 1 << cur
            if (u_ok & b):
                nxt = cur - 9
                if d_tbl[nxt] == nd:
                    cut_h |= STEP_H_MASK[nxt]
                    nb = 1 << nxt
                    if not (vis & nb):
                        vis |= nb
                        if nd > 0:
                            q.append(nxt)
            if (d_ok & b):
                nxt = cur + 9
                if d_tbl[nxt] == nd:
                    cut_h |= STEP_H_MASK[cur]
                    nb = 1 << nxt
                    if not (vis & nb):
                        vis |= nb
                        if nd > 0:
                            q.append(nxt)
            if (l_ok & b):
                nxt = cur - 1
                if d_tbl[nxt] == nd:
                    cut_v |= STEP_V_MASK[nxt]
                    nb = 1 << nxt
                    if not (vis & nb):
                        vis |= nb
                        if nd > 0:
                            q.append(nxt)
            if (r_ok & b):
                nxt = cur + 1
                if d_tbl[nxt] == nd:
                    cut_v |= STEP_V_MASK[cur]
                    nb = 1 << nxt
                    if not (vis & nb):
                        vis |= nb
                        if nd > 0:
                            q.append(nxt)
        return cut_h, cut_v


# ---------------------------------------------------------------------------
# Exact Zero-Wall Retrograde BFS Oracle (13,122 states)
# ---------------------------------------------------------------------------

class _Oracle:
    def __init__(self, cap: int = 32) -> None:
        self.cap = cap
        self.cache: Dict[Tuple[int, int], Tuple[bytearray, List[int]]] = {}

    def get(self, b: _Board) -> Tuple[bytearray, List[int]]:
        key = (b.h_occ, b.v_occ)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        res, dtm = self._solve_table(b.u_ok, b.d_ok, b.l_ok, b.r_ok)
        if len(self.cache) >= self.cap:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (res, dtm)
        return res, dtm

    @staticmethod
    def _solve_table(u_ok: int, d_ok: int, l_ok: int, r_ok: int) -> Tuple[bytearray, List[int]]:
        # Index: ((turn * 81) + p0) * 81 + p1
        n_states = 2 * CELL_COUNT * CELL_COUNT
        adj: List[Tuple[int, ...]] = []
        adj_mask = [0] * CELL_COUNT
        for sq in range(CELL_COUNT):
            bit = 1 << sq
            nbrs = []
            m = 0
            if u_ok & bit:
                x = sq - 9
                nbrs.append(x)
                m |= 1 << x
            if d_ok & bit:
                x = sq + 9
                nbrs.append(x)
                m |= 1 << x
            if l_ok & bit:
                x = sq - 1
                nbrs.append(x)
                m |= 1 << x
            if r_ok & bit:
                x = sq + 1
                nbrs.append(x)
                m |= 1 << x
            adj.append(tuple(nbrs))
            adj_mask[sq] = m

        res = bytearray(n_states)  # 0 unknown, 1 win (+1), 2 loss (-1)
        dtm = [0] * n_states
        outdeg = bytearray(n_states)
        preds: List[List[int]] = [[] for _ in range(n_states)]
        q: deque[int] = deque()

        for p0 in range(CELL_COUNT):
            p0_adj = adj[p0]
            p0_m = adj_mask[p0]
            p0_n = len(p0_adj)
            p0_row0 = p0 < 9
            base0 = p0 * 81
            base1 = 6561 + p0 * 81
            for p1 in range(CELL_COUNT):
                if p0 == p1:
                    continue
                s0 = base0 + p1
                if p1 >= 72:
                    res[s0] = 2  # P1 just reached row 8 -> P0 loses now
                    q.append(s0)
                else:
                    if p0_m & (1 << p1):
                        mvs = _pawn_moves_raw(p0, p1, u_ok, d_ok, l_ok, r_ok)
                        outdeg[s0] = len(mvs)
                        cbase = 6561 + p1
                        for np0 in mvs:
                            preds[np0 * 81 + cbase].append(s0)
                    else:
                        outdeg[s0] = p0_n
                        cbase = 6561 + p1
                        for np0 in p0_adj:
                            preds[np0 * 81 + cbase].append(s0)

                s1 = base1 + p1
                if p0_row0:
                    res[s1] = 2  # P0 just reached row 0 -> P1 loses now
                    q.append(s1)
                else:
                    if adj_mask[p1] & (1 << p0):
                        mvs = _pawn_moves_raw(p1, p0, u_ok, d_ok, l_ok, r_ok)
                        outdeg[s1] = len(mvs)
                        cbase = p0 * 81
                        for np1 in mvs:
                            preds[cbase + np1].append(s1)
                    else:
                        outdeg[s1] = len(adj[p1])
                        cbase = p0 * 81
                        for np1 in adj[p1]:
                            preds[cbase + np1].append(s1)

        while q:
            c = q.popleft()
            rc = res[c]
            dc = dtm[c] + 1
            if rc == 2:  # child is LOSS -> predecessors WIN
                for u in preds[c]:
                    if res[u] == 0:
                        res[u] = 1
                        dtm[u] = dc
                        q.append(u)
            else:  # child is WIN -> decrement outdeg of predecessors
                for u in preds[c]:
                    if res[u] == 0:
                        od = outdeg[u] - 1
                        outdeg[u] = od
                        if od == 0:
                            res[u] = 2
                            dtm[u] = dc
                            q.append(u)

        return res, dtm


# ---------------------------------------------------------------------------
# Pure Rule-Based Evaluation & Focused Move Generation
# ---------------------------------------------------------------------------

def _evaluate_rule_based(
    b: _Board,
    dist_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]],
    eval_cache: Dict[int, int],
) -> int:
    cached = eval_cache.get(b.zob)
    if cached is not None:
        return cached

    stm = b.turn
    opp = 1 - stm
    p0, p1 = b.pawn[0], b.pawn[1]
    if stm == 0:
        if p0 < 9:
            return EVAL_CLAMP
        if p1 >= 72:
            return -EVAL_CLAMP
    else:
        if p1 >= 72:
            return EVAL_CLAMP
        if p0 < 9:
            return -EVAL_CLAMP

    d0_tbl, d1_tbl = b.compute_dist_tables(dist_cache)
    if stm == 0:
        p_stm, p_opp = p0, p1
        d_tbl_stm, d_tbl_opp = d0_tbl, d1_tbl
        w_stm, w_opp = b.wl[0], b.wl[1]
    else:
        p_stm, p_opp = p1, p0
        d_tbl_stm, d_tbl_opp = d1_tbl, d0_tbl
        w_stm, w_opp = b.wl[1], b.wl[0]

    d_stm = d_tbl_stm[p_stm]
    d_opp = d_tbl_opp[p_opp]

    # 1. Path lead + side-to-move tempo
    score = 100 * (d_opp - d_stm) + 43

    # 2. Phase-scaled wall lead
    min_d = d_stm if d_stm < d_opp else d_opp
    if min_d > 8:
        min_d = 8
    wall_val = 110 + (50 * min_d) // 8
    score += wall_val * (w_stm - w_opp)

    # 3. Effective walls (capped by opponent distance)
    eff_stm = w_stm if w_stm < d_opp else d_opp
    eff_opp = w_opp if w_opp < d_stm else d_stm
    score += 32 * (eff_stm - eff_opp)

    # 4. Zero-wall race term
    if w_stm == 0 and w_opp == 0:
        score += 650 if d_stm <= d_opp else -650

    # 5. Wall exhaustion advantage (corrected Duel9 sign)
    if w_opp == 0 and w_stm > 0:
        score += 45 + 20 * (w_stm if w_stm < 4 else 4)
    elif w_stm == 0 and w_opp > 0:
        score -= 45 + 20 * (w_opp if w_opp < 4 else 4)

    # 6. Close-quarters pawn jump / congestion tempo
    manh = abs(CELL_R[p_stm] - CELL_R[p_opp]) + abs(CELL_C[p_stm] - CELL_C[p_opp])
    if manh <= 2:
        pmoves = _pawn_moves_raw(p_stm, p_opp, b.u_ok, b.d_ok, b.l_ok, b.r_ok)
        if pmoves:
            best_nd = d_tbl_stm[pmoves[0]]
            for pm in pmoves[1:]:
                nd = d_tbl_stm[pm]
                if nd < best_nd:
                    best_nd = nd
            if best_nd <= d_stm - 2:
                score += 65
            elif best_nd >= d_stm:
                score -= 55

    # 7. Corridor exit mobility (at least 2 shortest-path exits when opp has walls)
    u_ok, d_ok, l_ok, r_ok = b.u_ok, b.d_ok, b.l_ok, b.r_ok
    if w_opp > 0:
        bs = 1 << p_stm
        tgt = d_stm - 1
        ex_s = (
            ((u_ok & bs) and d_tbl_stm[p_stm - 9] == tgt)
            + ((d_ok & bs) and d_tbl_stm[p_stm + 9] == tgt)
            + ((l_ok & bs) and d_tbl_stm[p_stm - 1] == tgt)
            + ((r_ok & bs) and d_tbl_stm[p_stm + 1] == tgt)
        )
        if ex_s >= 2:
            score += 18
    if w_stm > 0:
        bo = 1 << p_opp
        tgt = d_opp - 1
        ex_o = (
            ((u_ok & bo) and d_tbl_opp[p_opp - 9] == tgt)
            + ((d_ok & bo) and d_tbl_opp[p_opp + 9] == tgt)
            + ((l_ok & bo) and d_tbl_opp[p_opp - 1] == tgt)
            + ((r_ok & bo) and d_tbl_opp[p_opp + 1] == tgt)
        )
        if ex_o >= 2:
            score -= 18

    # 8. Centrality tie-breaker
    score += 3 * (CELL_CENTER_DIST[p_opp] - CELL_CENTER_DIST[p_stm])

    if score > EVAL_CLAMP:
        score = EVAL_CLAMP
    elif score < -EVAL_CLAMP:
        score = -EVAL_CLAMP

    if len(eval_cache) > 200000:
        eval_cache.clear()
    eval_cache[b.zob] = score
    return score


def _generate_ordered_candidates(
    b: _Board,
    is_root: bool,
    dist_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]],
    order_cache: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return list of (base_priority, move_int) ordered best-first."""
    ckey = (b.zob, 1 if is_root else 0)
    cached = order_cache.get(ckey)
    if cached is not None:
        return cached

    stm = b.turn
    opp = 1 - stm
    d0_tbl, d1_tbl = b.compute_dist_tables(dist_cache)
    d_tbl_stm = d0_tbl if stm == 0 else d1_tbl
    d_tbl_opp = d1_tbl if stm == 0 else d0_tbl
    p_stm = b.pawn[stm]
    p_opp = b.pawn[opp]
    d_stm = d_tbl_stm[p_stm]
    d_opp = d_tbl_opp[p_opp]
    goal_stm = GOAL_MASKS[stm]
    goal_opp = GOAL_MASKS[opp]

    cands: List[Tuple[int, int]] = []

    # 1. All legal pawn moves
    pmoves = _pawn_moves_raw(p_stm, p_opp, b.u_ok, b.d_ok, b.l_ok, b.r_ok)
    for m in pmoves:
        nd = d_tbl_stm[m]
        if nd == 0:
            prio = 50000 - CELL_CENTER_DIST[m]
        elif nd < d_stm:
            prio = 10000 + 250 * (d_stm - nd) - 5 * CELL_CENTER_DIST[m]
        elif nd == d_stm:
            prio = 1000 - 5 * CELL_CENTER_DIST[m]
        else:
            prio = -5000 - 200 * (nd - d_stm) - 5 * CELL_CENTER_DIST[m]
        cands.append((prio, m))

    # 2. Focused wall candidates if stm has walls remaining
    if b.wl[stm] > 0:
        w0_h, w0_v, w1_h, w1_v = b.witness_cut_masks(d0_tbl, d1_tbl)
        w_stm_h, w_stm_v = (w0_h, w0_v) if stm == 0 else (w1_h, w1_v)
        w_opp_h, w_opp_v = (w1_h, w1_v) if stm == 0 else (w0_h, w0_v)
        dag_h, dag_v = b.dag_cut_masks(opp, d_tbl_opp)

        h_occ, v_occ = b.h_occ, b.v_occ
        u_ok, d_ok, l_ok, r_ok = b.u_ok, b.d_ok, b.l_ok, b.r_ok
        occ_all = h_occ | v_occ

        k_pos = 14 if is_root else 10
        k_tac = 6 if is_root else 4

        pos_walls: List[Tuple[int, int]] = []
        backup_dag: List[Tuple[int, int]] = []

        def _eval_wall_deltas(is_v: bool, wi: int) -> Tuple[bool, int, int]:
            bit = 1 << wi
            if is_v:
                if (v_occ & V_OVERLAP_MASK[wi]) or (h_occ & bit):
                    return False, 0, 0
                l2 = l_ok ^ V_WEST_BLK[wi]
                r2 = r_ok ^ V_EAST_BLK[wi]
                u2, d2 = u_ok, d_ok
                stm_cut = bit & w_stm_v
                opp_wit = bit & w_opp_v
                in_dag = bit & dag_v
            else:
                if (h_occ & H_OVERLAP_MASK[wi]) or (v_occ & bit):
                    return False, 0, 0
                u2 = u_ok ^ H_NORTH_BLK[wi]
                d2 = d_ok ^ H_SOUTH_BLK[wi]
                l2, r2 = l_ok, r_ok
                stm_cut = bit & w_stm_h
                opp_wit = bit & w_opp_h
                in_dag = bit & dag_h

            isolated = (occ_all & ANCHOR_MASK_5X5[wi]) == 0

            if not stm_cut:
                my_delta = 0
            else:
                nd_s = _bb_dist(p_stm, goal_stm, u2, d2, l2, r2)
                if nd_s < 0:
                    return False, 0, 0
                my_delta = nd_s - d_stm

            if not in_dag:
                op_delta = 0
                if (not isolated) and opp_wit and not _has_path_bb(p_opp, goal_opp, u2, d2, l2, r2):
                    return False, 0, 0
            else:
                nd_o = _bb_dist(p_opp, goal_opp, u2, d2, l2, r2)
                if nd_o < 0:
                    return False, 0, 0
                op_delta = nd_o - d_opp

            return True, my_delta, op_delta

        # 2a. Opponent shortest-path DAG cut walls
        m = dag_h
        while m:
            lsb = m & -m
            wi = lsb.bit_length() - 1
            m ^= lsb
            ok, md, od = _eval_wall_deltas(False, wi)
            if ok:
                gain = od - md
                if gain >= 1:
                    pos_walls.append((gain * 150 + od * 25 - md * 15, H_BASE + wi))
                elif od >= 1 and md == od and d_opp <= 4:
                    backup_dag.append((od * 20 - md * 25, H_BASE + wi))

        m = dag_v
        while m:
            lsb = m & -m
            wi = lsb.bit_length() - 1
            m ^= lsb
            ok, md, od = _eval_wall_deltas(True, wi)
            if ok:
                gain = od - md
                if gain >= 1:
                    pos_walls.append((gain * 150 + od * 25 - md * 15, V_BASE + wi))
                elif od >= 1 and md == od and d_opp <= 4:
                    backup_dag.append((od * 20 - md * 25, V_BASE + wi))

        pos_walls.sort(reverse=True)
        for prio, wall_m in pos_walls[:k_pos]:
            cands.append((2000 + prio, wall_m))

        # 2b. Tactical / template / focus walls when opp has walls or pawns are close
        manh = abs(CELL_R[p_stm] - CELL_R[p_opp]) + abs(CELL_C[p_stm] - CELL_C[p_opp])
        if b.wl[opp] > 0 or manh <= 2:
            tac_h = 0
            tac_v = 0
            mh = h_occ
            while mh:
                lsb = mh & -mh
                wi = lsb.bit_length() - 1
                mh ^= lsb
                tac_h |= HW_TPL_H[wi]
                tac_v |= HW_TPL_V[wi]
            mv = v_occ
            while mv:
                lsb = mv & -mv
                wi = lsb.bit_length() - 1
                mv ^= lsb
                tac_h |= VW_TPL_H[wi]
                tac_v |= VW_TPL_V[wi]

            focus = CELL_FOCUS_SLOTS_MASK[p_stm] | CELL_FOCUS_SLOTS_MASK[p_opp]
            tac_h |= focus | (EDGE_H_COLS_MASK & (CELL_FOCUS_SLOTS_MASK[p_opp] | CELL_FOCUS_SLOTS_MASK[p_stm]))
            tac_v |= focus
            tac_h &= ~dag_h
            tac_v &= ~dag_v

            r_o, c_o = CELL_R[p_opp], CELL_C[p_opp]
            tac_walls: List[Tuple[int, int]] = []

            m = tac_h
            while m:
                lsb = m & -m
                wi = lsb.bit_length() - 1
                m ^= lsb
                ok, md, _ = _eval_wall_deltas(False, wi)
                if ok and md <= 0:
                    wr, wc = wi >> 3, wi & 7
                    prox = 20 - (abs(wr - r_o) + abs(wc - c_o)) * 2
                    if (EDGE_H_COLS_MASK >> wi) & 1:
                        prox += 4
                    tac_walls.append((prox, H_BASE + wi))

            m = tac_v
            while m:
                lsb = m & -m
                wi = lsb.bit_length() - 1
                m ^= lsb
                ok, md, _ = _eval_wall_deltas(True, wi)
                if ok and md <= 0:
                    wr, wc = wi >> 3, wi & 7
                    prox = 20 - (abs(wr - r_o) + abs(wc - c_o)) * 2
                    tac_walls.append((prox, V_BASE + wi))

            tac_walls.sort(reverse=True)
            for prio, wall_m in tac_walls[:k_tac]:
                cands.append((400 + prio, wall_m))
        elif not pos_walls and backup_dag:
            backup_dag.sort(reverse=True)
            for prio, wall_m in backup_dag[:2]:
                cands.append((300 + prio, wall_m))

    cands.sort(reverse=True)
    if len(order_cache) > 120000:
        order_cache.clear()
    order_cache[ckey] = cands
    return cands


# ---------------------------------------------------------------------------
# Iterative-Deepening PVS Negamax Search Engine
# ---------------------------------------------------------------------------

class _SearchEngine:
    def __init__(self, oracle: _Oracle) -> None:
        self.oracle = oracle
        self.tt: Dict[int, Tuple[int, int, int, int]] = {}
        self.dist_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        self.eval_cache: Dict[int, int] = {}
        self.order_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.killers: List[List[int]] = [[0, 0] for _ in range(MAX_ID_DEPTH + 4)]
        self.history: List[int] = [0] * HIST_ACTIONS
        self.cmh: List[int] = [0] * (HIST_ACTIONS * HIST_ACTIONS)
        self.nodes = 0
        self.hard_stop = 0.0
        self.aborted = False
        self.root_scores: Dict[int, int] = {}
        self.oracle_builds = 0

    def reset_match(self) -> None:
        self.tt.clear()
        self.order_cache.clear()
        for k in self.killers:
            k[0] = 0
            k[1] = 0
        self.history = [0] * HIST_ACTIONS
        self.cmh = [0] * (HIST_ACTIONS * HIST_ACTIONS)

    def _age_tables(self) -> None:
        h = self.history
        for i in range(HIST_ACTIONS):
            h[i] >>= 1
        cm = self.cmh
        for i, v in enumerate(cm):
            if v:
                cm[i] = v >> 1
        if len(self.tt) > 300000:
            self.tt.clear()
        if len(self.order_cache) > 100000:
            self.order_cache.clear()

    @staticmethod
    def _to_tt(score: int, ply: int) -> int:
        if score >= MATE_BAND:
            return score + ply
        if score <= -MATE_BAND:
            return score - ply
        return score

    @staticmethod
    def _from_tt(score: int, ply: int) -> int:
        if score >= MATE_BAND:
            return score - ply
        if score <= -MATE_BAND:
            return score + ply
        return score

    def _pvs(
        self,
        b: _Board,
        ply: int,
        depth: int,
        alpha: int,
        beta: int,
        prev_code: int,
    ) -> Tuple[int, int]:
        self.nodes += 1
        if (self.nodes & 127) == 0 and time.perf_counter() >= self.hard_stop:
            self.aborted = True
            return 0, 0

        stm = b.turn
        opp = 1 - stm
        if (GOAL_MASKS[opp] >> b.pawn[opp]) & 1:
            return -(MATE - ply), 0
        if (GOAL_MASKS[stm] >> b.pawn[stm]) & 1:
            return MATE - ply, 0

        # Exact 0-wall oracle evaluation if cached (or single fast build early in search)
        if b.wl[0] == 0 and b.wl[1] == 0:
            okey = (b.h_occ, b.v_occ)
            entry = self.oracle.cache.get(okey)
            if entry is None and self.oracle_builds == 0 and (self.hard_stop - time.perf_counter()) > 0.25:
                self.oracle_builds += 1
                entry = self.oracle.get(b)
            if entry is not None:
                res_tbl, dtm_tbl = entry
                idx = ((stm * CELL_COUNT) + b.pawn[0]) * CELL_COUNT + b.pawn[1]
                r = res_tbl[idx]
                if r == 1:
                    return MATE - (ply + dtm_tbl[idx]), 0
                if r == 2:
                    return -(MATE - (ply + dtm_tbl[idx])), 0
                return 0, 0

        if depth <= 0:
            return _evaluate_rule_based(b, self.dist_cache, self.eval_cache), 0

        zob = b.zob
        orig_alpha = alpha
        tt_move = 0
        tt_entry = self.tt.get(zob)
        if tt_entry is not None:
            t_depth, t_flag, t_raw, t_move = tt_entry
            tt_move = t_move
            t_score = self._from_tt(t_raw, ply)
            if ply > 0 and t_depth >= depth:
                if t_flag == TT_EXACT:
                    return t_score, t_move
                if t_flag == TT_LOWER:
                    if t_score > alpha:
                        alpha = t_score
                elif t_flag == TT_UPPER:
                    if t_score < beta:
                        beta = t_score
                if alpha >= beta:
                    return t_score, t_move

        # Reverse Futility Pruning on non-PV internal nodes
        if ply > 0 and (beta - alpha) == 1 and depth <= 3 and abs(beta) < MATE_BAND:
            st_eval = _evaluate_rule_based(b, self.dist_cache, self.eval_cache)
            if st_eval - RFP_MARGIN * depth >= beta:
                return st_eval, 0

        base_cands = _generate_ordered_candidates(b, ply == 0, self.dist_cache, self.order_cache)
        if not base_cands:
            return _evaluate_rule_based(b, self.dist_cache, self.eval_cache), 0

        k0, k1 = self.killers[ply]
        hist = self.history
        cmh = self.cmh
        cmh_off = (prev_code << 8) if prev_code >= 0 else -1

        scored: List[Tuple[int, int]] = []
        for base_p, m in base_cands:
            if m == tt_move:
                op = 1_000_000
            elif m == k0:
                op = 500_000
            elif m == k1:
                op = 400_000
            else:
                code = _action_code(m)
                op = base_p + hist[code] + (cmh[cmh_off | code] if cmh_off >= 0 else 0)
            scored.append((op, m))
        scored.sort(reverse=True)

        best_score = -INF
        best_move = scored[0][1]

        for idx, (_, m) in enumerate(scored):
            code = _action_code(m)
            b.make_move(m)
            if idx == 0:
                sc, _ = self._pvs(b, ply + 1, depth - 1, -beta, -alpha, code)
                sc = -sc
            else:
                r = 1 if (idx >= 4 and depth >= 3 and m >= 100) else 0
                sc, _ = self._pvs(b, ply + 1, depth - 1 - r, -alpha - 1, -alpha, code)
                sc = -sc
                if not self.aborted and r > 0 and sc > alpha:
                    sc, _ = self._pvs(b, ply + 1, depth - 1, -alpha - 1, -alpha, code)
                    sc = -sc
                if not self.aborted and alpha < sc < beta:
                    sc, _ = self._pvs(b, ply + 1, depth - 1, -beta, -alpha, code)
                    sc = -sc
            b.unmake_move()

            if self.aborted:
                return 0, 0

            if ply == 0:
                self.root_scores[m] = sc

            if sc > best_score:
                best_score = sc
                best_move = m
                if sc > alpha:
                    alpha = sc
                    if alpha >= beta:
                        if m != k0:
                            self.killers[ply][1] = k0
                            self.killers[ply][0] = m
                        bonus = depth * depth
                        nh = hist[code] + bonus
                        hist[code] = CMH_MAX if nh > CMH_MAX else nh
                        if cmh_off >= 0:
                            cidx = cmh_off | code
                            nc = cmh[cidx] + bonus
                            cmh[cidx] = CMH_MAX if nc > CMH_MAX else nc
                        self.tt[zob] = (depth, TT_LOWER, self._to_tt(best_score, ply), best_move)
                        return best_score, best_move

        flag = TT_EXACT if best_score > orig_alpha else TT_UPPER
        self.tt[zob] = (depth, flag, self._to_tt(best_score, ply), best_move)
        return best_score, best_move

    def search(self, b: _Board, deadline_start: float) -> Tuple[int, int, int, int]:
        """Run iterative-deepening PVS. Returns (best_move, best_score, completed_depth, total_nodes)."""
        self._age_tables()
        self.nodes = 0
        self.aborted = False
        self.oracle_builds = 0
        self.root_scores = {}

        soft_stop = deadline_start + (TIME_BUDGET_MS * SOFT_PCT) / 1000.0
        self.hard_stop = deadline_start + (TIME_BUDGET_MS * HARD_PCT) / 1000.0

        cands = _generate_ordered_candidates(b, True, self.dist_cache, self.order_cache)
        best_move = cands[0][1] if cands else b.gen_pawn_moves()[0]
        best_score = 0
        completed_depth = 0
        total_nodes = 0
        prev_iter_nodes = 1
        prev_iter_ms = 0.5
        best_root_scores: Dict[int, int] = {}

        for depth in range(1, MAX_ID_DEPTH + 1):
            now = time.perf_counter()
            if depth >= 2:
                if now >= soft_stop:
                    break
                if depth >= 3:
                    # Predict next iteration time from effective branching factor
                    ebf = self.nodes / max(1, prev_iter_nodes)
                    if ebf < 2.2:
                        ebf = 2.2
                    elif ebf > 5.5:
                        ebf = 5.5
                    pred_sec = (prev_iter_ms / 1000.0) * ebf
                    if now + pred_sec > self.hard_stop:
                        break

            iter_start = time.perf_counter()
            self.nodes = 0
            self.aborted = False
            self.root_scores = {}

            sc, mv = self._pvs(b, 0, depth, -INF, INF, -1)
            iter_ms = (time.perf_counter() - iter_start) * 1000.0
            total_nodes += self.nodes

            if self.aborted:
                break

            if mv != 0:
                best_move = mv
                best_score = sc
                completed_depth = depth
                best_root_scores = dict(self.root_scores)
            prev_iter_nodes = max(1, self.nodes)
            prev_iter_ms = max(0.2, iter_ms)

            if depth >= 2 and abs(best_score) >= MATE_BAND:
                break

        self.root_scores = best_root_scores
        return best_move, best_score, completed_depth, total_nodes


# ---------------------------------------------------------------------------
# Layer 0 Safeguards (S0 - S6)
# ---------------------------------------------------------------------------

def _opp_immediate_wins(b: _Board, opp: int) -> List[int]:
    opp_goal = GOAL0_MASK if opp == 0 else GOAL1_MASK
    mvs = b.gen_pawn_moves(opp)
    return [m for m in mvs if (opp_goal >> m) & 1]


def _legal_wall_blocks_win(
    b: _Board,
    opp: int,
    dist_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]],
    d_tbl_stm: Tuple[int, ...],
    d_tbl_opp: Tuple[int, ...],
    w0_h: int,
    w0_v: int,
    w1_h: int,
    w1_v: int,
) -> Optional[int]:
    """Find best legal wall (for stm) eliminating all immediate 1-ply opp wins."""
    stm = 1 - opp
    if b.wl[stm] <= 0:
        return None
    p_opp = b.pawn[opp]
    p_stm = b.pawn[stm]
    d_stm = d_tbl_stm[p_stm]
    d_opp = d_tbl_opp[p_opp]
    r_o, c_o = CELL_R[p_opp], CELL_C[p_opp]

    # Candidate walls touching 2-cell neighborhood around opp (sufficient for 1-ply win block)
    cand_mask = CELL_FOCUS_SLOTS_MASK[p_opp]
    best_w: Optional[int] = None
    best_key = (-999999, -999)

    for is_v in (False, True):
        m = cand_mask
        while m:
            lsb = m & -m
            wi = lsb.bit_length() - 1
            m ^= lsb
            if not b.wall_legal_flags(is_v, wi, w0_h, w0_v, w1_h, w1_v):
                continue
            wm = (V_BASE if is_v else H_BASE) + wi
            b.make_move(wm)
            wins_after = _opp_immediate_wins(b, opp)
            if not wins_after:
                nd0, nd1 = b.compute_dist_tables(dist_cache)
                my_nd = (nd0 if stm == 0 else nd1)[p_stm]
                op_nd = (nd1 if stm == 0 else nd0)[p_opp]
                gain = (op_nd - d_opp) - (my_nd - d_stm)
                key = (gain * 100 - (my_nd - d_stm) * 20, -abs(CELL_R[p_opp] - (wi >> 3)))
                if key > best_key:
                    best_key = key
                    best_w = wm
            b.unmake_move()

    if best_w is not None:
        return best_w

    # Exhaustive fallback over all 128 wall slots if local neighborhood missed a subtle case
    for is_v in (False, True):
        for wi in range(WALL_SLOTS):
            if (cand_mask >> wi) & 1:
                continue
            if not b.wall_legal_flags(is_v, wi, w0_h, w0_v, w1_h, w1_v):
                continue
            wm = (V_BASE if is_v else H_BASE) + wi
            b.make_move(wm)
            if not _opp_immediate_wins(b, opp):
                nd0, nd1 = b.compute_dist_tables(dist_cache)
                my_nd = (nd0 if stm == 0 else nd1)[p_stm]
                op_nd = (nd1 if stm == 0 else nd0)[p_opp]
                b.unmake_move()
                gain = (op_nd - d_opp) - (my_nd - d_stm)
                key = (gain * 100 - (my_nd - d_stm) * 20, -(wi))
                if key > best_key:
                    best_key = key
                    best_w = wm
            else:
                b.unmake_move()
    return best_w


def _is_unwallable_at_dist_1(b: _Board, q: int, opp: int) -> bool:
    """Check if opp standing at dist-1 square q cannot be stopped by any single wall."""
    stm = 1 - opp
    if b.wl[stm] <= 0:
        return True
    orig_p = b.pawn[opp]
    b.pawn[opp] = q
    wins = _opp_immediate_wins(b, opp)
    if not wins:
        b.pawn[opp] = orig_p
        return False
    # Check if any single wall near q blocks all wins from q
    r_q, c_q = CELL_R[q], CELL_C[q]
    blocked = False
    for is_v in (False, True):
        for wr in range(max(0, r_q - 2), min(8, r_q + 2)):
            for wc in range(max(0, c_q - 2), min(8, c_q + 2)):
                wi = wr * 8 + wc
                bit = 1 << wi
                if is_v:
                    if (b.v_occ & V_OVERLAP_MASK[wi]) or (b.h_occ & bit):
                        continue
                    l2 = b.l_ok ^ V_WEST_BLK[wi]
                    r2 = b.r_ok ^ V_EAST_BLK[wi]
                    u2, d2 = b.u_ok, b.d_ok
                else:
                    if (b.h_occ & H_OVERLAP_MASK[wi]) or (b.v_occ & bit):
                        continue
                    u2 = b.u_ok ^ H_NORTH_BLK[wi]
                    d2 = b.d_ok ^ H_SOUTH_BLK[wi]
                    l2, r2 = b.l_ok, b.r_ok
                if not (
                    _has_path_bb(b.pawn[stm], GOAL_MASKS[stm], u2, d2, l2, r2)
                    and _has_path_bb(q, GOAL_MASKS[opp], u2, d2, l2, r2)
                ):
                    continue
                opp_goal = GOAL0_MASK if opp == 0 else GOAL1_MASK
                mvs = _pawn_moves_raw(q, b.pawn[stm], u2, d2, l2, r2)
                if not any((opp_goal >> mv) & 1 for mv in mvs):
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            break
    b.pawn[opp] = orig_p
    return not blocked


def _pre_search_safeguard(
    b: _Board,
    oracle: _Oracle,
    dist_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]],
    d0_tbl: Tuple[int, ...],
    d1_tbl: Tuple[int, ...],
) -> Tuple[Optional[int], str]:
    """Run S1-S4 pre-search safeguards. Returns (forced_move, tag) or (None, '')."""
    stm = b.turn
    opp = 1 - stm
    d_tbl_stm = d0_tbl if stm == 0 else d1_tbl
    d_tbl_opp = d1_tbl if stm == 0 else d0_tbl
    p_stm = b.pawn[stm]
    p_opp = b.pawn[opp]
    d_stm = d_tbl_stm[p_stm]
    d_opp = d_tbl_opp[p_opp]
    goal_stm = GOAL_MASKS[stm]

    # S1a: Immediate 1-ply win for stm
    stm_pmoves = b.gen_pawn_moves(stm)
    for pm in stm_pmoves:
        if (goal_stm >> pm) & 1:
            return pm, "S1_WIN"

    # S1b: Opponent immediate 1-ply win threat
    opp_wins = _opp_immediate_wins(b, opp)
    if opp_wins:
        # First check jump-bridge escape / body-block by a pawn move (saves a wall!)
        best_pm: Optional[int] = None
        best_pm_d = 999
        for pm in stm_pmoves:
            b.make_move(pm)
            if not _opp_immediate_wins(b, opp):
                nd = d_tbl_stm[pm]
                if nd < best_pm_d:
                    best_pm_d = nd
                    best_pm = pm
            b.unmake_move()
        if best_pm is not None:
            return best_pm, "S1_PAWN_ESCAPE"

        # Otherwise emergency blocking wall
        w0_h, w0_v, w1_h, w1_v = b.witness_cut_masks(d0_tbl, d1_tbl)
        blk = _legal_wall_blocks_win(b, opp, dist_cache, d_tbl_stm, d_tbl_opp, w0_h, w0_v, w1_h, w1_v)
        if blk is not None:
            return blk, "S1_WALL_BLOCK"

    # S2 / S3: 2-ply un-wallable threat / anti-fork when opp is close and we don't trivially out-race
    if d_opp <= 2 and b.wl[stm] > 0 and not (d_stm <= d_opp and b.wl[opp] == 0):
        opp_nexts = b.gen_pawn_moves(opp)
        unwallable_targets = [
            q for q in opp_nexts
            if d_tbl_opp[q] <= 1 and _is_unwallable_at_dist_1(b, q, opp)
        ]
        # Also trigger anti-fork if opp has >= 2 distinct advances to dist 1 while stm has == 1 wall left
        dist1_advances = [q for q in opp_nexts if d_tbl_opp[q] <= 1]
        if unwallable_targets or (len(dist1_advances) >= 2 and b.wl[stm] == 1):
            w0_h, w0_v, w1_h, w1_v = b.witness_cut_masks(d0_tbl, d1_tbl)
            dag_h, dag_v = b.dag_cut_masks(opp, d_tbl_opp)
            cand_h = dag_h | CELL_FOCUS_SLOTS_MASK[p_opp]
            cand_v = dag_v | CELL_FOCUS_SLOTS_MASK[p_opp]

            best_w: Optional[int] = None
            best_score = -999999
            for is_v, cmask in ((False, cand_h), (True, cand_v)):
                m = cmask
                while m:
                    lsb = m & -m
                    wi = lsb.bit_length() - 1
                    m ^= lsb
                    if not b.wall_legal_flags(is_v, wi, w0_h, w0_v, w1_h, w1_v):
                        continue
                    wm = (V_BASE if is_v else H_BASE) + wi
                    b.make_move(wm)
                    nd0, nd1 = b.compute_dist_tables(dist_cache)
                    op_tbl2 = nd1 if stm == 0 else nd0
                    my_tbl2 = nd0 if stm == 0 else nd1
                    op_d2 = op_tbl2[p_opp]
                    my_d2 = my_tbl2[p_stm]
                    # Check whether all un-wallable threats/forks are neutralized
                    opp_m2 = b.gen_pawn_moves(opp)
                    still_bad = any(
                        op_tbl2[q] <= 1 and _is_unwallable_at_dist_1(b, q, opp)
                        for q in opp_m2
                    )
                    if not still_bad:
                        gain = (op_d2 - d_opp) - (my_d2 - d_stm)
                        sc = gain * 200 + (op_d2 - d_opp) * 60 - (my_d2 - d_stm) * 40
                        if sc > best_score:
                            best_score = sc
                            best_w = wm
                    b.unmake_move()
            if best_w is not None:
                return best_w, "S2_ANTI_THREAT"

    # S4: Exact 0-wall endgame solver via retrograde BFS _Oracle
    if b.wl[0] == 0 and b.wl[1] == 0 and stm_pmoves:
        res_tbl, dtm_tbl = oracle.get(b)
        cur_idx = ((stm * CELL_COUNT) + b.pawn[0]) * CELL_COUNT + b.pawn[1]
        cur_r = res_tbl[cur_idx]
        if cur_r == 1:
            # Forced win: pick child with LOSS (2) and minimum DTM
            best_pm = stm_pmoves[0]
            best_dtm = 999999
            for pm in stm_pmoves:
                b.make_move(pm)
                ci = ((1 - stm) * CELL_COUNT + b.pawn[0]) * CELL_COUNT + b.pawn[1]
                cr = res_tbl[ci]
                cd = dtm_tbl[ci]
                b.unmake_move()
                if cr == 2 and cd < best_dtm:
                    best_dtm = cd
                    best_pm = pm
            return best_pm, "S4_ORACLE_WIN"
        if cur_r == 2:
            # Forced loss: maximize resistance DTM, tiebreak shortest distance
            best_pm = stm_pmoves[0]
            best_key = (-1, -999)
            for pm in stm_pmoves:
                b.make_move(pm)
                ci = ((1 - stm) * CELL_COUNT + b.pawn[0]) * CELL_COUNT + b.pawn[1]
                cd = dtm_tbl[ci]
                b.unmake_move()
                key = (cd, -d_tbl_stm[pm])
                if key > best_key:
                    best_key = key
                    best_pm = pm
            return best_pm, "S4_ORACLE_RESIST"

    return None, ""


def _post_search_safeguard(
    b: _Board,
    chosen: int,
    best_score: int,
    root_scores: Dict[int, int],
    dist_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]],
    d0_tbl: Tuple[int, ...],
    d1_tbl: Tuple[int, ...],
    visited_states: Dict[int, int],
    recent_cells: deque,
) -> Tuple[int, str]:
    """S5 root self-harm/waste filter + S6 anti-shuffle."""
    stm = b.turn
    opp = 1 - stm
    d_tbl_stm = d0_tbl if stm == 0 else d1_tbl
    d_tbl_opp = d1_tbl if stm == 0 else d0_tbl
    p_stm = b.pawn[stm]
    p_opp = b.pawn[opp]
    d_stm = d_tbl_stm[p_stm]
    d_opp = d_tbl_opp[p_opp]

    def _is_move_bad(m: int) -> Tuple[bool, int]:
        """Return (is_bad, result_zob) for S5 validation."""
        b.make_move(m)
        nz = b.zob
        opp_wins = _opp_immediate_wins(b, opp)
        if opp_wins:
            b.unmake_move()
            return True, nz
        if m >= 100:
            nd0, nd1 = b.compute_dist_tables(dist_cache)
            my_d2 = (nd0 if stm == 0 else nd1)[p_stm]
            op_d2 = (nd1 if stm == 0 else nd0)[p_opp]
            op_delta = op_d2 - d_opp
            my_delta = my_d2 - d_stm
            # Filter pure wall waste vs wall-less opp, or self-harm without opp delay
            if op_delta <= 0 and my_delta > 0:
                b.unmake_move()
                return True, nz
            if b.wl[opp] == 0 and op_delta <= 0 and abs(CELL_R[p_stm] - CELL_R[p_opp]) + abs(CELL_C[p_stm] - CELL_C[p_opp]) > 2:
                b.unmake_move()
                return True, nz
        b.unmake_move()
        return False, nz

    tag = "PVS"
    bad, chosen_nz = _is_move_bad(chosen)
    if bad and root_scores:
        # Try ranked root alternatives that pass S5
        alts = sorted(root_scores.items(), key=lambda kv: kv[1], reverse=True)
        for alt_m, _ in alts:
            alt_bad, alt_nz = _is_move_bad(alt_m)
            if not alt_bad:
                chosen = alt_m
                chosen_nz = alt_nz
                bad = False
                tag = "S5_FILTERED"
                break
        if bad:
            # Fallback to shortest-distance safe pawn move
            for pm in b.gen_pawn_moves(stm):
                b.make_move(pm)
                ow = _opp_immediate_wins(b, opp)
                pnz = b.zob
                b.unmake_move()
                if not ow:
                    chosen = pm
                    chosen_nz = pnz
                    tag = "S5_SAFE_PAWN"
                    break

    # S6: Anti-shuffle / cycle break (skip on forced mate)
    if chosen < 100 and abs(best_score) < MATE_BAND:
        nd = d_tbl_stm[chosen]
        is_repeat = visited_states.get(chosen_nz, 0) >= 1 and nd >= d_stm
        is_pingpong = len(recent_cells) >= 2 and chosen == recent_cells[-2] and nd >= d_stm
        if is_repeat or is_pingpong:
            alts = sorted(root_scores.items(), key=lambda kv: kv[1], reverse=True) if root_scores else []
            for alt_m, alt_sc in alts:
                if alt_m == chosen:
                    continue
                if alt_sc < best_score - 120:
                    break
                alt_bad, alt_nz = _is_move_bad(alt_m)
                if alt_bad:
                    continue
                if alt_m < 100:
                    alt_nd = d_tbl_stm[alt_m]
                    if visited_states.get(alt_nz, 0) == 0 and (alt_nd < d_stm or len(recent_cells) < 2 or alt_m != recent_cells[-2]):
                        chosen = alt_m
                        tag = "S6_ANTISHUFFLE"
                        break
                else:
                    if visited_states.get(alt_nz, 0) == 0:
                        chosen = alt_m
                        tag = "S6_ANTISHUFFLE"
                        break

    return chosen, tag


def _sanitize_action(action_dict: Dict[str, Any], observation: Dict[str, Any], b: _Board, d_tbl_stm: Tuple[int, ...]) -> Dict[str, Any]:
    """Guarantee returned action is in authoritative observation legal sets."""
    legal_moves = observation.get("legal_moves") or []
    legal_walls = observation.get("legal_walls") or []

    if action_dict.get("type") == "move":
        if action_dict.get("to") in legal_moves:
            return action_dict
    elif action_dict.get("type") == "wall":
        for w in legal_walls:
            if (
                w.get("orientation") == action_dict.get("orientation")
                and w.get("row") == action_dict.get("row")
                and w.get("col") == action_dict.get("col")
            ):
                return action_dict

    if legal_moves:
        best_xy = min(legal_moves, key=lambda xy: d_tbl_stm[int(xy[0]) * 9 + int(xy[1])])
        return {"type": "move", "to": [int(best_xy[0]), int(best_xy[1])]}
    if legal_walls:
        w0 = legal_walls[0]
        return {
            "type": "wall",
            "orientation": w0["orientation"],
            "row": int(w0["row"]),
            "col": int(w0["col"]),
        }
    return {"type": "pass"}


# ---------------------------------------------------------------------------
# Module-Level Persistent Engine & Match State + Public Entry Point
# ---------------------------------------------------------------------------

_ORACLE = _Oracle(cap=32)
_ENGINE = _SearchEngine(_ORACLE)
_VISITED_STATES: Dict[int, int] = {}
_RECENT_CELLS: deque = deque(maxlen=6)
_LAST_TURN = -1
_LAST_PID = -1
_LAST_STATS: Dict[str, Any] = {}


def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Select an action for the current Wrongway / Quoridor observation."""
    global _LAST_TURN, _LAST_PID, _LAST_STATS

    t0 = time.perf_counter()

    if observation.get("done"):
        return {"type": "pass"}

    legal_moves = observation.get("legal_moves") or []
    legal_walls = observation.get("legal_walls") or []
    if not legal_moves and not legal_walls:
        return {"type": "pass"}

    turn = int(observation.get("turn", 0))
    pid = int(observation.get("player_id", observation.get("current_player", 0)))
    you_w = int(observation.get("you", {}).get("walls_remaining", 10))
    opp_w = int(observation.get("opponent", {}).get("walls_remaining", 10))

    # S0: Detect new match / respawn and clear cross-match state
    if turn < _LAST_TURN or pid != _LAST_PID or (turn <= 1 and you_w == MAX_WALLS and opp_w == MAX_WALLS):
        _VISITED_STATES.clear()
        _RECENT_CELLS.clear()
        _ENGINE.reset_match()

    _LAST_TURN = turn
    _LAST_PID = pid

    b = _Board.from_observation(observation)
    _VISITED_STATES[b.zob] = _VISITED_STATES.get(b.zob, 0) + 1
    _RECENT_CELLS.append(b.pawn[b.turn])

    d0_tbl, d1_tbl = b.compute_dist_tables(_ENGINE.dist_cache)
    d_tbl_stm = d0_tbl if b.turn == 0 else d1_tbl

    # Pre-search Layer 0 Safeguards (S1 - S4)
    forced, tag = _pre_search_safeguard(b, _ORACLE, _ENGINE.dist_cache, d0_tbl, d1_tbl)
    if forced is not None:
        ans = _sanitize_action(_action_to_dict(forced), observation, b, d_tbl_stm)
        _LAST_STATS = {
            "mode": tag,
            "depth": 0,
            "nodes": 0,
            "score": 0,
            "ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        return ans

    # Deep Iterative-Deepening PVS Search
    best_m, best_sc, comp_d, total_nodes = _ENGINE.search(b, t0)

    # Post-search Layer 0 Safeguards (S5 - S6)
    final_m, post_tag = _post_search_safeguard(
        b,
        best_m,
        best_sc,
        _ENGINE.root_scores,
        _ENGINE.dist_cache,
        d0_tbl,
        d1_tbl,
        _VISITED_STATES,
        _RECENT_CELLS,
    )

    ans = _sanitize_action(_action_to_dict(final_m), observation, b, d_tbl_stm)
    _LAST_STATS = {
        "mode": post_tag,
        "depth": comp_d,
        "nodes": total_nodes,
        "score": best_sc,
        "ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    return ans
