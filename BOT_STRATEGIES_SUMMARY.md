# Strategy Summary: Quoridors & Wrongway Bots

This document summarizes the search, pruning, candidate generation, evaluation, and endgame strategies implemented by the three ported bots in `submissions/`:

- [`wrongway_bot.py`](submissions/wrongway_bot.py) — ported from `wrongway.app` (`bot.js` & `ai.js`, duel9 hard mode)
- [`quoridors_expert.py`](submissions/quoridors_expert.py) — ported from `quoridors.com` (`QSeriesBot` Expert / "Strategist")
- [`quoridors_ace_brfast.py`](submissions/quoridors_ace_brfast.py) — ported from `quoridors.com` (`ACE1 BR Fast` / `ace:brfast`)

---

## High-Level Comparison

| Bot | Origin / Style | Search Algorithm | Move / Wall Pruning & Ordering | Evaluation | Endgame Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `wrongway_bot.py` | `wrongway.app` (duel9 hard mode) | Iterative-deepening Negamax + Alpha-Beta (depths 1–3, `0.85s` cap) | Downhill BFS along opponent's shortest path (top 6–8 walls) | 6-feature linear heuristic + root bonuses / anti-oscillation | Immediate sprint when `my_walls == 0` or (`my_dist <= 2` and `my_dist <= opp_dist`) |
| `quoridors_expert.py` | `quoridors.com` (`QSeriesBot` Expert) | Iterative-deepening Alpha-Beta Minimax (up to depth 10, `0.72s` / 1.2M nodes) | Opponent path + 3×3 focus + wall-fight templates (capped at 24), LMR, 2-slot killers, history, TT | Phase-weighted path lead + wall lead | Zero-wall race verdict (`\|gap\| >= 2`), DAG-cut walls when opponent has 0 walls, anti-shuffle |
| `quoridors_ace_brfast.py` | `quoridors.com` (`ACE1 BR Fast`) | Iterative-deepening PVS Negamax (up to depth 64, predictive `780ms` soft/hard time control) | Shortest-path DAG path-delta ordering, butterfly + countermove history, LMR, LMP, RFP, TT | Distilled `1215→192→1` neural value MLP (centipawn logit) | Exact 13,122-state retrograde 0-wall oracle + exact proof-tree solver at `≤ 2` total walls |

---

## 1. `wrongway_bot.py`

Ported from `wrongway.app` (`bot.js` & `ai.js`), using duel9 hard-mode "Waller" weights (`submissions/wrongway_bot.py`).

### Search & Time Management
- **Iterative-Deepening Negamax with Alpha-Beta Pruning** (`_negamax`, `choose_action`, lines 216–440):
  - Searches `target_depth` from `1` to `3` under a `0.85s` deadline (`TimeoutError` guard).
  - Re-sorts root candidates by score after each completed depth (`all_candidates.sort(key=lambda c: c["score"], reverse=True)`).
  - Stops early if a forced win (`>= BOT_WIN - 50`, where `BOT_WIN = 100_000`) is found.
  - Inside `_negamax`, pawn moves are ordered closest-to-goal first (`dist_me_map[m[0]][m[1]]`).

### Candidate Wall Generation
- **Downhill Shortest-Path Wall Generator** (`_find_candidate_walls`, lines 129–213):
  - Runs a downhill BFS from the opponent's position along shortest-path edges up to `lookahead` steps (`6` at root, `5` inside `_negamax`).
  - Collects horizontal and vertical walls adjacent to those shortest-path transitions.
  - Filters out conflicting walls, walls that trap either player (`>= 999`), and walls that fail to increase the opponent's shortest path by at least `1` (`n_opp_d - cur_opp_d < 1`).
  - Ranks walls by net path gain `(n_opp_d - cur_opp_d) - (n_my_d - cur_my_d)` and keeps the top `max_k` (`8` at root, `6` in search, `4` when `depth <= 1`).

### Evaluation & Root Biases
- **6-Feature Linear Heuristic** (`_evaluate`, lines 26–30, 101–110):
  - Uses `WEIGHTS = [42.8, 100.0, 141.2, 30.4, 646.2, -21.9]`:
    1. Tempo constant: `+42.8`
    2. Path lead: `100.0 * (d_op - d_me)`
    3. Wall lead: `141.2 * (w_me - w_op)`
    4. Effective walls: `30.4 * (min(w_me, d_op) - min(w_op, d_me))`
    5. Zero-wall race endgame: `+646.2` if `w_me == 0 and w_op == 0 and d_me <= d_op`, `-646.2` if `w_me == 0 and w_op == 0 and d_me > d_op`, else `0`
    6. Wall exhaustion indicator: `-21.9 * ((1 if w_op == 0 else 0) - (1 if w_me == 0 else 0))`
- **Root-Level Tactical Bonuses & Anti-Oscillation** (lines 355–387):
  - `FORWARD_DRIVE = 40.0` added to root moves that strictly reduce `my_dist`.
  - `WALL_BIAS = 20.0` added to root candidate walls.
  - Anti-oscillation (`_recent_positions`, `BACKTRACK_PENALTY = 60.0`): penalizes revisiting any of the last 3 prior positions by `-60.0 * (4 - idx_from_end) / 4.0`.

### Endgame & Immediate Shortcuts
- Immediately returns any 1-ply goal-reaching move (lines 321–325).
- **Endgame Sprint** (lines 343–352): if `my_walls == 0` or (`my_dist <= 2 and my_dist <= opp_dist`), skips search and picks the legal move with minimum shortest-path distance.

---

## 2. `quoridors_expert.py`

Ported from `quoridors.com` (`QSeriesBot` Expert / "Strategist", `submissions/quoridors_expert.py`).

### Board Representation & Prechecks
- **Blocked-Edge 9×9 Board (`_Board`)** (lines 232–558):
  - Tracks per-cell blocked direction bitmasks (`blocked`), 64-bit Zobrist hashes (`hash_lo`, `hash_hi`), and wall bitmasks (`hw_mask`, `vw_mask`).
  - Uses O(1) wall-overlap bitmasks (`H_OVERLAP_MASK`, `V_OVERLAP_MASK`) and a conservative 5×5 / edge-anchor precheck (`wall_needs_path_check`, lines 340–347) so BFS path checks (`has_path`) only run when a wall is on the board edge or near 2+ existing walls.
  - Caches multi-source BFS distance fields from both goal rows keyed by wall bitmask (`dist_cache`, up to `DIST_CACHE_MAX = 16_384`).

### Search Engine
- **Iterative-Deepening Alpha-Beta Minimax** (`_minimax`, `choose_action`, lines 942–1028, 1126–1135):
  - Searches up to `MAX_DEPTH = 10` under `TIME_BUDGET_SEC = 0.72s` and `NODE_BUDGET = 1_200_000`.
  - **Transposition Table**: up to `TT_MAX = 300_000` entries storing `hash_hi`, `depth`, score, bound flag (`TT_EXACT`, `TT_LOWER`, `TT_UPPER`), and `tt_move`.
  - **Move Ordering** (`_order_moves`, `_record_cutoff`, lines 895–939): orders `tt_move` → 2-slot killer moves per ply (`killers[ply]`) → history heuristic (`history[m] += depth * depth` on beta cutoffs) → pawn moves before walls.
  - **Late Move Reductions (LMR)** (lines 997–1005): when `depth >= LMR_MIN_DEPTH` (`4`) and move index `i >= LMR_AFTER` (`4`), reduces depth by `1` and re-searches at full `depth - 1` if the reduced search improves `alpha` (maximizing) or `beta` (minimizing).
  - **Frontier Wall Cutoff**: walls are only generated when `depth >= WALL_FLOOR` (`2`).

### Candidate Wall Generation
- **`_candidates`** (lines 652–863):
  - **When opponent has 0 walls (`b.wl[opp] == 0`)**: calls `_dag_cut_walls` to build the opponent's full shortest-path DAG and emits all legal walls that cut any DAG edge.
  - **Normal play**:
    1. **Focus + Path Slots**: combines the 3×3 neighborhood wall slots around both pawns (`CELL_FOCUS_SLOTS_MASK`) with slots along one BFS shortest path for the opponent (`_shortest_path_slots_mask`), capped at `WALL_CAP = 24` slots sorted by Manhattan distance to the opponent.
    2. **Wall-Fight Templates**: adds template neighbor slots (`HW_TPL_H`, `HW_TPL_V`, `VW_TPL_H`, `VW_TPL_V`) around all placed walls, plus edge-column horizontal slots (`EDGE_H_COLS_MASK`) when `hist_len >= 6`, capped at `24` closest to either pawn.

### Evaluation, Endgame & Anti-Shuffle
- **Phase-Weighted Heuristic** (`_heuristic`, lines 562–569):
  - `(opp_d - ai_d) + w * (wl[ai] - wl[1 - ai])`
  - Wall value `w = 0.05 + 0.15 * min(ai_d, opp_d, 8) / 8.0` (ranges from `0.05` at distance 0 to `0.20` at distance ≥ 8).
- **Exact Zero-Wall Race Verdict** (`_race_verdict`, lines 572–580):
  - At leaf nodes when `wl[0] == 0 and wl[1] == 0` and `|ai_d - opp_d| >= 2`, returns `900 - ai_d` if ahead or `-(900 - opp_d)` if behind.
- **Anti-Shuffle Guard** (`_anti_shuffle`, lines 1052–1080):
  - Maintains `_VISITED_POSITIONS` (Zobrist state keys across turns in a match). If the chosen pawn move leads to an already-visited state, replaces it with the shortest-path non-repeating pawn move if one exists.

---

## 3. `quoridors_ace_brfast.py`

Ported from `quoridors.com` (`ACE1 BR Fast` / `ace:brfast`, `submissions/quoridors_ace_brfast.py`).

### Bitboard Rules Core
- **81-Bit Directional Bitboards (`_Board`)** (lines 185–633):
  - Maintains `u_ok`, `d_ok`, `l_ok`, `r_ok` bitboards for bit-parallel flood-fill path verification (`_has_path_bb`).
  - Uses `witness_cut_masks` (computing wall slots cutting a single witness shortest path for each player) combined with the 5×5 / edge-anchor precheck (`wall_needs_path_check`) to generate all legal moves rapidly.

### Distilled Neural Leaf Evaluation
- **`1215 → 192 → 1` Quantized MLP (`_unpack_net`, `_net_cp`)** (lines 638–770, 1106–1293):
  - Embeds compressed int8 weights from `ace1-fast.bin` (ReLU hidden layer, `tanh` output).
  - Converts `tanh` output `vs` to centipawns via `300.0 * ln((1 + vs) / (1 - vs))` (clamped to `±9999`) and caches scores in `eval_cache` (`120_000` max).
  - Encodes 15 9×9 spatial/scalar channels: pawn positions, remaining walls, side to move, open N/E/S/W edges, clamped shortest-path distances (`<= 81`), placed H/V walls, and **illegal/sealed H/V wall slots** (`wcache` + `seal_cache`).
  - Evaluates incrementally using differential 24-bit packed BigInt SIMD additions.

### Exact Endgame Oracle & Near-Endgame Solver
- **Hands-Empty Retrograde Oracle (`_Oracle`)** (lines 807–890):
  - When `wl[0] == 0 and wl[1] == 0`, solves all `81 * 81 * 2 = 13,122` `(p0, p1, turn)` states via retrograde wavefront BFS (LRU-cached for 64 wall layouts), yielding exact `WIN`/`LOSS`/`DRAW` and plies-to-mate (`±(MATE - ply - kv)`).
- **Near-Endgame Proof-Tree Solver (`_solve_near_endgame`)** (lines 903–1024, 1310–1331):
  - Activates whenever total remaining walls `wl[0] + wl[1] <= 2` (`SOLVE_KMAX = 2`).
  - Runs an iterative-deepening full-width game-tree search (`SOLVE_NODE_BUDGET = 200_000`, `SOLVE_TIME_SEC = 0.025s`, `SOLVE_MAX_DEPTH = 160`) backed by `_Oracle`, memoizing proven positions in `proven_memo` with exact mate scores (`±(MATE - ply - mk)`).

### Barrier-Race Search Engine (`_KaBrEngine`)
- **Iterative-Deepening PVS Negamax** (lines 1540–1771):
  - Searches up to `MAX_ID_DEPTH = 64` using Principal Variation Search (zero-window scout searches for non-PV moves, re-searching with full window on fail-high).
  - **Predictive Time Control**: `TIME_BUDGET_MS = 780.0`, `SOFT_PCT = 0.55` (~429 ms), `HARD_PCT = 0.84` (~655 ms). Stops before starting a new depth once past `soft_t` or if `_predicted_iteration_ms(prev_iter_ms, last_iter_ms)` exceeds remaining hard time; also stops immediately upon proving a mate (`abs(best_s) >= MATE_BAND = 29000`).
- **DAG-Based Move Ordering** (`_base_order`, `_order_moves`, lines 1358–1530):
  - Computes all shortest-path DAG edges (`_dag_edges`) for both players.
  - Marks walls touching either player's DAG as "tight" (`tight_h`, `tight_v`) and computes exact `op_delta - my_delta` path changes for them; non-tight walls are grouped directly into `zero`.
  - Orders moves as:
    1. `tt_move`
    2. Immediate goal-winning pawn moves
    3. History-scored moves using butterfly `hist[side][action]` + **countermove history** (`cmh[(side, prev_action, action)]`, gravity-updated up to `CMH_MAX = 32767`)
    4. Remaining pawn moves (shortest distance first) → `pos` walls (`op_delta - my_delta > 0`) → `zero` walls → `neg` walls
  - Ages butterfly history by halving between depths (`_age_history`).
- **Pruning & Reductions** (lines 1559–1647):
  - **Depth-Preferred Transposition Table** (`250_000` entries) with mate-ply adjustment (`_to_tt` / `_from_tt`).
  - **Reverse Futility Pruning (RFP)**: on null-window nodes at `1 <= depth <= 3` (non-mate), prunes if `static_cp - 100 * depth >= beta`.
  - **Late Move Pruning (LMP)**: on null-window nodes at `1 <= depth <= 3`, skips non-TT, non-tight walls past index `LMP_THRESH = (6, 12, 24)`.
  - **Late Move Reductions (LMR)**: at `depth >= 3` when `len(mvs) > 6`, reduces non-TT, non-tight walls at `idx >= 6` by `1` ply, re-searching at full depth on fail-high.
