# Safeguard Component (Layer 0) — Implementation Plan

## 1. Objective & Scope

This document defines the architecture and implementation specification for **Layer 0: The Safeguard Component** ([submissions/safeguard_bot.py](submissions/safeguard_bot.py)).

Layer 0 is the highest-priority anti-loss layer in a Wrongway / Quoridor agent. It sits above any tactical search or neural evaluation module (and can also run standalone) to eliminate technical forfeitures, missed immediate wins, missed 1–2 ply pawn-jump losses, un-wallable 2-ply fork threats, self-harming or decorative walls, wasted tempo in winning races, and infinite oscillation loops.

This component is designed to be:
- **Self-contained**: Python 3.9+ standard library only, single-file Arena compliant (`choose_action(observation)`).
- **Composable**: Exposes `SafeguardLayer.select_or_filter(observation, candidate_action=None)` so it can run standalone (`candidate_action=None`) or act as an authoritative safety gate around any downstream search bot's proposed action.
- **Deterministic & Fast**: Bitwise-periodic deadline guards (`0.45s` soft / `0.70s` hard under the `1.0s` Arena cap) and zero heavy import overhead.

---

## 2. Priority Pipeline (Stages S0–S6)

Every turn flows top-down through seven ordered stages:

| Stage | Name | Primary Sources | Loss Modes Prevented |
| :--- | :--- | :--- | :--- |
| **S0** | Runtime, Budget & State Hygiene Guard | `quoridors_ace_brfast.py`, `quoridors_expert.py`, `SAFEGUARD_RULEBOOK.md` (`SR-01`, `SR-03`, `SR-11`) | Timeouts, worker-kill memory corruption, invalid actions, player-1 orientation bugs |
| **S1** | Immediate 1-Ply Win Short-Circuit | All competitive bots (`SR-05`) | Missing 1-move wins (including jumps onto `goal_row`) |
| **S2** | Pawn-Aware Immediate Loss & Jump-Threat Defense | New pawn-aware guard + `SR-04` exact jump parity | Missing opponent 1-ply jump wins; walking/staying to form a jump-bridge |
| **S3** | 2-Ply Anti-Fork Horizon Guard | New un-wallable threat detector | Letting opponent reach an un-wallable `dist == 1` cell at `opp_dist == 2` |
| **S4** | Zero-Wall & Winning-Race Sprint Guard | `quoridors_ace_brfast.py`, `quoridors_expert.py`, `wrongway_bot.py` (`SR-06`) | Self-blocking or wasting walls when `my_walls == 0`, in winning 0-wall races, or `dist <= 2` |
| **S5** | Shortest-Path DAG & Self-Harm Wall Safeguard | `quoridors_ace_brfast.py`, `quoridors_expert.py`, `wrongway_bot.py` (`SR-07`, `SR-09`) | Decorative walls, net-negative walls, burning final walls, 1-wall `>= 4`-step self-traps |
| **S6** | State-History Anti-Shuffle & Safe Move Selection | `quoridors_expert.py` (`_anti_shuffle`), `SR-10` | Symmetric shortest-path dancing, backtracking loops, narrow-corridor tie-breaks |

---

## 3. Detailed Stage Specifications

### Stage S0 — Runtime, Budget & State Hygiene Guard
1. **Cross-Match & Respawn Reset**:
   - Clear `_visited_states` and `_recent_positions` whenever `turn <= 1`, total remaining walls `my_walls + opp_walls > _last_wall_sum`, or elapsed wall-clock time since the previous call exceeds `10.0s`.
2. **Pre-Computed Emergency Fallback**:
   - Immediately compute a valid fallback action from `observation["legal_moves"]` and `observation["distances"]` (filtering `distances >= 0`, tie-breaking toward center column `4`) in `< 0.1 ms`.
   - If `legal_moves` is empty, fall back to `legal_walls[0]`, or `{"type": "pass"}` if no legal action exists.
3. **Bitwise Periodic Clock Guard**:
   - Inside all wall-scanning and BFS loops, invoke `deadline.check()` (`(ticks & 7) == 0` checking `time.perf_counter() >= hard_deadline`), preventing the unguarded loop timeouts observed in `wrongway_bot` and `counter_hard` in `logs/tournament/summary.json`.
4. **Authoritative Output Gate**:
   - Before returning *any* action from `select_or_filter` or `choose_action`, verify `"move"` destinations against `observation["legal_moves"]` and `"wall"` tuples against `observation["legal_walls"]`. If verification fails, return the pre-computed emergency fallback.

### Stage S1 — Immediate 1-Ply Win Short-Circuit
- Check `observation["legal_moves"]` for any move `m` with `m[0] == you["goal_row"]`.
- Because Arena's `legal_moves` includes straight and diagonal jumps over an adjacent opponent, this immediately captures 1-move jump wins as well as normal goal steps.

### Stage S2 — Pawn-Aware Immediate Loss & Jump-Threat Defense
Standard shortest-path grids (`observation["distances"]`) ignore pawn positions. Stage S2 computes exact jump-aware next-turn moves `_get_legal_moves(opp_pos, test_my_pos, h_walls, v_walls)` matching `environment.py`:
1. **Detect Current 1-Ply Opponent Threat (`opp_wins_now`)**:
   - Check whether the opponent currently has any legal move reaching `opp_goal` (either `opp_dist == 1` or a jump over `my_pos` onto `opp_goal`).
2. **Classify Safe vs. Blunder Pawn Moves**:
   - Partition `legal_moves` into `safe_pawn_moves` (moves `m` where opponent has *no* 1-ply win on their next turn) and `unsafe_pawn_moves` (moves that leave/create a 1-ply win or jump-bridge for the opponent).
3. **Resolve Active 1-Ply Threat**:
   - If `opp_wins_now` is non-empty and `opp_dist >= 2` (meaning the threat exists only because `my_pos` acts as a jump-bridge) and `safe_pawn_moves` exist, immediately prefer the best safe pawn move—defusing the threat at zero wall cost.
   - Otherwise, if `my_walls > 0`, search `legal_walls` around `opp_pos` and `my_pos` (the only anchors capable of altering `opp_pos`'s 1-ply edges) for walls that eliminate *all* 1-ply opponent wins, picking the one maximizing `(new_opp_dist - new_my_dist, new_opp_dist - opp_dist)`.
   - If `candidate_action` was provided and is a wall or safe pawn move that *also* eliminates all 1-ply opponent wins, keep it; otherwise override with the best emergency block/safe move.
4. **Blunder Filter**:
   - If `safe_pawn_moves` is non-empty, forbid selecting any move in `unsafe_pawn_moves` (blocking jump-bridge blunders).

### Stage S3 — 2-Ply Anti-Fork Horizon Guard
When `opp_dist == 2`, `my_dist > 1`, and `my_walls > 0`:
1. Inspect every move `u` in `_get_legal_moves(opp_pos, my_pos, h_walls, v_walls)` where `opp_dist_map[u[0]][u[1]] == 1`.
2. Test whether `u` would be **un-wallable** on the following turn (`_is_unwallable_at_dist_1(u, ...)`): i.e., no legal wall exists around `u` (given existing walls and path-blocking rules) that removes all 1-ply goal steps from `u`.
3. If any reachable `u` is un-wallable once landed on, search `legal_walls` cutting `opp_pos`'s edges (`{opp_r-1, opp_r} x {opp_c-1, opp_c}`) for a legal wall that removes all un-wallable `opp_step1` threats while maximizing `(new_opp_dist - new_my_dist, new_opp_dist - opp_dist)`.
4. If `candidate_action` was supplied and does not eliminate the un-wallable threat (or if running standalone), return the pre-emptive block wall.

### Stage S4 — Zero-Wall & Winning-Race Sprint Guard
When no S1–S3 emergency overrides apply:
- **Hands-Empty Sprint (`my_walls == 0`)**: Skip all wall logic and advance along the best move in `safe_pawn_moves`.
- **Winning Pure Race Lock (`opp_walls == 0` and `my_dist <= opp_dist`)**: Opponent cannot delay us with walls and we are ahead/tied on our turn—never place a wall; advance via the shortest-path safe move.
- **Close Finish Sprint (`my_dist <= 2` and `my_dist <= opp_dist`)**: When within 2 steps of goal and not behind, sprint directly to the goal without wasting tempo on walls.

### Stage S5 — Shortest-Path DAG & Self-Harm Wall Safeguard
Any non-emergency wall (either a `candidate_action` wall being vetted or a standalone defensive wall considered when `opp_dist < my_dist` or `opp_dist <= 3`) must pass all five checks:
1. **Opponent DAG Cut**: The wall must block at least one edge of the opponent's shortest-path DAG (`_dag_cut_wall_set(opp_pos, opp_dist_map, h_walls, v_walls)`). When `opp_walls == 0`, walls are strictly restricted to this DAG cut set.
2. **Positive Disruption**: `op_delta = new_opp_dist - opp_dist >= 1`.
3. **Strict Net Gain (No Self-Harm)**: `gain = op_delta - (new_my_dist - my_dist) >= 1`.
4. **Wall Conservation**:
   - When `my_walls <= 2` and `opp_dist > 3`, require `gain >= 2`.
   - When `my_walls == 1` and `new_my_dist > new_opp_dist` and `opp_dist > 2`, reject (avoid burning the final wall while still behind far from goal).
5. **1-Wall Self-Trap Guard**: When `opp_walls > 0`, compute `_max_opp_1wall_delay` across walls cutting our own shortest-path DAG; reject walls that introduce a single-wall vulnerability delaying us by `>= 4` steps (when our current vulnerability was `< 4`).

When running standalone (`candidate_action is None`) and behind (`opp_dist < my_dist`), Stage S5 selects the highest-scoring wall passing checks 1–5 (preferring walls that flip or tie the race `new_opp_dist >= new_my_dist`, or deliver `gain >= 2`).

### Stage S6 — State-History Anti-Shuffle & Safe Move Selection
- Tracks canonical visited board states `(my_pos, opp_pos, sorted_h_walls, sorted_v_walls)` in `_visited_states` and recent pawn squares in `_recent_positions`.
- Ranks `safe_pawn_moves` by:
  1. Shortest-path distance (`d_m`)
  2. State repeat / backtrack avoidance (if the top move repeats a visited state or recent cell, substitutes the best non-repeating move with `d_alt <= my_dist`, or `d_alt <= best_d + 1` if needed to break a loop)
  3. 1-wall self-trap awareness when adjacent/near corridors (`opp_walls > 0`)
  4. Centrality tie-breaking (`abs(col - 4)`)

---

## 4. Integration Interface

```python
from submissions.safeguard_bot import SafeguardLayer, choose_action

# Standalone Layer 0 agent:
action = choose_action(observation)

# Wrapping a downstream search layer:
safeguard = SafeguardLayer()
safe_action = safeguard.select_or_filter(observation, candidate_action=search_action)
```
