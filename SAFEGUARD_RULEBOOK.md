# Wrongway Agent Safety Guardrail Rulebook (Avoiding Losses)

This rulebook documents the **agent-side safety guardrails** implemented across the bots in `submissions/` to prevent losing matches—whether by technical turn forfeitures, tactical blunders, wasted walls/tempo, search timeouts, or infinite oscillation loops. It also summarizes the Arena constraints that enforce those losses and the remaining blind spots in the current bots.

---

## 1. Anti-Forfeiture Guardrails (Never Lose Free Turns)

Under Wrongway Arena rules ([environment.py](environment.py), [arena.py](arena.py), [agent_runner.py](agent_runner.py)), any timeout, unhandled exception, or illegal move/wall forfeits your turn without changing the board—handing the opponent a free tempo advantage and respawning your worker on timeouts/crashes.

| Rule ID | Guardrail | Implementation in Agents | Loss Mode Prevented |
| --- | --- | --- | --- |
| **SR-01** | **Sub-1s Search Deadline & Layered Fallback** | • [wrongway_bot.py](submissions/wrongway_bot.py), `waller_*`, `runner_*`, `counter_*`: set `deadline = start_time + 0.85`s (`0.15s` buffer under Arena's `1.0s` cap) and check `time.perf_counter() > deadline` inside iterative deepening and `_negamax`.<br>• [minimax_agent.py](submissions/minimax_agent.py): sets `TIME_LIMIT = 0.85`s, checks `time.monotonic() >= deadline`, and raises `SearchTimeout`.<br>• **Double fallback:** both keep the best move from the last completed depth; if depth 1 is interrupted, `wrongway_bot.py` falls back to the shortest-distance move in `my_dist_map` and `minimax_agent.py` pre-initializes `best_action` to the shortest-distance move in `observation["distances"]`. | Losing turns (and losing cached globals on worker kill) due to `choose_action` exceeding `1.0s`. |
| **SR-02** | **Client-Side Legality & Path-Block Pre-Filter** | • [wrongway_bot.py](submissions/wrongway_bot.py) (`_find_candidate_walls`): checks `0 <= wr < 8` and `0 <= wc < 8`, rejects overlaps/crossings via `_wall_conflicts`, and rejects any wall where re-running BFS yields `n_opp_d >= 999` or `n_my_d >= 999`.<br>• [minimax_agent.py](submissions/minimax_agent.py): checks `_wall_does_not_overlap` and rejects walls where `bfs(...) == float("inf")` for either player.<br>• [wall_builder.py](submissions/wall_builder.py) & `*_easy.py`: select only from Arena-provided `observation["legal_walls"]`.<br>• All bots return `{"type": "pass"}` if no legal action exists (`greedy.py` also filters unreachable moves `distances >= 0` first). | Turn forfeitures from `"wall_blocks_path"`, `"wall_overlap"`, `"wall_crossing"`, or out-of-bounds coordinates. |
| **SR-03** | **Dynamic Perspective & Goal Normalization** | Always read `you["goal_row"]` and `opponent["goal_row"]` instead of assuming row `0` (`my_goal`). `wrongway_bot.py` and `minimax_agent.py` normalize the root agent to slot `0`/maximizer regardless of whether `player_id` is `0` or `1`. | Walking the wrong direction or defending the wrong end when swapped to Player `1`. |
| **SR-04** | **Exact Jump Parity in Search** | `_get_legal_moves` ([wrongway_bot.py](submissions/wrongway_bot.py)) and `all_legal_pawn_moves` ([minimax_agent.py](submissions/minimax_agent.py)) replicate `environment.py` jump rules: straight jump over an adjacent opponent if the rear edge is open, diagonal sidesteps only if the rear edge is blocked. | Mis-evaluating threats or planning illegal pawn steps inside deep search. |

---

## 2. Tactical & Tempo Guardrails (Never Throw Away Winning Races)

| Rule ID | Guardrail | Implementation in Agents | Loss Mode Prevented |
| --- | --- | --- | --- |
| **SR-05** | **Immediate-Win Short-Circuit** | Before any BFS wall scan or tree search, `wrongway_bot.py`, `*_hard.py`, `*_normal.py`, `*_easy.py`, `wall_builder.py`, and `minimax_agent.py` scan `legal_moves` for `m[0] == my_goal` (or `you["goal_row"]`) and immediately return that move. | Missing a 1-move win due to placing a pointless wall, triggering search timeout, or hitting `BLUNDER_RATE`. |
| **SR-06** | **Endgame Sprint Shortcut** | `wrongway_bot.py`, `*_hard.py`, and `*_normal.py` immediately skip tree search and play the shortest-BFS move whenever:<br>`my_walls == 0 or (my_dist <= 2 and my_dist <= opp_dist)` | Wasting turns/time searching when you have no walls left or are within 2 steps and not behind in the race (also shields `_normal` bots from blundering at the finish line). |
| **SR-07** | **Strict Positive-Disruption Wall Filter** | `_find_candidate_walls` in `wrongway_bot.py`, `*_hard.py`, and `*_normal.py` discards every wall unless it strictly delays the opponent by at least 1 step (`n_opp_d - cur_opp_d >= 1`), sorts by net path differential `(n_opp_d - cur_opp_d) - (n_my_d - cur_my_d)`, and only searches top `max_k`. [wall_builder.py](submissions/wall_builder.py) requires `gain > 0` and only searches walls when `enemy_distance <= own_distance + 1`. | Losing tempo and finite walls on decorative or self-hurting walls. |
| **SR-08** | **Shortest-Mate / Longest-Defense Terminal Scoring** | • `wrongway_bot.py` & `*_hard.py`: `_negamax` scores in-tree wins as `BOT_WIN - ply` (`100_000 - ply`) and stops iterative deepening early when `current_best_score >= BOT_WIN - 50`.<br>• `minimax_agent.py`: `terminal_score` returns `10000.0 + depth * 10.0` (agent win) and `-10000.0 - depth * 10.0` (opponent win). | Delaying a forced win (giving opponent time to wall) or failing to resist a losing line. |
| **SR-09** | **Wall Conservation & Zero-Wall Race Guard** | • `wrongway_bot.py` / `*_hard.py` `_evaluate`: heavily penalizes losing a zero-wall race via `WEIGHTS[4] = 646.2` (`+646.2` when `w_me == 0 and w_op == 0 and d_me <= d_op`, `-646.2` when `d_me > d_op`), caps wall utility against remaining distance (`WEIGHTS[3] * (min(w_me, d_op) - min(w_op, d_me))`), and weights wall lead via `WEIGHTS[2]` (`141.2` waller / `156.9` counter / `196.1` runner).<br>• `minimax_agent.py`: adds `W_WALL * (my_walls - opp_walls)` (`W_WALL = 2.0`, while `W_DIST = 10.0`), requiring a wall to yield more than `0.2` path steps of net value to spend it. | Burning all 10 walls early and getting out-walled or trapped defenseless in the endgame. |

---

## 3. Anti-Loop & State-Hygiene Guardrails

| Rule ID | Guardrail | Implementation in Agents | Loss Mode Prevented |
| --- | --- | --- | --- |
| **SR-10** | **Anti-Oscillation / Backtrack Penalty** | • `wrongway_bot.py`, `*_hard.py`, `*_normal.py`: store up to 6 recent cells in `_recent_positions` and subtract `BACKTRACK_PENALTY * (4 - idx_from_end) / 4.0` (`BACKTRACK_PENALTY = 60.0`) from root moves revisiting any of the last 4 squares.<br>• `minimax_agent.py`: stores up to 10 positions in `RECENT_POSITIONS` and subtracts `REPEAT_PENALTY * count` (`REPEAT_PENALTY = 8.0`) at the MAX root (`is_root=True`). | Back-and-forth dancing between symmetric shortest-path squares that surrenders tempo or drags winning games into `500`-turn draws. |
| **SR-11** | **Cross-Match Reset & Stateless Rebuild** | `wrongway_bot.py` & `*_hard.py` reset `_recent_positions` when `turn <= 1` or `now - _last_match_reset > 10.0`s; `minimax_agent.py` clears `RECENT_POSITIONS` when `turn <= 1`. Both reconstruct wall sets and board state cleanly from `observation` on every turn. | Stale history from a prior match causing false backtrack penalties, or crashing after an Arena worker respawn clears module globals. |
| **SR-12** | **Blunder Suppression** | Competitive tiers (`wrongway_bot.py`, `*_hard.py`, `minimax_agent.py`) set `BLUNDER_RATE = 0.0` (no random candidate downgrade), whereas `*_normal.py` bots inject `BLUNDER_RATE = 0.25` from the top `3` candidates (`BLUNDER_POOL = 3`). | Throwing decisive advantages via random suboptimal moves. |

---

## 4. Guardrail Coverage by Bot

| Bot | SR-01 (Time) | SR-02 (Wall Check) | SR-05 (1-Move Win) | SR-06 (Sprint) | SR-07 (Net-Gain Wall) | SR-10 (Anti-Loop) | SR-12 (0% Blunder) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [wrongway_bot.py](submissions/wrongway_bot.py) | `0.85s` + fallback | Full BFS check | Yes | Yes | `d_opp >= +1`, top `K` | `60.0` penalty | Yes |
| `waller_hard.py` / `counter_hard.py` / `runner_hard.py` | `0.85s` + fallback | Full BFS check | Yes | Yes | `d_opp >= +1`, top `K` | `60.0` penalty | `BLUNDER_RATE = 0.0` |
| `*_normal.py` | `0.85s` + fallback | Full BFS check | Yes | Yes | `d_opp >= +1`, top `K` | `60.0` penalty | No (`0.25`) |
| `*_easy.py` | Greedy (instant) | `legal_walls` | Yes | Greedy only | No (random nearby wall) | No | Random walls (`0.05`–`0.25`) |
| [minimax_agent.py](submissions/minimax_agent.py) | `0.85s` + fallback | Full BFS check | Yes | No | Top `18` by heuristic (no `d_opp >= +1` filter) | `8.0` penalty | Yes |
| [wall_builder.py](submissions/wall_builder.py) | 0-ply BFS | `legal_walls` | Yes | No | `gain > 0` | No | Yes |
| [greedy.py](submissions/greedy.py) | `O(1)` lookup | Never walls | Shortest `distances` | Always | Never walls | No | Yes |

---

## 5. Known Remaining Vulnerabilities (Where Agents Can Still Lose)

Even with the guardrails above, code inspection shows several gaps that still cause losses:

1. **Downhill-Only Wall Blind Spot (`wrongway_bot.py`, `*_hard.py`):** `_find_candidate_walls` only proposes walls adjacent to the **opponent's** shortest path within `lookahead` (4–6 steps). It will not generate defensive walls off that corridor to protect its own path or set up deeper multi-wall box traps.
2. **Shallow Horizon & Wall Branch Caps (`MAX_SEARCH_DEPTH = 3`, `MAX_WALL_BRANCHES = 18`):** `wrongway_bot.py` and `*_hard.py` search at most 3 plies deep, and `minimax_agent.py` caps wall branching at `MAX_WALL_BRANCHES = 18` per node within its `0.85s` budget, so multi-wall traps beyond those horizons or outside the top 18 heuristic-ordered walls can be missed.
3. **Pawn-Unaware BFS Distance Maps:** Distance heuristics in all bots (`_compute_distance_map` / `bfs` / `observation["distances"]`) ignore pawn collisions, so blocked corridors where a jump or sidestep changes actual steps to goal are only resolved inside move-level search plies.
4. **Wall-Exhaustion Sign Quirk in `_evaluate` ([wrongway_bot.py](submissions/wrongway_bot.py#L26-L109)):** While `WEIGHTS[4] = 646.2` strongly guards against losing a `0 vs 0` wall race, `WEIGHTS[5] = -21.9` multiplies `((1.0 if w_op == 0 else 0.0) - (1.0 if w_me == 0 else 0.0))`—giving `+21.9` when `w_me == 0` and `-21.9` when `w_op == 0` (partially offsetting `WEIGHTS[2]` + `WEIGHTS[3]`).
5. **Lack of Endgame Sprint Shortcut in [minimax_agent.py](submissions/minimax_agent.py):** Unlike `wrongway_bot.py`, `minimax_agent.py` does not short-circuit deep search when `my_walls == 0` or when close and ahead (`my_dist <= 2 and my_dist <= opp_dist`), continuing iterative deepening up to `0.85s` and applying `REPEAT_PENALTY` at the root.
6. **Unchecked Random Walls in `*_easy.py`:** `waller_easy.py` (`WALL_CHANCE = 0.25`), `counter_easy.py` (`0.15`), and `runner_easy.py` (`0.05`) place random walls within `±2` cells of the opponent without checking if they slow the opponent at all—often wasting a turn and hurting themselves.
7. **Schedule/Hash-Sensitive Ordering:** `wrongway_bot.py` / `*_hard.py` build `h_walls`/`v_walls`/`proposed_walls` as `set`s (whose iteration order varies with Python `PYTHONHASHSEED`), and `minimax_agent.py` cuts off via `time.monotonic() >= deadline` (`0.85s`), causing tie-break and search-depth variations across runs under OS scheduling load.

---

## 6. Underlying Engine Constraints Enforcing Losses ([environment.py](environment.py) & [agent_runner.py](agent_runner.py))

* **Board & Match End:** Reach opponent's starting row first (`9x9`, `10` walls each, `max_turns = 500` draws cleanly except goal on turn 500 wins).
* **Wall Legality:** No overlap, no crossing, both players must keep a BFS path to their goal row (`_wall_error`).
* **Fault Penalties:** Invalid actions (`step`), exceptions (`BaseException`), or timeouts (`startup_timeout = 5.0s`, turn `timeout = 1.0s`, `MAX_MESSAGE_BYTES = 1 MiB`) forfeit the turn (`skip_turn`); timeouts/crashes terminate and respawn the child worker cleanly.
