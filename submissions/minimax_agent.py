"""Iterative Deepening Minimax + Alpha-Beta Pruning Agent for Wrongway Arena."""
from collections import deque
import time

# ==============================================================================
# EXPLICIT TUNABLE PARAMETERS
# ==============================================================================
TIME_LIMIT = 0.85            # Max search time per turn in seconds (Arena limit is 1.0s)
MAX_DEPTH = 10               # Iterative deepening maximum depth
W_DIST = 10.0                # Weight for (opponent_distance - my_distance)
W_WALL = 2.0                 # Weight for (my_walls_remaining - opponent_walls_remaining)
W_PROG = 0.5                 # Weight for (my_progress - opponent_progress)
MAX_WALL_BRANCHES = 18       # Keep top K walls after move ordering so Alpha-Beta searches deeper
REPEAT_PENALTY = 8.0         # Penalty for revisiting recent cells (prevents oscillation)
# ==============================================================================

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_WALLS = tuple((ori, r, c) for ori in ("H", "V") for r in range(8) for c in range(8))
RECENT_POSITIONS = deque(maxlen=10)


class SearchTimeout(Exception):
    """Raised when iterative deepening reaches the turn deadline."""


def _walls_to_blocked(h_walls, v_walls):
    b = set()
    for r, c in h_walls:
        for u, v in (((r, c), (r + 1, c)), ((r, c + 1), (r + 1, c + 1))):
            b.add((u, v))
            b.add((v, u))
    for r, c in v_walls:
        for u, v in (((r, c), (r, c + 1)), ((r + 1, c), (r + 1, c + 1))):
            b.add((u, v))
            b.add((v, u))
    return frozenset(b)


class State:
    """Compact board state for Minimax tree search."""
    __slots__ = ("pos", "goals", "walls_rem", "h_walls", "v_walls", "blocked")

    def __init__(self, pos, goals, walls_rem, h_walls, v_walls, blocked=None):
        self.pos = pos              # ((my_r, my_c), (opp_r, opp_c))
        self.goals = goals          # (my_goal_row, opp_goal_row)
        self.walls_rem = walls_rem  # (my_walls, opp_walls)
        self.h_walls = h_walls      # frozenset of (r, c)
        self.v_walls = v_walls      # frozenset of (r, c)
        self.blocked = blocked if blocked is not None else _walls_to_blocked(h_walls, v_walls)


def bfs(position, goal_row, blocked):
    """BFS shortest path from position to goal_row (ignores pawns)."""
    if position[0] == goal_row:
        return 0
    q = deque([(position, 0)])
    visited = {position}
    while q:
        (r, c), dist = q.popleft()
        nd = dist + 1
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                nxt = (nr, nc)
                if nxt not in visited and ((r, c), nxt) not in blocked:
                    if nr == goal_row:
                        return nd
                    visited.add(nxt)
                    q.append((nxt, nd))
    return float("inf")


def terminal(state):
    return state.pos[0][0] == state.goals[0] or state.pos[1][0] == state.goals[1]


def terminal_score(state, depth):
    # Prefer faster wins and slower losses via depth bonus
    if state.pos[0][0] == state.goals[0]:
        return 10000.0 + depth * 10.0
    if state.pos[1][0] == state.goals[1]:
        return -10000.0 - depth * 10.0
    return 0.0


def heuristic(state):
    """HEURISTIC(state): 10*(opp_dist - my_dist) + 2*wall_adv + 0.5*progress_adv."""
    my_distance = bfs(state.pos[0], state.goals[0], state.blocked)
    opponent_distance = bfs(state.pos[1], state.goals[1], state.blocked)

    wall_advantage = state.walls_rem[0] - state.walls_rem[1]
    my_progress = 8 - abs(state.pos[0][0] - state.goals[0])
    opponent_progress = 8 - abs(state.pos[1][0] - state.goals[1])
    progress_advantage = my_progress - opponent_progress

    return (
        W_DIST * (opponent_distance - my_distance)
        + W_WALL * wall_advantage
        + W_PROG * progress_advantage
    )


def all_legal_pawn_moves(state, player):
    """Generate legal pawn moves (player=0 for MAX, player=1 for MIN), including jumps."""
    r, c = state.pos[player]
    opp = state.pos[1 - player]
    blocked = state.blocked
    dests = set()
    for dr, dc in DIRS:
        adj = (r + dr, c + dc)
        if not (0 <= adj[0] < 9 and 0 <= adj[1] < 9) or ((r, c), adj) in blocked:
            continue
        if adj != opp:
            dests.add(adj)
        else:
            jmp = (adj[0] + dr, adj[1] + dc)
            if 0 <= jmp[0] < 9 and 0 <= jmp[1] < 9 and (adj, jmp) not in blocked:
                dests.add(jmp)
            else:
                for sdr, sdc in ((dc, dr), (-dc, -dr)):
                    side = (adj[0] + sdr, adj[1] + sdc)
                    if 0 <= side[0] < 9 and 0 <= side[1] < 9 and (adj, side) not in blocked:
                        dests.add(side)
    return [{"type": "move", "to": [d[0], d[1]]} for d in sorted(dests)]


def _wall_does_not_overlap(state, ori, r, c):
    if ori == "H":
        if (r, c) in state.v_walls:
            return False
        return not ((r, c - 1) in state.h_walls or (r, c) in state.h_walls or (r, c + 1) in state.h_walls)
    if (r, c) in state.h_walls:
        return False
    return not ((r - 1, c) in state.v_walls or (r, c) in state.v_walls or (r + 1, c) in state.v_walls)


def apply_action(state, action, player):
    """Return new child State after player applies action."""
    if action["type"] == "move":
        to = tuple(action["to"])
        new_pos = (to, state.pos[1]) if player == 0 else (state.pos[0], to)
        return State(new_pos, state.goals, state.walls_rem, state.h_walls, state.v_walls, state.blocked)
    if action["type"] == "wall":
        ori, r, c = action["orientation"], action["row"], action["col"]
        new_w = (state.walls_rem[0] - 1, state.walls_rem[1]) if player == 0 else (state.walls_rem[0], state.walls_rem[1] - 1)
        if ori == "H":
            new_h, new_v = state.h_walls | {(r, c)}, state.v_walls
            edges = (((r, c), (r + 1, c)), ((r, c + 1), (r + 1, c + 1)))
        else:
            new_h, new_v = state.h_walls, state.v_walls | {(r, c)}
            edges = (((r, c), (r, c + 1)), ((r + 1, c), (r + 1, c + 1)))
        added = {(u, v) for a, b in edges for u, v in ((a, b), (b, a))}
        return State(state.pos, state.goals, new_w, new_h, new_v, state.blocked | added)
    return state


def generate_and_order_actions(state, player, deadline):
    """GENERATE_LEGAL_ACTIONS + ORDER_ACTIONS sorted by HEURISTIC(child)."""
    if time.monotonic() >= deadline:
        raise SearchTimeout

    scored_moves = []
    for mv in all_legal_pawn_moves(state, player):
        child = apply_action(state, mv, player)
        scored_moves.append((heuristic(child), mv))

    scored_walls = []
    if state.walls_rem[player] > 0:
        for ori, r, c in ALL_WALLS:
            if time.monotonic() >= deadline:
                raise SearchTimeout
            if not _wall_does_not_overlap(state, ori, r, c):
                continue
            w_act = {"type": "wall", "orientation": ori, "row": r, "col": c}
            child = apply_action(state, w_act, player)
            # Ensure neither player is completely cut off from their goal
            d0 = bfs(child.pos[0], child.goals[0], child.blocked)
            if d0 == float("inf"):
                continue
            d1 = bfs(child.pos[1], child.goals[1], child.blocked)
            if d1 == float("inf"):
                continue
            wall_adv = child.walls_rem[0] - child.walls_rem[1]
            prog_adv = (8 - abs(child.pos[0][0] - child.goals[0])) - (8 - abs(child.pos[1][0] - child.goals[1]))
            h_val = W_DIST * (d1 - d0) + W_WALL * wall_adv + W_PROG * prog_adv
            scored_walls.append((h_val, w_act))

    reverse = (player == 0)  # MAX_PLAYER -> DESCENDING, MIN_PLAYER -> ASCENDING
    scored_moves.sort(key=lambda x: x[0], reverse=reverse)
    scored_walls.sort(key=lambda x: x[0], reverse=reverse)

    # Combine ordered pawn moves + top ordered walls
    ordered = [act for _, act in scored_moves] + [act for _, act in scored_walls[:MAX_WALL_BRANCHES]]
    return ordered or [{"type": "pass"}]


def minimax(state, depth, alpha, beta, player, deadline, is_root=False, recent_counts=None):
    """MINIMAX with Alpha-Beta pruning (player=0 is MAX_PLAYER, player=1 is MIN_PLAYER)."""
    if time.monotonic() >= deadline:
        raise SearchTimeout

    if terminal(state):
        return terminal_score(state, depth), None

    if depth == 0:
        return heuristic(state), None

    actions = generate_and_order_actions(state, player, deadline)
    best_action = None

    if player == 0:  # MAX_PLAYER
        best_score = -float("inf")
        for action in actions:
            child = apply_action(state, action, 0)
            score, _ = minimax(child, depth - 1, alpha, beta, 1, deadline)
            if is_root and recent_counts and action["type"] == "move":
                score -= REPEAT_PENALTY * recent_counts.get(tuple(action["to"]), 0)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score, best_action
    else:  # MIN_PLAYER
        best_score = float("inf")
        for action in actions:
            child = apply_action(state, action, 1)
            score, _ = minimax(child, depth - 1, alpha, beta, 0, deadline)
            if score < best_score:
                best_score = score
                best_action = action
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score, best_action


def choose_action(observation):
    """Arena entry point: Iterative Deepening Alpha-Beta Minimax."""
    start_time = time.monotonic()
    deadline = start_time + TIME_LIMIT
    legal_moves = observation.get("legal_moves", [])
    if not legal_moves:
        return {"type": "pass"}

    if observation.get("turn", 0) <= 1:
        RECENT_POSITIONS.clear()

    you = observation["you"]
    opp = observation["opponent"]
    my_pos = tuple(you["position"])
    opp_pos = tuple(opp["position"])
    RECENT_POSITIONS.append(my_pos)
    recent_counts = {pos: RECENT_POSITIONS.count(pos) for pos in RECENT_POSITIONS}

    # Immediate 1-step win check
    for m in legal_moves:
        if m[0] == you["goal_row"]:
            return {"type": "move", "to": list(m)}

    h_walls = frozenset((w[0], w[1]) for w in observation["walls"]["horizontal"])
    v_walls = frozenset((w[0], w[1]) for w in observation["walls"]["vertical"])
    state = State(
        pos=(my_pos, opp_pos),
        goals=(you["goal_row"], opp["goal_row"]),
        walls_rem=(you["walls_remaining"], opp["walls_remaining"]),
        h_walls=h_walls,
        v_walls=v_walls,
    )

    # Safe fallback: best 1-step pawn move by shortest path
    distances = observation["distances"]
    best_action = {
        "type": "move",
        "to": list(min(legal_moves, key=lambda m: distances[m[0]][m[1]] if distances[m[0]][m[1]] >= 0 else 999)),
    }

    # Iterative Deepening loop
    for depth in range(1, MAX_DEPTH + 1):
        try:
            _, action = minimax(
                state,
                depth,
                -float("inf"),
                float("inf"),
                0,
                deadline,
                is_root=True,
                recent_counts=recent_counts,
            )
            if action is not None:
                best_action = action
        except SearchTimeout:
            break

    return best_action
