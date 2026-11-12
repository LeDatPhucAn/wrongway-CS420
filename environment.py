"""Deterministic, dependency-free rules for the two-player Wrongway game.

Coordinates are zero-based ``(row, column)`` pairs.  Horizontal wall ``(r, c)``
blocks movement between rows r and r + 1 in columns c and c + 1.  Vertical wall
``(r, c)`` blocks movement between columns c and c + 1 in rows r and r + 1.
The public observations and transition results contain only JSON-safe values.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple


Cell = Tuple[int, int]
Adjacency = Dict[Cell, List[Cell]]
BOARD_SIZE = 9
WALLS_PER_PLAYER = 10
_DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))


class Environment:
    """A complete game state; every attempted action consumes one turn.

    Invalid actions leave the board intact and forfeit the current turn.  A
    turn limit ends the game in a draw, except that reaching the goal on the
    final allowed turn is still a win.  Calling ``step`` after the game ends
    raises ``RuntimeError``.
    """

    def __init__(self, max_turns: int = 500, starting_player: int = 0) -> None:
        if type(max_turns) is not int or max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        self._check_player(starting_player)
        self.max_turns = max_turns
        self.positions = [(8, 4), (0, 4)]
        self.goal_rows = [0, 8]
        self.walls_remaining = [WALLS_PER_PLAYER, WALLS_PER_PLAYER]
        self.horizontal_walls: Set[Cell] = set()
        self.vertical_walls: Set[Cell] = set()
        self.current_player = starting_player
        self.turn_count = 0
        self.winner: Optional[int] = None
        self.done = False
        self.termination_reason: Optional[str] = None

    @staticmethod
    def _check_player(player: int) -> None:
        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be 0 or 1")

    def _player(self, player: Optional[int]) -> int:
        if player is None:
            return self.current_player
        self._check_player(player)
        return player

    @staticmethod
    def _inside(cell: Cell) -> bool:
        return 0 <= cell[0] < BOARD_SIZE and 0 <= cell[1] < BOARD_SIZE

    def _edge_open(self, first: Cell, second: Cell) -> bool:
        """Whether adjacent board squares have an unobstructed shared edge."""
        if not self._inside(first) or not self._inside(second):
            return False
        row, col = first
        other_row, other_col = second
        if abs(row - other_row) + abs(col - other_col) != 1:
            return False
        if row != other_row:
            edge_row = min(row, other_row)
            return (edge_row, col) not in self.horizontal_walls and (
                edge_row, col - 1
            ) not in self.horizontal_walls
        edge_col = min(col, other_col)
        return (row, edge_col) not in self.vertical_walls and (
            row - 1, edge_col
        ) not in self.vertical_walls

    def _adjacency(self) -> Adjacency:
        graph: Adjacency = {}
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = (row, col)
                graph[cell] = [
                    (row + dr, col + dc)
                    for dr, dc in _DIRECTIONS
                    if self._edge_open(cell, (row + dr, col + dc))
                ]
        return graph

    def legal_moves(self, player: Optional[int] = None) -> List[List[int]]:
        """Legal destinations, including straight jumps or forced diagonals.

        A pawn can jump its adjacent opponent if the edge behind the opponent
        is open.  When that edge is blocked by a wall or the board boundary,
        the pawn may instead move to an open square beside the opponent.
        """
        player = self._player(player)
        if self.done:
            return []
        row, col = self.positions[player]
        opponent = self.positions[1 - player]
        destinations: Set[Cell] = set()
        for dr, dc in _DIRECTIONS:
            adjacent = (row + dr, col + dc)
            if not self._edge_open((row, col), adjacent):
                continue
            if adjacent != opponent:
                destinations.add(adjacent)
                continue
            beyond = (adjacent[0] + dr, adjacent[1] + dc)
            if self._edge_open(adjacent, beyond):
                destinations.add(beyond)
            else:
                for side_dr, side_dc in ((dc, dr), (-dc, -dr)):
                    beside = (adjacent[0] + side_dr, adjacent[1] + side_dc)
                    if self._edge_open(adjacent, beside):
                        destinations.add(beside)
        return [list(cell) for cell in sorted(destinations)]

    @staticmethod
    def _wall_edges(orientation: str, row: int, col: int) -> Set[Tuple[Cell, Cell]]:
        if orientation == "H":
            pairs = (
                ((row, col), (row + 1, col)),
                ((row, col + 1), (row + 1, col + 1)),
            )
        else:
            pairs = (
                ((row, col), (row, col + 1)),
                ((row + 1, col), (row + 1, col + 1)),
            )
        return {(a, b) for first, second in pairs for a, b in ((first, second), (second, first))}

    @staticmethod
    def _has_route(start: Cell, goal_row: int, graph: Adjacency,
                   blocked: Set[Tuple[Cell, Cell]]) -> bool:
        pending = deque([start])
        visited = {start}
        while pending:
            cell = pending.popleft()
            if cell[0] == goal_row:
                return True
            for neighbor in graph[cell]:
                if neighbor not in visited and (cell, neighbor) not in blocked:
                    visited.add(neighbor)
                    pending.append(neighbor)
        return False

    def _wall_error(self, player: int, orientation: str, row: int, col: int,
                    graph: Optional[Adjacency] = None) -> Optional[str]:
        if self.walls_remaining[player] <= 0:
            return "no_walls_remaining"
        if orientation == "H":
            if any((row, col + delta) in self.horizontal_walls for delta in (-1, 0, 1)):
                return "wall_overlap"
            if (row, col) in self.vertical_walls:
                return "wall_crossing"
        else:
            if any((row + delta, col) in self.vertical_walls for delta in (-1, 0, 1)):
                return "wall_overlap"
            if (row, col) in self.horizontal_walls:
                return "wall_crossing"
        if graph is None:
            graph = self._adjacency()
        blocked = self._wall_edges(orientation, row, col)
        for other in (0, 1):
            if not self._has_route(self.positions[other], self.goal_rows[other], graph, blocked):
                return "wall_blocks_path"
        return None

    def legal_wall_actions(self, player: Optional[int] = None) -> List[Dict[str, Any]]:
        """All playable walls, accounting for inventory and both escape routes.

        Entries omit ``type`` to match the compact observation schema.  Add
        ``type: 'wall'`` when submitting one as an action.
        """
        player = self._player(player)
        if self.done or self.walls_remaining[player] <= 0:
            return []
        graph = self._adjacency()
        return [
            {"orientation": orientation, "row": row, "col": col}
            for orientation in ("H", "V")
            for row in range(BOARD_SIZE - 1)
            for col in range(BOARD_SIZE - 1)
            if self._wall_error(player, orientation, row, col, graph) is None
        ]

    def distance_map(self, player: int) -> List[List[int]]:
        """BFS distances to this player's goal, respecting walls, ignoring pawns."""
        self._check_player(player)
        distances = [[-1] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        pending = deque()
        for col in range(BOARD_SIZE):
            goal = (self.goal_rows[player], col)
            distances[goal[0]][goal[1]] = 0
            pending.append(goal)
        graph = self._adjacency()
        while pending:
            row, col = pending.popleft()
            for next_row, next_col in graph[(row, col)]:
                if distances[next_row][next_col] == -1:
                    distances[next_row][next_col] = distances[row][col] + 1
                    pending.append((next_row, next_col))
        return distances

    def _walls_snapshot(self) -> Dict[str, List[List[int]]]:
        return {
            "horizontal": [list(cell) for cell in sorted(self.horizontal_walls)],
            "vertical": [list(cell) for cell in sorted(self.vertical_walls)],
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return a detached, serializable board state for logging or replay."""
        return {
            "board_size": BOARD_SIZE,
            "positions": [list(cell) for cell in self.positions],
            "goal_rows": list(self.goal_rows),
            "walls_remaining": list(self.walls_remaining),
            "walls": self._walls_snapshot(),
            "current_player": self.current_player,
            "turn": self.turn_count,
            "max_turns": self.max_turns,
            "done": self.done,
            "winner": self.winner,
            "termination_reason": self.termination_reason,
        }

    def get_observation(self, player: Optional[int] = None) -> Dict[str, Any]:
        """Return the complete board from one player's perspective."""
        player = self._player(player)
        opponent = 1 - player

        def pawn_info(who: int) -> Dict[str, Any]:
            return {
                "position": list(self.positions[who]),
                "goal_row": self.goal_rows[who],
                "walls_remaining": self.walls_remaining[who],
            }

        return {
            "board_size": BOARD_SIZE,
            "player_id": player,
            "current_player": self.current_player,
            "turn": self.turn_count,
            "max_turns": self.max_turns,
            "you": pawn_info(player),
            "opponent": pawn_info(opponent),
            "walls": self._walls_snapshot(),
            "legal_moves": self.legal_moves(player),
            "legal_walls": self.legal_wall_actions(player),
            "distances": self.distance_map(player),
            "done": self.done,
            "winner": self.winner,
        }

    def _finish_turn(self, status: str, reason: str,
                     action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        player = self.current_player
        self.turn_count += 1
        if self.positions[player][0] == self.goal_rows[player]:
            self.done = True
            self.winner = player
            self.termination_reason = "goal"
        elif self.turn_count >= self.max_turns:
            self.done = True
            self.termination_reason = "max_turns"
        self.current_player = 1 - player
        return {
            "status": status,
            "reason": reason,
            "player": player,
            "action": action,
            "winner": self.winner,
            "done": self.done,
        }

    def _require_active(self) -> None:
        if self.done:
            raise RuntimeError("the game has already ended")

    def step(self, action: Any) -> Dict[str, Any]:
        """Apply a move, wall placement, or pass; malformed actions forfeit a turn."""
        self._require_active()
        if not isinstance(action, dict):
            return self._finish_turn("invalid", "action_must_be_object", None)
        action_type = action.get("type")
        if action_type == "pass":
            return self.skip_turn("pass")
        if action_type == "move":
            destination = action.get("to")
            if not isinstance(destination, (list, tuple)) or len(destination) != 2 or any(
                type(value) is not int for value in destination
            ):
                return self._finish_turn("invalid", "invalid_move_coordinates", None)
            target = list(destination)
            if target not in self.legal_moves():
                return self._finish_turn("invalid", "illegal_move", None)
            self.positions[self.current_player] = tuple(destination)
            return self._finish_turn("ok", "move", {"type": "move", "to": target})
        if action_type == "wall":
            orientation = action.get("orientation")
            row, col = action.get("row"), action.get("col")
            if orientation not in ("H", "V"):
                return self._finish_turn("invalid", "invalid_wall_orientation", None)
            if type(row) is not int or type(col) is not int or not (
                0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1
            ):
                return self._finish_turn("invalid", "invalid_wall_coordinates", None)
            error = self._wall_error(self.current_player, orientation, row, col)
            if error is not None:
                return self._finish_turn("invalid", error, None)
            walls = self.horizontal_walls if orientation == "H" else self.vertical_walls
            walls.add((row, col))
            self.walls_remaining[self.current_player] -= 1
            return self._finish_turn("ok", "wall", {
                "type": "wall", "orientation": orientation, "row": row, "col": col,
            })
        return self._finish_turn("invalid", "unknown_action_type", None)

    def skip_turn(self, reason: str = "timeout") -> Dict[str, Any]:
        """Forfeit a turn, for example after a timeout or agent exception."""
        self._require_active()
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        return self._finish_turn("pass", reason, {"type": "pass"})
