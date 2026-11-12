"""Ví dụ nâng cao: BFS để ước lượng lợi ích của một tường.

File này độc lập, chỉ dùng thư viện chuẩn. BFS bỏ qua quân cờ; đây là
ước lượng chiến thuật, còn legal_moves/legal_walls do arena kiểm tra luật.
"""

from collections import deque


def add_wall_edges(blocked, orientation, row, col):
    """Thêm hai cạnh bị tường chặn, theo cả hai chiều, vào một set."""
    if orientation == "H":
        pairs = [
            ((row, col), (row + 1, col)),
            ((row, col + 1), (row + 1, col + 1)),
        ]
    else:
        pairs = [
            ((row, col), (row, col + 1)),
            ((row + 1, col), (row + 1, col + 1)),
        ]

    for start, end in pairs:
        blocked.add((start, end))
        blocked.add((end, start))


def blocked_edges(walls):
    blocked = set()
    for row, col in walls["horizontal"]:
        add_wall_edges(blocked, "H", row, col)
    for row, col in walls["vertical"]:
        add_wall_edges(blocked, "V", row, col)
    return blocked


def distance_to_goal(position, goal_row, size, blocked):
    """BFS: số bước ít nhất từ position tới một ô bất kỳ trên goal_row."""
    start = tuple(position)
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current, distance = queue.popleft()
        row, col = current
        if row == goal_row:
            return distance

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if neighbor in visited or (current, neighbor) in blocked:
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))

    # Tường hợp lệ luôn giữ đường tới đích; giá trị này phòng trường hợp
    # dùng hàm BFS với một cấu hình tùy ý khi tự thử nghiệm.
    return size * size


def choose_action(observation):
    moves = observation["legal_moves"]
    you = observation["you"]
    opponent = observation["opponent"]

    # Có thể về đích ngay thì không cần cân nhắc đặt tường.
    for move in moves:
        if move[0] == you["goal_row"]:
            return {"type": "move", "to": list(move)}

    distances = observation["distances"]
    best_move = min(
        moves,
        key=lambda move: distances[move[0]][move[1]]
        if distances[move[0]][move[1]] >= 0
        else float("inf"),
        default=None,
    )

    size = observation["board_size"]
    blocked = blocked_edges(observation["walls"])
    own_distance = distance_to_goal(you["position"], you["goal_row"], size, blocked)
    enemy_distance = distance_to_goal(
        opponent["position"], opponent["goal_row"], size, blocked
    )

    best_wall = None
    # Minh họa: ưu tiên tường làm đối thủ chậm hơn mình. Đặt tường cũng mất
    # một lượt, nên lợi ích về quãng đường này chưa bảo đảm lợi thế thực tế.
    best_gain = 0
    if enemy_distance <= own_distance + 1:
        for wall in observation["legal_walls"]:
            candidate = blocked.copy()
            add_wall_edges(candidate, wall["orientation"], wall["row"], wall["col"])
            own_after = distance_to_goal(
                you["position"], you["goal_row"], size, candidate
            )
            enemy_after = distance_to_goal(
                opponent["position"], opponent["goal_row"], size, candidate
            )
            gain = (enemy_after - enemy_distance) - (own_after - own_distance)
            if gain > best_gain:
                best_gain = gain
                best_wall = wall

    if best_wall is not None:
        return {
            "type": "wall",
            "orientation": best_wall["orientation"],
            "row": best_wall["row"],
            "col": best_wall["col"],
        }
    if best_move is not None:
        return {"type": "move", "to": list(best_move)}
    return {"type": "pass"}
