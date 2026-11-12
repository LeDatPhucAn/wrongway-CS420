"""Bot tham lam: luôn đi theo khoảng cách ngắn nhất tới hàng đích."""


def choose_action(observation):
    moves = observation["legal_moves"]
    if not moves:
        return {"type": "pass"}

    distances = observation["distances"]
    reachable = [move for move in moves if distances[move[0]][move[1]] >= 0]
    candidates = reachable or moves
    best_move = min(
        candidates,
        key=lambda move: distances[move[0]][move[1]]
        if distances[move[0]][move[1]] >= 0
        else float("inf"),
    )
    return {"type": "move", "to": list(best_move)}
