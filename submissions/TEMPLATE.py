"""Mẫu bài nộp: chỉ cần hoàn thiện hàm choose_action(observation).

Sao chép file này thành MSSV.py rồi thay đổi chiến thuật của bạn.
Không cần cài đặt luật chơi hoặc nhập thư viện của arena.
"""


def choose_action(observation):
    """Đi tới ô hợp lệ có khoảng cách tới đích nhỏ nhất."""
    legal_moves = observation["legal_moves"]
    distances = observation["distances"]

    if not legal_moves:
        return {"type": "pass"}

    best_move = legal_moves[0]
    best_distance = float("inf")

    # Mỗi move là một list [hàng, cột]. Khoảng cách -1 là không tới được.
    for move in legal_moves:
        row, col = move
        distance = distances[row][col]
        if distance >= 0 and distance < best_distance:
            best_move = move
            best_distance = distance

    return {"type": "move", "to": list(best_move)}
