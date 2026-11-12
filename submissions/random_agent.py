"""Bot ngẫu nhiên dùng random của Python; arena khởi tạo seed cho mỗi bot."""

import random


def choose_action(observation):
    moves = observation["legal_moves"]
    walls = observation["legal_walls"]

    # Thỉnh thoảng đặt một tường hợp lệ; phần lớn lượt dùng để di chuyển.
    if walls and (not moves or random.random() < 0.2):
        wall = random.choice(walls)
        return {
            "type": "wall",
            "orientation": wall["orientation"],
            "row": wall["row"],
            "col": wall["col"],
        }

    if moves:
        return {"type": "move", "to": list(random.choice(moves))}

    return {"type": "pass"}
