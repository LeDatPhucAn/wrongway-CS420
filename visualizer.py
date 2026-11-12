"""Plain terminal board: 0/1 are pawns, # and = are walls."""


def render_board(state, names=None):
    size = state["board_size"]
    width = 4 * size - 1
    canvas = [[" "] * width for _ in range(2 * size - 1)]
    for row in range(size):
        for col in range(size):
            canvas[2 * row][4 * col + 1] = "."
    for row, col in state["walls"]["horizontal"]:
        for x in range(4 * col, 4 * col + 7):
            canvas[2 * row + 1][x] = "="
    for row, col in state["walls"]["vertical"]:
        for y in range(2 * row, 2 * row + 3):
            canvas[y][4 * col + 3] = "#"
    for player, (row, col) in enumerate(state["positions"]):
        canvas[2 * row][4 * col + 1] = str(player)
    lines = ["     " + "   ".join(str(col) for col in range(size))]
    for index, line in enumerate(canvas):
        label = str(index // 2) if index % 2 == 0 else " "
        lines.append(" {}  {}".format(label, "".join(line).rstrip()))
    names = names or ["Player 0", "Player 1"]
    lines.append(" | ".join("P{} {}: {} walls, goal row {}".format(
        p, names[p], state["walls_remaining"][p], state["goal_rows"][p])
        for p in range(2)))
    return "\n".join(lines)
