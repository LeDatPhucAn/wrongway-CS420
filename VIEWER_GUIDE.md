# Wrongway Web Replay Viewer Guide

The Wrongway Replay Viewer is a modern, responsive web application inspired by [wrongway.app](https://wrongway.app/) and equipped with the playback, multi-match, and diagnostics capabilities of the bombIT tournament viewer.

It runs locally with **zero external dependencies** (using Python's standard library and vanilla browser HTML5/SVG/CSS).

---

## 1. Quick Start

### Option A: Open Any Match Replay Directly
```powershell
python replay.py logs/demo.json --web
```
This automatically launches your browser and loads the match with interactive replay controls.

### Option B: Start the Viewer & Browse All Logs
```powershell
python serve.py --open
```
Opens the viewer at `http://127.0.0.1:8420/`. Click the **📂 Logs** button in the top bar to browse, search, and switch between any match logs stored in `logs/`.

### Option C: Run a Match and Immediately View It
```powershell
python arena.py --player1 submissions/greedy.py --player2 submissions/wall_builder.py --log logs/my_match.json --open
```
Executes the game and opens the replay in the browser as soon as the winner is decided.

### Option D: Run a Multi-Match Series (Evaluation)
```powershell
python series.py --player1 submissions/greedy.py --player2 submissions/wall_builder.py --games 10 --out logs/series.json
python replay.py logs/series.json --web
```
Runs a 10-game tournament (alternating sides each game) and loads the entire series into a playlist where you can flip between games using `W` and `S`.

### Option E: Offline / No Server Needed
Double-click `viewer/index.html` in your file explorer to open it directly in Edge, Chrome, Firefox, or Safari (`file://`). Click **Open File** or drag & drop any `.json` match log onto the window.

---

## 2. Choosing and Switching Logs

The viewer reads directly from your `logs/` directory without requiring any system file explorer:

1. **Instant Selection on Startup:** When you open the viewer, all available logs in `logs/` are automatically detected and displayed as interactive cards on the home screen. Click any match to begin playing immediately.
2. **Top Bar "Open Log" Button:** Click **📂 Open Log** in the header at any time:
   - **Direct `logs/` Browser:** Browse all matches recorded in your `logs/` directory.
   - **Search & Filter:** Instantly filter by agent name (e.g. `greedy`), winner, or file name.
   - **Sort:** Sort by newest date, turn count, or filename.
   - **Live Refresh:** Click **↻ Refresh** to pick up newly saved matches without reloading the page.
3. **Drag & Drop:** You can still optionally drag any `.json` match or series file directly into the browser window.
4. **Recent History:** Remembers your recently viewed logs in local storage for quick access.

---

## 3. Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| <kbd>Space</kbd> | Play / Pause playback |
| <kbd>A</kbd> or <kbd>←</kbd> | Step back to previous turn |
| <kbd>D</kbd> or <kbd>→</kbd> | Step forward to next turn |
| <kbd>Q</kbd> or <kbd>Home</kbd> | Jump to first frame (turn 0) |
| <kbd>E</kbd> or <kbd>End</kbd> | Jump to final frame |
| <kbd>W</kbd> | Previous match in series / playlist |
| <kbd>S</kbd> | Next match in series / playlist |
| <kbd>+</kbd> / <kbd>-</kbd> | Increase / decrease playback speed (0.25x – 8x) |
| <kbd>P</kbd> | Cycle shortest path overlay (Both → P0 → P1 → None) |
| <kbd>F</kbd> | Flip board orientation (view from Player 1's side) |
| <kbd>T</kbd> | Toggle Dark / Light theme |
| <kbd>?</kbd> | Show keyboard shortcuts modal |
| <kbd>Esc</kbd> | Close any open modal or dropdown |

---

## 4. UI & Visual Features

- **wrongway.app Design System:** Clean rounded cards, soft shadows, 40px grid background, and responsive layout for mobile and desktop.
- **Player Cards:**
  - **Wall Tracker:** 10 visual pill segments that deplete as walls are used, plus numerical counter.
  - **Goal Distance:** Live BFS shortest-path distance to opposite goal row, with delta indicator comparing to the previous turn (e.g. `▲ -1` or `▼ +2`).
  - **Latency Sparkline:** Mini time-series chart showing each agent's response time per turn, with a dashed red limit line at 1.0s.
  - **Cumulative Turn Statistics:** Tracks OK moves, Passes, Timeouts, Invalids, and Exceptions up to the active frame.
- **Board Overlays (View Menu 👁):**
  - **Shortest Path:** BFS dotted path in red (Player 0) and blue (Player 1) showing optimal paths to the goal.
  - **Distance Heatmap:** Displays BFS distance from each cell to the selected player's goal.
  - **Wall Coloring:** Toggles distinguishing wall ownership by player color or uniform neutral slate.
  - **Flip Board:** Rotates board view 180° for Player 1 perspective while keeping coordinate labels accurate.
- **Turn Scrubber & Event Markers:**
  - Colored ticks along the timeline highlight where walls were placed, turns were passed, or invalid/timeout events occurred.
  - Hovering any tick reveals a quick summary tooltip.
- **Interactive Move List:**
  - Chronological list of turns with filter chips (**All**, **Walls**, **Issues**).
  - Clicking any turn immediately jumps to that state.
  - Auto-scrolls with playback.
- **Match Summary:** Reaching the final frame displays a winner summary and head-to-head comparison table.
- **Exporting Snapshots:** Click the **📷 PNG** button to download a high-resolution image of the current board state.

---

## 5. Running Tests

Run the backend test suite:
```powershell
python -m unittest tests/test_viewer_backend.py
```
This tests static file delivery, `/api/logs` directory indexing, path traversal safety, series execution, and BFS distance parity.
