/**
 * Rules and graph calculations matching environment.py exactly.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};
  const BOARD_SIZE = 9;
  const DIRECTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

  function wallKey(r, c) {
    return `${r},${c}`;
  }

  function makeWallSet(wallsList) {
    const set = new Set();
    if (!wallsList) return set;
    for (let i = 0; i < wallsList.length; i++) {
      set.add(wallKey(wallsList[i][0], wallsList[i][1]));
    }
    return set;
  }

  function inside(r, c) {
    return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
  }

  function edgeOpen(r1, c1, r2, c2, hSet, vSet) {
    if (!inside(r1, c1) || !inside(r2, c2)) return false;
    if (Math.abs(r1 - r2) + Math.abs(c1 - c2) !== 1) return false;

    if (r1 !== r2) {
      const edgeRow = Math.min(r1, r2);
      if (hSet.has(wallKey(edgeRow, c1))) return false;
      if (hSet.has(wallKey(edgeRow, c1 - 1))) return false;
      return true;
    } else {
      const edgeCol = Math.min(c1, c2);
      if (vSet.has(wallKey(r1, edgeCol))) return false;
      if (vSet.has(wallKey(r1 - 1, edgeCol))) return false;
      return true;
    }
  }

  function computeDistanceMap(goalRow, horizontalWalls, verticalWalls) {
    const hSet = horizontalWalls instanceof Set ? horizontalWalls : makeWallSet(horizontalWalls);
    const vSet = verticalWalls instanceof Set ? verticalWalls : makeWallSet(verticalWalls);

    const distances = [];
    for (let r = 0; r < BOARD_SIZE; r++) {
      distances.push(new Array(BOARD_SIZE).fill(-1));
    }

    const queue = [];
    for (let c = 0; c < BOARD_SIZE; c++) {
      distances[goalRow][c] = 0;
      queue.push([goalRow, c]);
    }

    let head = 0;
    while (head < queue.length) {
      const [r, c] = queue[head++];
      const currentDist = distances[r][c];

      for (let i = 0; i < DIRECTIONS.length; i++) {
        const nr = r + DIRECTIONS[i][0];
        const nc = c + DIRECTIONS[i][1];
        if (inside(nr, nc) && distances[nr][nc] === -1) {
          if (edgeOpen(r, c, nr, nc, hSet, vSet)) {
            distances[nr][nc] = currentDist + 1;
            queue.push([nr, nc]);
          }
        }
      }
    }

    return distances;
  }

  function computeShortestPath(startCell, goalRow, horizontalWalls, verticalWalls) {
    if (!startCell) return [];
    const distances = computeDistanceMap(goalRow, horizontalWalls, verticalWalls);
    let curR = startCell[0];
    let curC = startCell[1];
    if (distances[curR][curC] === -1) return [];

    const path = [[curR, curC]];
    const hSet = horizontalWalls instanceof Set ? horizontalWalls : makeWallSet(horizontalWalls);
    const vSet = verticalWalls instanceof Set ? verticalWalls : makeWallSet(verticalWalls);

    while (curR !== goalRow) {
      const currentDist = distances[curR][curC];
      let nextR = -1;
      let nextC = -1;
      let found = false;

      // Prefer moving forward towards goal
      for (let i = 0; i < DIRECTIONS.length; i++) {
        const nr = curR + DIRECTIONS[i][0];
        const nc = curC + DIRECTIONS[i][1];
        if (inside(nr, nc) && distances[nr][nc] === currentDist - 1) {
          if (edgeOpen(curR, curC, nr, nc, hSet, vSet)) {
            nextR = nr;
            nextC = nc;
            found = true;
            break;
          }
        }
      }

      if (!found) break;
      curR = nextR;
      curC = nextC;
      path.push([curR, curC]);
      if (path.length > 100) break; // Guard against infinite loop
    }

    return path;
  }

  window.WW.rules = {
    edgeOpen,
    computeDistanceMap,
    computeShortestPath,
    makeWallSet
  };
})(window);
