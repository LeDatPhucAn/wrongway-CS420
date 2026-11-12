/**
 * Geometry and coordinate calculations for Wrongway 9x9 board.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};

  const BOARD_SIZE = 9;
  const NUM_WALLS = 10;
  const CELL_SIZE = 50;
  const GUTTER = 12;
  const PADDING = 28;

  const BOARD_PX = BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * GUTTER; // 546
  const TOTAL_PX = BOARD_PX + 2 * PADDING; // 602

  function cellToXY(row, col) {
    const x = PADDING + col * (CELL_SIZE + GUTTER);
    const y = PADDING + row * (CELL_SIZE + GUTTER);
    return { x, y, size: CELL_SIZE };
  }

  function centerOfCell(row, col) {
    const xy = cellToXY(row, col);
    return {
      cx: xy.x + CELL_SIZE / 2,
      cy: xy.y + CELL_SIZE / 2
    };
  }

  function wallToRect(orientation, row, col) {
    // Horizontal: between row & row+1, spans col & col+1
    // Vertical: between col & col+1, spans row & row+1
    const rx = GUTTER / 2;
    if (orientation === 'H') {
      const x = PADDING + col * (CELL_SIZE + GUTTER);
      const y = PADDING + row * (CELL_SIZE + GUTTER) + CELL_SIZE;
      const width = 2 * CELL_SIZE + GUTTER;
      const height = GUTTER;
      return { x, y, width, height, rx };
    } else {
      const x = PADDING + col * (CELL_SIZE + GUTTER) + CELL_SIZE;
      const y = PADDING + row * (CELL_SIZE + GUTTER);
      const width = GUTTER;
      const height = 2 * CELL_SIZE + GUTTER;
      return { x, y, width, height, rx };
    }
  }

  function isJump(fromCell, toCell) {
    if (!fromCell || !toCell) return false;
    const dr = Math.abs(fromCell[0] - toCell[0]);
    const dc = Math.abs(fromCell[1] - toCell[1]);
    // Manhattan distance 2 (straight jump) or diagonal jump
    return (dr === 2 && dc === 0) || (dr === 0 && dc === 2) || (dr === 1 && dc === 1);
  }

  window.WW.geometry = {
    BOARD_SIZE,
    NUM_WALLS,
    CELL_SIZE,
    GUTTER,
    PADDING,
    BOARD_PX,
    TOTAL_PX,
    cellToXY,
    centerOfCell,
    wallToRect,
    isJump
  };
})(window);
