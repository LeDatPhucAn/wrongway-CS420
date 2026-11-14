/**
 * SVG Board Renderer for Wrongway.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};
  const { geometry, rules } = window.WW;
  const SVG_NS = 'http://www.w3.org/2000/svg';

  class BoardRenderer {
    constructor(container) {
      this.container = container;
      this.options = {
        showCoords: true,
        showPath: 'both', // 'none', 'p0', 'p1', 'both'
        showHeatmap: 'none', // 'none', 'p0', 'p1'
        flipBoard: false,
        colorWallsByOwner: true,
        animations: true
      };

      this.currentState = null;
      this.prevState = null;
      this.currentTurnRecord = null;
      this.wallOwners = new Map(); // key "H:r:c" -> player id
      this.interactiveConfig = null;

      this._initSVG();
    }

    _initSVG() {
      this.container.innerHTML = '';
      const svg = document.createElementNS(SVG_NS, 'svg');
      svg.setAttribute('viewBox', `0 0 ${geometry.TOTAL_PX} ${geometry.TOTAL_PX}`);
      svg.setAttribute('class', 'ww-board-svg');

      // Definitions (gradients, filters, markers)
      const defs = document.createElementNS(SVG_NS, 'defs');
      defs.innerHTML = `
        <filter id="pawn-glow-p0" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#e54d3c" flood-opacity="0.45"/>
        </filter>
        <filter id="pawn-glow-p1" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#3b82f6" flood-opacity="0.45"/>
        </filter>
        <filter id="wall-shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="rgba(15,23,42,0.25)"/>
        </filter>
        <linearGradient id="p0-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#ef4444"/>
          <stop offset="100%" stop-color="#b91c1c"/>
        </linearGradient>
        <linearGradient id="p1-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#3b82f6"/>
          <stop offset="100%" stop-color="#1d4ed8"/>
        </linearGradient>
        <linearGradient id="wall-gloss" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="rgba(255,255,255,0.2)"/>
          <stop offset="100%" stop-color="rgba(0,0,0,0.15)"/>
        </linearGradient>
        <marker id="arrow-p0" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#ef4444"/>
        </marker>
        <marker id="arrow-p1" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#3b82f6"/>
        </marker>
      `;
      svg.appendChild(defs);

      // Group layers
      this.boardGroup = document.createElementNS(SVG_NS, 'g');
      this.boardGroup.setAttribute('class', 'ww-board-group');

      this.coordsLayer = document.createElementNS(SVG_NS, 'g');
      this.coordsLayer.setAttribute('class', 'ww-coords-layer');

      this.cellsLayer = document.createElementNS(SVG_NS, 'g');
      this.cellsLayer.setAttribute('class', 'ww-cells-layer');

      this.goalTintsLayer = document.createElementNS(SVG_NS, 'g');
      this.goalTintsLayer.setAttribute('class', 'ww-goal-tints-layer');

      this.heatmapLayer = document.createElementNS(SVG_NS, 'g');
      this.heatmapLayer.setAttribute('class', 'ww-heatmap-layer');

      this.pathLayer = document.createElementNS(SVG_NS, 'g');
      this.pathLayer.setAttribute('class', 'ww-path-layer');

      this.lastMoveLayer = document.createElementNS(SVG_NS, 'g');
      this.lastMoveLayer.setAttribute('class', 'ww-last-move-layer');

      this.wallsLayer = document.createElementNS(SVG_NS, 'g');
      this.wallsLayer.setAttribute('class', 'ww-walls-layer');

      this.pawnsLayer = document.createElementNS(SVG_NS, 'g');
      this.pawnsLayer.setAttribute('class', 'ww-pawns-layer');

      this.interactiveLayer = document.createElementNS(SVG_NS, 'g');
      this.interactiveLayer.setAttribute('class', 'ww-interactive-layer');

      this.boardGroup.appendChild(this.coordsLayer);
      this.boardGroup.appendChild(this.goalTintsLayer);
      this.boardGroup.appendChild(this.cellsLayer);
      this.boardGroup.appendChild(this.heatmapLayer);
      this.boardGroup.appendChild(this.pathLayer);
      this.boardGroup.appendChild(this.lastMoveLayer);
      this.boardGroup.appendChild(this.wallsLayer);
      this.boardGroup.appendChild(this.pawnsLayer);
      this.boardGroup.appendChild(this.interactiveLayer);

      svg.appendChild(this.boardGroup);
      svg.addEventListener('contextmenu', (e) => {
        if (this.interactiveConfig && this.interactiveConfig.enabled && this.interactiveConfig.onToggleOrientation) {
          e.preventDefault();
          this.interactiveConfig.onToggleOrientation();
        }
      });
      this.container.appendChild(svg);
      this.svg = svg;

      this._renderStaticBoard();
    }

    setInteractive(config) {
      this.interactiveConfig = config || null;
      this._renderInteractive();
    }

    setOptions(newOptions) {
      Object.assign(this.options, newOptions);
      this._renderStaticBoard();
      if (this.currentState) {
        this.render(this.currentState, this.prevState, this.currentTurnRecord);
      }
    }

    setWallOwners(wallOwnersMap) {
      this.wallOwners = wallOwnersMap || new Map();
    }

    _renderStaticBoard() {
      // Clear static elements
      this.coordsLayer.innerHTML = '';
      this.cellsLayer.innerHTML = '';
      this.goalTintsLayer.innerHTML = '';

      const { BOARD_SIZE, PADDING, CELL_SIZE, GUTTER, TOTAL_PX } = geometry;

      // Flip board class on svg
      if (this.options.flipBoard) {
        this.svg.classList.add('ww-board-flipped');
      } else {
        this.svg.classList.remove('ww-board-flipped');
      }

      // Draw coordinates if enabled
      if (this.options.showCoords) {
        for (let i = 0; i < BOARD_SIZE; i++) {
          const center = PADDING + i * (CELL_SIZE + GUTTER) + CELL_SIZE / 2;

          // Col labels (top & bottom)
          const topText = document.createElementNS(SVG_NS, 'text');
          topText.setAttribute('x', center);
          topText.setAttribute('y', PADDING / 2 + 4);
          topText.setAttribute('class', 'ww-coord-text');
          topText.textContent = i;
          this.coordsLayer.appendChild(topText);

          const bottomText = document.createElementNS(SVG_NS, 'text');
          bottomText.setAttribute('x', center);
          bottomText.setAttribute('y', TOTAL_PX - PADDING / 2 + 4);
          bottomText.setAttribute('class', 'ww-coord-text');
          bottomText.textContent = i;
          this.coordsLayer.appendChild(bottomText);

          // Row labels (left & right)
          const leftText = document.createElementNS(SVG_NS, 'text');
          leftText.setAttribute('x', PADDING / 2);
          leftText.setAttribute('y', center + 4);
          leftText.setAttribute('class', 'ww-coord-text');
          leftText.textContent = i;
          this.coordsLayer.appendChild(leftText);

          const rightText = document.createElementNS(SVG_NS, 'text');
          rightText.setAttribute('x', TOTAL_PX - PADDING / 2);
          rightText.setAttribute('y', center + 4);
          rightText.setAttribute('class', 'ww-coord-text');
          rightText.textContent = i;
          this.coordsLayer.appendChild(rightText);
        }
      }

      // Goal row tints
      // Row 0 is Player 0's goal (target for red pawn)
      const r0xy = geometry.cellToXY(0, 0);
      const r8xy = geometry.cellToXY(8, 0);
      const rowWidth = BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * GUTTER;

      const p0GoalRect = document.createElementNS(SVG_NS, 'rect');
      p0GoalRect.setAttribute('x', r0xy.x - 4);
      p0GoalRect.setAttribute('y', r0xy.y - 4);
      p0GoalRect.setAttribute('width', rowWidth + 8);
      p0GoalRect.setAttribute('height', CELL_SIZE + 8);
      p0GoalRect.setAttribute('rx', 8);
      p0GoalRect.setAttribute('class', 'ww-goal-row ww-goal-p0');
      this.goalTintsLayer.appendChild(p0GoalRect);

      // Row 8 is Player 1's goal (target for blue pawn)
      const p1GoalRect = document.createElementNS(SVG_NS, 'rect');
      p1GoalRect.setAttribute('x', r8xy.x - 4);
      p1GoalRect.setAttribute('y', r8xy.y - 4);
      p1GoalRect.setAttribute('width', rowWidth + 8);
      p1GoalRect.setAttribute('height', CELL_SIZE + 8);
      p1GoalRect.setAttribute('rx', 8);
      p1GoalRect.setAttribute('class', 'ww-goal-row ww-goal-p1');
      this.goalTintsLayer.appendChild(p1GoalRect);

      // Draw Cells
      for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
          const { x, y, size } = geometry.cellToXY(r, c);
          const rect = document.createElementNS(SVG_NS, 'rect');
          rect.setAttribute('x', x);
          rect.setAttribute('y', y);
          rect.setAttribute('width', size);
          rect.setAttribute('height', size);
          rect.setAttribute('rx', 6);
          const cellClass = (r + c) % 2 === 0 ? 'ww-cell ww-cell-a' : 'ww-cell ww-cell-b';
          rect.setAttribute('class', cellClass);
          rect.setAttribute('data-row', r);
          rect.setAttribute('data-col', c);
          this.cellsLayer.appendChild(rect);
        }
      }
    }

    render(state, prevState = null, turnRecord = null) {
      this.currentState = state;
      this.prevState = prevState;
      this.currentTurnRecord = turnRecord;

      this._renderHeatmap(state);
      this._renderShortestPaths(state);
      this._renderLastMove(state, prevState, turnRecord);
      this._renderWalls(state, prevState, turnRecord);
      this._renderPawns(state, prevState, turnRecord);
      this._renderInteractive();
    }

    _renderInteractive() {
      if (!this.interactiveLayer) return;
      this.interactiveLayer.innerHTML = '';

      const cfg = this.interactiveConfig;
      if (!cfg || !cfg.enabled || cfg.busy) return;

      const mode = cfg.mode || 'move';
      if (mode === 'move') {
        const moves = cfg.legalMoves || [];
        for (let i = 0; i < moves.length; i++) {
          const [r, c] = moves[i];
          const cellXY = geometry.cellToXY(r, c);
          const { cx, cy } = geometry.centerOfCell(r, c);

          const g = document.createElementNS(SVG_NS, 'g');
          g.setAttribute('class', 'ww-move-target');

          const hit = document.createElementNS(SVG_NS, 'rect');
          hit.setAttribute('x', cellXY.x);
          hit.setAttribute('y', cellXY.y);
          hit.setAttribute('width', cellXY.size);
          hit.setAttribute('height', cellXY.size);
          hit.setAttribute('rx', 6);
          hit.setAttribute('class', 'ww-move-target-hit');

          const ring = document.createElementNS(SVG_NS, 'circle');
          ring.setAttribute('cx', cx);
          ring.setAttribute('cy', cy);
          ring.setAttribute('r', geometry.CELL_SIZE * 0.28);
          ring.setAttribute('class', 'ww-move-target-ring');

          const dot = document.createElementNS(SVG_NS, 'circle');
          dot.setAttribute('cx', cx);
          dot.setAttribute('cy', cy);
          dot.setAttribute('r', 5);
          dot.setAttribute('class', 'ww-move-target-dot');

          g.appendChild(hit);
          g.appendChild(ring);
          g.appendChild(dot);
          g.addEventListener('click', () => {
            if (cfg.onMove) cfg.onMove([r, c]);
          });
          this.interactiveLayer.appendChild(g);
        }
      } else if (mode === 'H' || mode === 'V') {
        const walls = cfg.legalWalls || [];
        for (let i = 0; i < walls.length; i++) {
          const w = walls[i];
          if (w.orientation !== mode) continue;

          const rect = geometry.wallToRect(mode, w.row, w.col);
          const g = document.createElementNS(SVG_NS, 'g');
          g.setAttribute('class', 'ww-wall-slot');

          const preview = document.createElementNS(SVG_NS, 'rect');
          preview.setAttribute('x', rect.x);
          preview.setAttribute('y', rect.y);
          preview.setAttribute('width', rect.width);
          preview.setAttribute('height', rect.height);
          preview.setAttribute('rx', rect.rx);
          preview.setAttribute('class', 'ww-wall-ghost');

          // Slightly expanded hit area for effortless clicking
          const hitPad = 4;
          const hit = document.createElementNS(SVG_NS, 'rect');
          hit.setAttribute('x', rect.x - (mode === 'V' ? hitPad : 0));
          hit.setAttribute('y', rect.y - (mode === 'H' ? hitPad : 0));
          hit.setAttribute('width', rect.width + (mode === 'V' ? hitPad * 2 : 0));
          hit.setAttribute('height', rect.height + (mode === 'H' ? hitPad * 2 : 0));
          hit.setAttribute('fill', 'transparent');
          hit.setAttribute('class', 'ww-wall-hit');

          g.appendChild(preview);
          g.appendChild(hit);
          g.addEventListener('click', () => {
            if (cfg.onWall) cfg.onWall({ orientation: mode, row: w.row, col: w.col });
          });
          this.interactiveLayer.appendChild(g);
        }
      }
    }

    _renderHeatmap(state) {
      this.heatmapLayer.innerHTML = '';
      if (this.options.showHeatmap === 'none') return;

      const player = this.options.showHeatmap === 'p0' ? 0 : 1;
      const goalRow = state.goal_rows ? state.goal_rows[player] : (player === 0 ? 0 : 8);
      const distances = rules.computeDistanceMap(
        goalRow,
        state.walls.horizontal,
        state.walls.vertical
      );

      for (let r = 0; r < geometry.BOARD_SIZE; r++) {
        for (let c = 0; c < geometry.BOARD_SIZE; c++) {
          const d = distances[r][c];
          if (d < 0) continue;
          const { cx, cy } = geometry.centerOfCell(r, c);
          const text = document.createElementNS(SVG_NS, 'text');
          text.setAttribute('x', cx);
          text.setAttribute('y', cy + 4);
          text.setAttribute('class', 'ww-heatmap-text');
          text.textContent = d;
          this.heatmapLayer.appendChild(text);
        }
      }
    }

    _renderShortestPaths(state) {
      this.pathLayer.innerHTML = '';
      if (this.options.showPath === 'none') return;

      const showP0 = this.options.showPath === 'p0' || this.options.showPath === 'both';
      const showP1 = this.options.showPath === 'p1' || this.options.showPath === 'both';

      if (showP0 && state.positions && state.positions[0]) {
        this._drawPathPolyline(state.positions[0], 0, state.walls, 'p0');
      }
      if (showP1 && state.positions && state.positions[1]) {
        this._drawPathPolyline(state.positions[1], 8, state.walls, 'p1');
      }
    }

    _drawPathPolyline(startPos, goalRow, walls, playerClass) {
      const path = rules.computeShortestPath(startPos, goalRow, walls.horizontal, walls.vertical);
      if (path.length < 2) return;

      const points = path.map(([r, c]) => {
        const { cx, cy } = geometry.centerOfCell(r, c);
        return `${cx},${cy}`;
      }).join(' ');

      const polyline = document.createElementNS(SVG_NS, 'polyline');
      polyline.setAttribute('points', points);
      polyline.setAttribute('class', `ww-path-line ww-path-${playerClass}`);
      this.pathLayer.appendChild(polyline);
    }

    _renderLastMove(state, prevState, turnRecord) {
      this.lastMoveLayer.innerHTML = '';
      if (!turnRecord || turnRecord.status !== 'ok' || !turnRecord.action) return;

      const action = turnRecord.action;
      if (action.type !== 'move') return;

      const player = turnRecord.player;
      const prevPos = prevState && prevState.positions ? prevState.positions[player] : null;
      const newPos = action.to;

      if (!prevPos || !newPos) return;

      const fromPt = geometry.centerOfCell(prevPos[0], prevPos[1]);
      const toPt = geometry.centerOfCell(newPos[0], newPos[1]);

      // Ghost at from position
      const ghost = document.createElementNS(SVG_NS, 'circle');
      ghost.setAttribute('cx', fromPt.cx);
      ghost.setAttribute('cy', fromPt.cy);
      ghost.setAttribute('r', geometry.CELL_SIZE * 0.34);
      ghost.setAttribute('class', `ww-last-move-ghost ww-pawn-${player === 0 ? 'p0' : 'p1'}`);
      this.lastMoveLayer.appendChild(ghost);

      // Arrow line from previous to new position
      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', fromPt.cx);
      line.setAttribute('y1', fromPt.cy);
      line.setAttribute('x2', toPt.cx);
      line.setAttribute('y2', toPt.cy);
      line.setAttribute('class', `ww-last-move-arrow ww-arrow-p${player}`);
      line.setAttribute('marker-end', `url(#arrow-p${player})`);
      this.lastMoveLayer.appendChild(line);
    }

    _renderWalls(state, prevState, turnRecord) {
      this.wallsLayer.innerHTML = '';
      const horizontal = state.walls ? state.walls.horizontal : [];
      const vertical = state.walls ? state.walls.vertical : [];

      const newlyPlacedKey = (turnRecord && turnRecord.status === 'ok' && turnRecord.action && turnRecord.action.type === 'wall')
        ? `${turnRecord.action.orientation}:${turnRecord.action.row}:${turnRecord.action.col}`
        : null;

      // Draw horizontal walls
      for (let i = 0; i < horizontal.length; i++) {
        const [r, c] = horizontal[i];
        const rect = geometry.wallToRect('H', r, c);
        const key = `H:${r}:${c}`;
        const owner = this.wallOwners.get(key);
        const isNew = key === newlyPlacedKey;
        this._createWallElement(rect, 'H', owner, isNew);
      }

      // Draw vertical walls
      for (let i = 0; i < vertical.length; i++) {
        const [r, c] = vertical[i];
        const rect = geometry.wallToRect('V', r, c);
        const key = `V:${r}:${c}`;
        const owner = this.wallOwners.get(key);
        const isNew = key === newlyPlacedKey;
        this._createWallElement(rect, 'V', owner, isNew);
      }
    }

    _createWallElement(rectCoords, orientation, owner, isNew) {
      const g = document.createElementNS(SVG_NS, 'g');
      let classes = `ww-wall ww-wall-${orientation.toLowerCase()}`;
      if (this.options.colorWallsByOwner && owner !== undefined) {
        classes += ` ww-wall-p${owner}`;
      }
      if (isNew && this.options.animations) {
        classes += ' ww-wall-new';
      }
      g.setAttribute('class', classes);

      const rect = document.createElementNS(SVG_NS, 'rect');
      rect.setAttribute('x', rectCoords.x);
      rect.setAttribute('y', rectCoords.y);
      rect.setAttribute('width', rectCoords.width);
      rect.setAttribute('height', rectCoords.height);
      rect.setAttribute('rx', rectCoords.rx);
      rect.setAttribute('filter', 'url(#wall-shadow)');
      g.appendChild(rect);

      // Gloss top line
      const gloss = document.createElementNS(SVG_NS, 'rect');
      gloss.setAttribute('x', rectCoords.x + 1);
      gloss.setAttribute('y', rectCoords.y + 1);
      gloss.setAttribute('width', Math.max(1, rectCoords.width - 2));
      gloss.setAttribute('height', Math.max(1, rectCoords.height / 2 - 1));
      gloss.setAttribute('rx', rectCoords.rx / 2);
      gloss.setAttribute('fill', 'url(#wall-gloss)');
      g.appendChild(gloss);

      this.wallsLayer.appendChild(g);
    }

    _renderPawns(state, prevState, turnRecord) {
      this.pawnsLayer.innerHTML = '';
      if (!state.positions) return;

      for (let player = 0; player < 2; player++) {
        const pos = state.positions[player];
        if (!pos) continue;

        const { cx, cy } = geometry.centerOfCell(pos[0], pos[1]);
        const radius = geometry.CELL_SIZE * 0.36;
        const pClass = player === 0 ? 'p0' : 'p1';

        const g = document.createElementNS(SVG_NS, 'g');
        let classes = `ww-pawn ww-pawn-${pClass}`;

        const isCurrentTurnPawn = turnRecord && turnRecord.player === player;
        const hasIssue = isCurrentTurnPawn && ['invalid', 'timeout', 'error'].includes(turnRecord.status);

        if (hasIssue && this.options.animations) {
          classes += ' ww-pawn-shake';
        }
        g.setAttribute('class', classes);

        // Pawn circle
        const circle = document.createElementNS(SVG_NS, 'circle');
        circle.setAttribute('cx', cx);
        circle.setAttribute('cy', cy);
        circle.setAttribute('r', radius);
        circle.setAttribute('fill', `url(#${pClass}-gradient)`);
        circle.setAttribute('filter', `url(#pawn-glow-${pClass})`);
        g.appendChild(circle);

        // Highlight ring
        const ring = document.createElementNS(SVG_NS, 'circle');
        ring.setAttribute('cx', cx);
        ring.setAttribute('cy', cy);
        ring.setAttribute('r', radius - 2);
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', 'rgba(255,255,255,0.4)');
        ring.setAttribute('stroke-width', '1.5');
        g.appendChild(ring);

        // Number/label inside pawn
        const text = document.createElementNS(SVG_NS, 'text');
        text.setAttribute('x', cx);
        text.setAttribute('y', cy + 4);
        text.setAttribute('class', 'ww-pawn-label');
        text.textContent = player;
        g.appendChild(text);

        // Status badge if turn failed
        if (hasIssue) {
          const badge = document.createElementNS(SVG_NS, 'g');
          badge.setAttribute('class', 'ww-pawn-badge');
          const badgeBg = document.createElementNS(SVG_NS, 'circle');
          badgeBg.setAttribute('cx', cx + radius - 2);
          badgeBg.setAttribute('cy', cy - radius + 2);
          badgeBg.setAttribute('r', 8);
          badgeBg.setAttribute('fill', turnRecord.status === 'timeout' ? '#f59e0b' : '#ef4444');
          badge.appendChild(badgeBg);

          const badgeIcon = document.createElementNS(SVG_NS, 'text');
          badgeIcon.setAttribute('x', cx + radius - 2);
          badgeIcon.setAttribute('y', cy - radius + 5);
          badgeIcon.setAttribute('class', 'ww-pawn-badge-text');
          badgeIcon.textContent = turnRecord.status === 'timeout' ? '⏱' : '!';
          badge.appendChild(badgeIcon);

          g.appendChild(badge);
        }

        this.pawnsLayer.appendChild(g);
      }
    }
  }

  window.WW.BoardRenderer = BoardRenderer;
})(window);
