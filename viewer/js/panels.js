/**
 * Top bar, player cards, HUD indicators, think-time sparkline, and match summary card.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};

  class PanelsManager {
    constructor(controller, elements) {
      this.controller = controller;
      this.elements = elements;

      this._bindEvents();
    }

    _bindEvents() {
      this.controller.on('load', (data) => {
        this.updateHeader();
        this.updatePlayerCards();
        this.checkSummaryCard();
      });

      this.controller.on('change', () => {
        this.updateHeader();
        this.updatePlayerCards();
        this.checkSummaryCard();
      });

      this.controller.on('playState', (isPlaying) => {
        if (this.elements.playStatusBadge) {
          this.elements.playStatusBadge.textContent = isPlaying ? 'PLAYING' : 'PAUSED';
          this.elements.playStatusBadge.className = isPlaying
            ? 'ww-badge ww-badge-playing'
            : 'ww-badge ww-badge-paused';
        }
      });
    }

    updateHeader() {
      const match = this.controller.matchData;
      if (!match) return;

      const p0Name = match.players[0] ? match.players[0].name : 'Player 0';
      const p1Name = match.players[1] ? match.players[1].name : 'Player 1';

      if (this.elements.matchTitle) {
        this.elements.matchTitle.innerHTML = `
          <span class="ww-title-p0">${p0Name}</span>
          <span class="ww-title-vs">vs</span>
          <span class="ww-title-p1">${p1Name}</span>
        `;
      }

      if (this.elements.turnCounter) {
        const turnNum = this.controller.currentIndex;
        const total = this.controller.totalTurns;
        this.elements.turnCounter.textContent = `Turn ${turnNum} / ${total}`;
      }

      if (this.elements.seedBadge && match.config) {
        this.elements.seedBadge.textContent = `Seed: ${match.config.seed ?? 0}`;
      }
    }

    updatePlayerCards() {
      const match = this.controller.matchData;
      const state = this.controller.currentFrame;
      if (!match || !state) return;

      const currIdx = this.controller.currentIndex;
      const prevIdx = Math.max(0, currIdx - 1);

      const pathLengths = this.controller.pathLengths[currIdx] || [-1, -1];
      const prevPathLengths = this.controller.pathLengths[prevIdx] || pathLengths;
      const cumStats = this.controller.cumulativeStats[currIdx] || [{}, {}];

      for (let p = 0; p < 2; p++) {
        const card = p === 0 ? this.elements.p0Card : this.elements.p1Card;
        if (!card) continue;

        const pName = match.players[p] ? match.players[p].name : `Player ${p}`;
        const isCurrentTurn = state.current_player === p && !state.done;

        if (isCurrentTurn) {
          card.classList.add('ww-card-active');
        } else {
          card.classList.remove('ww-card-active');
        }

        // Name
        const nameEl = card.querySelector('.ww-player-name');
        if (nameEl) nameEl.textContent = pName;

        // Starting player indicator
        const starterEl = card.querySelector('.ww-starter-badge');
        if (starterEl) {
          starterEl.style.display = match.config && match.config.starting_player === p ? 'inline-block' : 'none';
        }

        // Walls remaining (10 pills)
        const wallsLeft = state.walls_remaining ? state.walls_remaining[p] : 10;
        const wallPillsContainer = card.querySelector('.ww-wall-pills');
        if (wallPillsContainer) {
          wallPillsContainer.innerHTML = '';
          for (let w = 0; w < 10; w++) {
            const pill = document.createElement('span');
            pill.className = `ww-wall-pill ${w < wallsLeft ? 'ww-wall-pill-active' : 'ww-wall-pill-spent'}`;
            wallPillsContainer.appendChild(pill);
          }
        }
        const wallCountEl = card.querySelector('.ww-walls-count');
        if (wallCountEl) wallCountEl.textContent = `${wallsLeft} / 10`;

        // Distance to goal + delta
        const dist = pathLengths[p];
        const prevDist = prevPathLengths[p];
        const distEl = card.querySelector('.ww-dist-val');
        if (distEl) distEl.textContent = dist >= 0 ? dist : 'Blocked';

        const deltaEl = card.querySelector('.ww-dist-delta');
        if (deltaEl && currIdx > 0 && dist >= 0 && prevDist >= 0) {
          const delta = dist - prevDist;
          if (delta < 0) {
            deltaEl.className = 'ww-dist-delta ww-delta-better';
            deltaEl.textContent = `${delta}`;
          } else if (delta > 0) {
            deltaEl.className = 'ww-dist-delta ww-delta-worse';
            deltaEl.textContent = `+${delta}`;
          } else {
            deltaEl.className = 'ww-dist-delta ww-delta-same';
            deltaEl.textContent = '0';
          }
        } else if (deltaEl) {
          deltaEl.textContent = '';
        }

        // Cumulative stats counters
        const stats = cumStats[p] || { ok: 0, pass: 0, invalid: 0, timeout: 0, error: 0 };
        const okEl = card.querySelector('.ww-stat-ok');
        if (okEl) okEl.textContent = stats.ok;
        const passEl = card.querySelector('.ww-stat-pass');
        if (passEl) passEl.textContent = stats.pass;
        const issuesEl = card.querySelector('.ww-stat-issues');
        if (issuesEl) issuesEl.textContent = (stats.invalid + stats.timeout + stats.error);

        // Latency and sparkline
        this._renderSparkline(card, p);
      }
    }

    _renderSparkline(card, player) {
      const turns = this.controller.turns;
      const playerTurns = [];
      for (let i = 0; i < turns.length; i++) {
        if (turns[i].player === player && typeof turns[i].elapsed === 'number') {
          playerTurns.push(turns[i].elapsed);
        }
      }

      const sparklineSvg = card.querySelector('.ww-sparkline-svg');
      const avgEl = card.querySelector('.ww-time-avg');
      const maxEl = card.querySelector('.ww-time-max');

      if (!playerTurns.length) {
        if (avgEl) avgEl.textContent = '0ms';
        if (maxEl) maxEl.textContent = '0ms';
        if (sparklineSvg) sparklineSvg.innerHTML = '';
        return;
      }

      const sum = playerTurns.reduce((a, b) => a + b, 0);
      const avg = sum / playerTurns.length;
      const max = Math.max(...playerTurns);

      if (avgEl) avgEl.textContent = `${(avg * 1000).toFixed(1)}ms`;
      if (maxEl) maxEl.textContent = `${(max * 1000).toFixed(1)}ms`;

      if (!sparklineSvg) return;

      const w = 120;
      const h = 26;
      const limit = 1.0; // 1.0s timeout limit line
      const maxY = Math.max(limit * 1.1, max * 1.2, 0.05);

      const points = playerTurns.map((val, idx) => {
        const x = (idx / Math.max(1, playerTurns.length - 1)) * w;
        const y = h - (val / maxY) * h;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

      const limitY = (h - (limit / maxY) * h).toFixed(1);

      sparklineSvg.innerHTML = `
        <line x1="0" y1="${limitY}" x2="${w}" y2="${limitY}" stroke="#ef4444" stroke-dasharray="2,2" stroke-width="1" opacity="0.6"/>
        <polyline points="${points}" fill="none" stroke="${player === 0 ? '#ef4444' : '#3b82f6'}" stroke-width="1.8" stroke-linejoin="round"/>
      `;
    }

    checkSummaryCard() {
      if (!this.elements.summaryModal) return;

      const match = this.controller.matchData;
      if (!match) {
        this.elements.summaryModal.style.display = 'none';
        return;
      }

      // Show summary when at the very end of replay
      const isEnd = this.controller.currentIndex === this.controller.frames.length - 1;
      if (isEnd && match.winner !== undefined) {
        this._renderSummaryContent();
        this.elements.summaryModal.classList.add('ww-modal-open');
      } else {
        this.elements.summaryModal.classList.remove('ww-modal-open');
      }
    }

    _renderSummaryContent() {
      const match = this.controller.matchData;
      const body = this.elements.summaryContent;
      if (!body) return;

      const isDraw = match.winner === null || match.winner === undefined;
      const winnerName = isDraw ? 'Draw Game' : match.players[match.winner].name;
      const winnerClass = isDraw ? '' : `ww-winner-p${match.winner}`;
      const reasonStr = match.reason === 'goal'
        ? 'Reached opposite goal row'
        : `Turn limit reached (${match.total_turns || match.turns.length} turns)`;

      const p0Name = match.players[0].name;
      const p1Name = match.players[1].name;
      const p0Stats = match.stats ? match.stats[0] : { ok: 0, pass: 0, invalid: 0, timeout: 0, error: 0 };
      const p1Stats = match.stats ? match.stats[1] : { ok: 0, pass: 0, invalid: 0, timeout: 0, error: 0 };

      body.innerHTML = `
        <div class="ww-summary-header">
          <div class="ww-summary-trophy">${isDraw ? '🤝' : '🏆'}</div>
          <h2 class="ww-summary-winner ${winnerClass}">
            ${isDraw ? 'Draw' : `Winner: ${winnerName}`}
          </h2>
          <div class="ww-summary-reason">${reasonStr}</div>
        </div>

        <div class="ww-summary-table-wrap">
          <table class="ww-summary-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th class="ww-col-p0">${p0Name}</th>
                <th class="ww-col-p1">${p1Name}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Valid Moves (OK)</td>
                <td>${p0Stats.ok}</td>
                <td>${p1Stats.ok}</td>
              </tr>
              <tr>
                <td>Passed Turns</td>
                <td>${p0Stats.pass}</td>
                <td>${p1Stats.pass}</td>
              </tr>
              <tr>
                <td>Invalid Actions</td>
                <td class="${p0Stats.invalid > 0 ? 'ww-val-error' : ''}">${p0Stats.invalid}</td>
                <td class="${p1Stats.invalid > 0 ? 'ww-val-error' : ''}">${p1Stats.invalid}</td>
              </tr>
              <tr>
                <td>Timeouts (>1.0s)</td>
                <td class="${p0Stats.timeout > 0 ? 'ww-val-warn' : ''}">${p0Stats.timeout}</td>
                <td class="${p1Stats.timeout > 0 ? 'ww-val-warn' : ''}">${p1Stats.timeout}</td>
              </tr>
              <tr>
                <td>Crashes / Errors</td>
                <td class="${p0Stats.error > 0 ? 'ww-val-error' : ''}">${p0Stats.error}</td>
                <td class="${p1Stats.error > 0 ? 'ww-val-error' : ''}">${p1Stats.error}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="ww-summary-actions">
          <button class="ww-btn ww-btn-primary" id="ww-btn-replay-again">↻ Replay from Start</button>
          <button class="ww-btn ww-btn-secondary" id="ww-btn-close-summary">Close</button>
        </div>
      `;

      body.querySelector('#ww-btn-replay-again').addEventListener('click', () => {
        this.elements.summaryModal.classList.remove('ww-modal-open');
        this.controller.seek(0);
        this.controller.play();
      });

      body.querySelector('#ww-btn-close-summary').addEventListener('click', () => {
        this.elements.summaryModal.classList.remove('ww-modal-open');
      });
    }
  }

  window.WW.PanelsManager = PanelsManager;
})(window);
