/**
 * Interactive scrubber, event ticks, and turn history move list.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};

  class TimelineManager {
    constructor(controller, elements) {
      this.controller = controller;
      this.scrubber = elements.scrubber;
      this.ticksContainer = elements.ticksContainer;
      this.moveList = elements.moveList;
      this.filterChips = elements.filterChips;
      this.tooltip = elements.tooltip;

      this.currentFilter = 'all'; // 'all', 'walls', 'issues'
      this._bindEvents();
    }

    _bindEvents() {
      // Scrubber input
      if (this.scrubber) {
        this.scrubber.addEventListener('input', (e) => {
          this.controller.pause();
          this.controller.seek(parseInt(e.target.value, 10));
        });
      }

      // Filter chips
      if (this.filterChips) {
        this.filterChips.forEach(chip => {
          chip.addEventListener('click', (e) => {
            const filter = chip.getAttribute('data-filter');
            this.setFilter(filter);
          });
        });
      }

      // Controller listeners
      this.controller.on('load', (data) => {
        this.renderTicks();
        this.renderMoveList();
        this.updateScrubber();
      });

      this.controller.on('change', (currIdx) => {
        this.updateScrubber();
        this.highlightActiveMove(currIdx);
      });
    }

    setFilter(filter) {
      this.currentFilter = filter;
      if (this.filterChips) {
        this.filterChips.forEach(chip => {
          if (chip.getAttribute('data-filter') === filter) {
            chip.classList.add('ww-chip-active');
          } else {
            chip.classList.remove('ww-chip-active');
          }
        });
      }
      this.renderMoveList();
    }

    updateScrubber() {
      if (!this.scrubber) return;
      this.scrubber.max = Math.max(0, this.controller.totalFrames - 1);
      this.scrubber.value = this.controller.currentIndex;

      const progress = this.scrubber.max > 0 ? (this.controller.currentIndex / this.scrubber.max) * 100 : 0;
      this.scrubber.style.setProperty('--scrubber-progress', `${progress}%`);
    }

    renderTicks() {
      if (!this.ticksContainer) return;
      this.ticksContainer.innerHTML = '';
      const totalTurns = this.controller.totalTurns;
      if (totalTurns === 0) return;

      const turns = this.controller.turns;
      const fragment = document.createDocumentFragment();

      for (let i = 0; i < turns.length; i++) {
        const turn = turns[i];
        let tickClass = null;

        if (turn.status === 'invalid' || turn.status === 'error') {
          tickClass = 'ww-tick-error';
        } else if (turn.status === 'timeout') {
          tickClass = 'ww-tick-timeout';
        } else if (turn.action && turn.action.type === 'wall') {
          tickClass = turn.player === 0 ? 'ww-tick-wall-p0' : 'ww-tick-wall-p1';
        } else if (turn.status === 'pass') {
          tickClass = 'ww-tick-pass';
        }

        if (tickClass) {
          const tick = document.createElement('div');
          const leftPercent = ((i + 1) / totalTurns) * 100;
          tick.className = `ww-scrubber-tick ${tickClass}`;
          tick.style.left = `${leftPercent}%`;
          tick.setAttribute('data-turn', i + 1);

          tick.addEventListener('mouseenter', (e) => this._showTooltip(e, turn, i + 1));
          tick.addEventListener('mouseleave', () => this._hideTooltip());
          tick.addEventListener('click', () => {
            this.controller.pause();
            this.controller.seek(i + 1);
          });

          fragment.appendChild(tick);
        }
      }

      this.ticksContainer.appendChild(fragment);
    }

    _showTooltip(e, turn, turnNum) {
      if (!this.tooltip) return;
      const pName = this.controller.matchData.players[turn.player].name;
      let actionStr = 'Pass';
      if (turn.action) {
        if (turn.action.type === 'move') actionStr = `Move → [${turn.action.to.join(',')}]`;
        if (turn.action.type === 'wall') actionStr = `Wall ${turn.action.orientation} (${turn.action.row},${turn.action.col})`;
      }

      this.tooltip.innerHTML = `
        <div class="ww-tooltip-header">Turn ${turnNum} · ${pName}</div>
        <div class="ww-tooltip-body">${actionStr} · <span class="ww-status-${turn.status}">${turn.status.toUpperCase()}</span></div>
      `;

      const rect = e.target.getBoundingClientRect();
      this.tooltip.style.left = `${rect.left + rect.width / 2}px`;
      this.tooltip.style.top = `${rect.top - 40}px`;
      this.tooltip.classList.add('ww-tooltip-visible');
    }

    _hideTooltip() {
      if (!this.tooltip) return;
      this.tooltip.classList.remove('ww-tooltip-visible');
    }

    renderMoveList() {
      if (!this.moveList) return;
      this.moveList.innerHTML = '';
      const turns = this.controller.turns;
      const players = this.controller.matchData ? this.controller.matchData.players : [];

      const fragment = document.createDocumentFragment();

      for (let i = 0; i < turns.length; i++) {
        const turn = turns[i];
        const isWall = turn.action && turn.action.type === 'wall';
        const isIssue = ['invalid', 'timeout', 'error'].includes(turn.status);

        if (this.currentFilter === 'walls' && !isWall) continue;
        if (this.currentFilter === 'issues' && !isIssue) continue;

        const row = document.createElement('div');
        row.className = 'ww-move-row';
        row.setAttribute('data-frame-index', i + 1);

        const pName = players[turn.player] ? players[turn.player].name : `P${turn.player}`;
        let actionStr = 'Pass';
        let actionIcon = '⏸';
        if (turn.action) {
          if (turn.action.type === 'move') {
            actionStr = `Move [${turn.action.to.join(',')}]`;
            actionIcon = '♟';
          } else if (turn.action.type === 'wall') {
            actionStr = `Wall ${turn.action.orientation} (${turn.action.row}, ${turn.action.col})`;
            actionIcon = '🧱';
          }
        }

        const elapsedMs = typeof turn.elapsed === 'number' ? `${(turn.elapsed * 1000).toFixed(1)}ms` : '';

        row.innerHTML = `
          <div class="ww-move-num">${i + 1}</div>
          <div class="ww-move-player ww-player-dot-${turn.player}" title="${pName}"></div>
          <div class="ww-move-action">
            <span class="ww-action-icon">${actionIcon}</span>
            <span class="ww-action-desc">${actionStr}</span>
          </div>
          <div class="ww-move-time">${elapsedMs}</div>
          <div class="ww-move-status ww-status-badge ww-status-${turn.status}">${turn.status}</div>
        `;

        row.addEventListener('click', () => {
          this.controller.pause();
          this.controller.seek(i + 1);
        });

        fragment.appendChild(row);
      }

      this.moveList.appendChild(fragment);
      this.highlightActiveMove(this.controller.currentIndex);
    }

    highlightActiveMove(frameIndex) {
      if (!this.moveList) return;
      const prevActive = this.moveList.querySelector('.ww-move-row-active');
      if (prevActive) {
        prevActive.classList.remove('ww-move-row-active');
      }

      if (frameIndex > 0) {
        const currentActive = this.moveList.querySelector(`[data-frame-index="${frameIndex}"]`);
        if (currentActive) {
          currentActive.classList.add('ww-move-row-active');
          // Smooth scroll inside list container
          currentActive.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
      }
    }
  }

  window.WW.TimelineManager = TimelineManager;
})(window);
