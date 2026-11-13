/**
 * Main application coordinator, event wiring, and keyboard shortcuts.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};
  const { BoardRenderer, ReplayController, TimelineManager, PanelsManager, LogLoader, ExportManager } = window.WW;

  document.addEventListener('DOMContentLoaded', () => {
    // 1. Theme initialization
    const savedTheme = localStorage.getItem('ww_viewer_dark');
    if (savedTheme === '1' || (!savedTheme && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    }

    const themeToggleBtn = document.getElementById('ww-btn-theme');
    if (themeToggleBtn) {
      themeToggleBtn.addEventListener('click', () => {
        const isDark = document.documentElement.classList.toggle('dark');
        localStorage.setItem('ww_viewer_dark', isDark ? '1' : '0');
      });
    }

    // 2. Instantiate core components
    const boardContainer = document.getElementById('ww-board-container');
    const boardRenderer = new BoardRenderer(boardContainer);
    const controller = new ReplayController();
    const exportManager = new ExportManager(boardRenderer, controller);

    // 3. Setup timeline manager
    const timelineManager = new TimelineManager(controller, {
      scrubber: document.getElementById('ww-scrubber'),
      ticksContainer: document.getElementById('ww-scrubber-ticks'),
      moveList: document.getElementById('ww-move-list'),
      filterChips: document.querySelectorAll('.ww-filter-chip'),
      tooltip: document.getElementById('ww-scrubber-tooltip')
    });

    // 4. Setup panels manager
    const panelsManager = new PanelsManager(controller, {
      matchTitle: document.getElementById('ww-match-title'),
      turnCounter: document.getElementById('ww-turn-counter'),
      playStatusBadge: document.getElementById('ww-play-status-badge'),
      seedBadge: document.getElementById('ww-seed-badge'),
      p0Card: document.getElementById('ww-player-card-0'),
      p1Card: document.getElementById('ww-player-card-1'),
      summaryModal: document.getElementById('ww-summary-modal'),
      summaryContent: document.getElementById('ww-summary-modal-content')
    });

    // 5. Setup log loader
    const loader = new LogLoader(controller, {
      fileInput: document.getElementById('ww-file-input'),
      logsBtn: document.getElementById('ww-btn-logs'),
      logsModal: document.getElementById('ww-logs-modal'),
      logsList: document.getElementById('ww-logs-list'),
      emptyLogsList: document.getElementById('ww-empty-logs-list'),
      logsStatus: document.getElementById('ww-logs-status'),
      logsSearchInput: document.getElementById('ww-logs-search'),
      logsSortSelect: document.getElementById('ww-logs-sort'),
      logsRefreshBtn: document.getElementById('ww-btn-refresh-logs'),
      tournamentModal: document.getElementById('ww-tournament-modal'),
      tournamentContent: document.getElementById('ww-tournament-modal-content'),
      seriesBar: document.getElementById('ww-series-bar'),
      prevMatchBtn: document.getElementById('ww-btn-prev-match'),
      nextMatchBtn: document.getElementById('ww-btn-next-match')
    });

    // 6. Connect Board to Controller events
    controller.on('load', () => {
      boardRenderer.setWallOwners(controller.wallOwners);
      document.getElementById('ww-empty-state').style.display = 'none';
      document.getElementById('ww-main-view').style.display = 'grid';
    });

    controller.on('change', (currIdx, prevIdx, state, turnRecord) => {
      const prevState = controller.previousFrame;
      boardRenderer.render(state, prevState, turnRecord);
    });

    // 7. Replay playback controls buttons
    const playBtn = document.getElementById('ww-btn-play');
    const prevBtn = document.getElementById('ww-btn-prev');
    const nextBtn = document.getElementById('ww-btn-next');
    const firstBtn = document.getElementById('ww-btn-first');
    const lastBtn = document.getElementById('ww-btn-last');
    const speedSelect = document.getElementById('ww-select-speed');
    const loopToggle = document.getElementById('ww-toggle-loop');
    const snapshotBtn = document.getElementById('ww-btn-snapshot');

    if (playBtn) {
      playBtn.addEventListener('click', () => controller.togglePlay());
      controller.on('playState', (isPlaying) => {
        playBtn.innerHTML = isPlaying ? '⏸' : '▶';
        playBtn.title = isPlaying ? 'Pause (Space)' : 'Play (Space)';
      });
    }

    if (prevBtn) prevBtn.addEventListener('click', () => controller.prev());
    if (nextBtn) nextBtn.addEventListener('click', () => controller.next());
    if (firstBtn) firstBtn.addEventListener('click', () => controller.first());
    if (lastBtn) lastBtn.addEventListener('click', () => controller.last());

    if (speedSelect) {
      speedSelect.addEventListener('change', (e) => {
        controller.setSpeed(parseFloat(e.target.value));
      });
    }

    if (loopToggle) {
      loopToggle.addEventListener('change', (e) => {
        controller.setLoop(e.target.checked);
      });
    }

    if (snapshotBtn) {
      snapshotBtn.addEventListener('click', () => exportManager.exportPNG());
    }

    // 8. View Options & Dropdowns
    const viewOptions = {
      showCoords: document.getElementById('ww-opt-coords'),
      showPath: document.getElementById('ww-opt-path'),
      showHeatmap: document.getElementById('ww-opt-heatmap'),
      flipBoard: document.getElementById('ww-opt-flip'),
      colorWalls: document.getElementById('ww-opt-wall-color')
    };

    function applyViewOptions() {
      boardRenderer.setOptions({
        showCoords: viewOptions.showCoords ? viewOptions.showCoords.checked : true,
        showPath: viewOptions.showPath ? viewOptions.showPath.value : 'both',
        showHeatmap: viewOptions.showHeatmap ? viewOptions.showHeatmap.value : 'none',
        flipBoard: viewOptions.flipBoard ? viewOptions.flipBoard.checked : false,
        colorWallsByOwner: viewOptions.colorWalls ? viewOptions.colorWalls.checked : true
      });
    }

    Object.values(viewOptions).forEach(el => {
      if (el) el.addEventListener('change', applyViewOptions);
    });

    // View Dropdown open/close
    const viewMenuBtn = document.getElementById('ww-btn-view-menu');
    const viewDropdown = document.getElementById('ww-view-dropdown');
    if (viewMenuBtn && viewDropdown) {
      viewMenuBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        viewDropdown.classList.toggle('ww-dropdown-open');
      });
      document.addEventListener('click', () => {
        viewDropdown.classList.remove('ww-dropdown-open');
      });
      viewDropdown.addEventListener('click', (e) => e.stopPropagation());
    }

    // Modal Close buttons
    document.querySelectorAll('.ww-modal-close').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.closest('.ww-modal').classList.remove('ww-modal-open');
      });
    });

    // Leaderboard modal button & fetch
    const lbBtn = document.getElementById('ww-btn-leaderboard');
    const lbModal = document.getElementById('ww-leaderboard-modal');
    const lbWrap = document.getElementById('ww-leaderboard-table-wrap');
    const lbStatus = document.getElementById('ww-leaderboard-status');

    function loadLeaderboard() {
      if (!lbWrap) return;
      if (lbStatus) lbStatus.textContent = 'Fetching current rankings...';
      fetch('/api/leaderboard')
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(data => {
          if (!data || !data.length) {
            if (lbStatus) lbStatus.textContent = 'No agents ranked yet in ranking.db.';
            lbWrap.innerHTML = '<div class="ww-logs-empty">Play matches with arena.py or series.py to establish ratings.</div>';
            return;
          }
          if (lbStatus) lbStatus.textContent = `${data.length} agents ranked. Leaderboard updates automatically after every match.`;
          let html = `
            <table class="ww-summary-table ww-leaderboard-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Agent</th>
                  <th style="text-align:right;">Score (μ - 3σ)</th>
                  <th style="text-align:right;">Skill μ</th>
                  <th style="text-align:right;">Uncertainty σ</th>
                  <th style="text-align:center;">W / D / L</th>
                  <th style="text-align:right;">Win %</th>
                  <th style="text-align:right;">Avg Turns</th>
                </tr>
              </thead>
              <tbody>
          `;
          data.forEach(entry => {
            const rankBadge = entry.rank === 1 ? '🥇 #1' : (entry.rank === 2 ? '🥈 #2' : (entry.rank === 3 ? '🥉 #3' : `#${entry.rank}`));
            html += `
              <tr>
                <td style="font-weight:700;">${rankBadge}</td>
                <td style="font-weight:600;">${entry.name}</td>
                <td style="text-align:right; font-weight:700; color:var(--accent);">${entry.score.toFixed(2)}</td>
                <td style="text-align:right; color:var(--text-muted);">${entry.mu.toFixed(2)}</td>
                <td style="text-align:right; color:var(--text-subtle);">${entry.sigma.toFixed(2)}</td>
                <td style="text-align:center;">${entry.wins} / ${entry.draws} / ${entry.losses}</td>
                <td style="text-align:right;">${entry.win_rate.toFixed(1)}%</td>
                <td style="text-align:right; color:var(--text-muted);">${entry.avg_turns.toFixed(1)}</td>
              </tr>
            `;
          });
          html += '</tbody></table>';
          lbWrap.innerHTML = html;
        })
        .catch(err => {
          if (lbStatus) lbStatus.textContent = 'API unavailable (offline or file:// mode).';
          lbWrap.innerHTML = '<div class="ww-logs-empty">Start server via <code>python serve.py</code> to view live rankings.</div>';
        });
    }

    if (lbBtn && lbModal) {
      lbBtn.addEventListener('click', () => {
        lbModal.classList.add('ww-modal-open');
        loadLeaderboard();
      });
    }

    // Shortcuts modal button
    const helpBtn = document.getElementById('ww-btn-help');
    const helpModal = document.getElementById('ww-shortcuts-modal');
    if (helpBtn && helpModal) {
      helpBtn.addEventListener('click', () => {
        helpModal.classList.add('ww-modal-open');
      });
    }

    // Open file buttons (open logs modal instead of OS file explorer)
    const openFileBtns = document.querySelectorAll('.ww-btn-open-file');
    openFileBtns.forEach(b => {
      b.addEventListener('click', () => loader.openLogsModal());
    });

    // 9. Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      // Ignore if typing in input
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          controller.togglePlay();
          break;
        case 'a':
        case 'A':
        case 'ArrowLeft':
          e.preventDefault();
          controller.prev();
          break;
        case 'd':
        case 'D':
        case 'ArrowRight':
          e.preventDefault();
          controller.next();
          break;
        case 'q':
        case 'Q':
        case 'Home':
          e.preventDefault();
          controller.first();
          break;
        case 'e':
        case 'E':
        case 'End':
          e.preventDefault();
          controller.last();
          break;
        case 'w':
        case 'W':
          e.preventDefault();
          loader.prevMatch();
          break;
        case 's':
        case 'S':
          e.preventDefault();
          loader.nextMatch();
          break;
        case '+':
        case '=':
          e.preventDefault();
          if (speedSelect) {
            const nextIdx = Math.min(speedSelect.selectedIndex + 1, speedSelect.options.length - 1);
            speedSelect.selectedIndex = nextIdx;
            controller.setSpeed(parseFloat(speedSelect.value));
          }
          break;
        case '-':
        case '_':
          e.preventDefault();
          if (speedSelect) {
            const prevIdx = Math.max(0, speedSelect.selectedIndex - 1);
            speedSelect.selectedIndex = prevIdx;
            controller.setSpeed(parseFloat(speedSelect.value));
          }
          break;
        case 'p':
        case 'P':
          e.preventDefault();
          if (viewOptions.showPath) {
            const cycle = { 'both': 'p0', 'p0': 'p1', 'p1': 'none', 'none': 'both' };
            viewOptions.showPath.value = cycle[viewOptions.showPath.value] || 'both';
            applyViewOptions();
          }
          break;
        case 'f':
        case 'F':
          e.preventDefault();
          if (viewOptions.flipBoard) {
            viewOptions.flipBoard.checked = !viewOptions.flipBoard.checked;
            applyViewOptions();
          }
          break;
        case 't':
        case 'T':
          e.preventDefault();
          if (themeToggleBtn) themeToggleBtn.click();
          break;
        case '?':
          e.preventDefault();
          if (helpModal) helpModal.classList.toggle('ww-modal-open');
          break;
        case 'Escape':
          document.querySelectorAll('.ww-modal-open').forEach(m => m.classList.remove('ww-modal-open'));
          if (viewDropdown) viewDropdown.classList.remove('ww-dropdown-open');
          break;
      }
    });

    // 10. Check Initial URL for parameters
    loader.checkInitialUrl();
  });
})(window);
