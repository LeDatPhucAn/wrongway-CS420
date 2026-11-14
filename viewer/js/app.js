/**
 * Main application coordinator, event wiring, and keyboard shortcuts.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};
  const { BoardRenderer, ReplayController, TimelineManager, PanelsManager, LogLoader, ExportManager, HumanMatchClient } = window.WW;

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

    // Human vs Bot Match Client
    const humanClient = HumanMatchClient
      ? new HumanMatchClient(controller, boardRenderer, {
          onFlipChange: (flipped) => {
            if (viewOptions.flipBoard) {
              viewOptions.flipBoard.checked = !!flipped;
              applyViewOptions();
            }
          },
          onLoadReplayLog: (logPath) => loader.fetchAndLoadLog(logPath)
        })
      : null;

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

    // Leaderboard modal button & fetch (with Bot-vs-Bot per-opponent W/D/L drill-down)
    const lbBtn = document.getElementById('ww-btn-leaderboard');
    const lbModal = document.getElementById('ww-leaderboard-modal');
    const lbWrap = document.getElementById('ww-leaderboard-table-wrap');
    const lbStatus = document.getElementById('ww-leaderboard-status');

    function escapeHtml(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function openSeriesReplay(relPath) {
      if (!relPath) return;
      if (lbModal) lbModal.classList.remove('ww-modal-open');
      loader.fetchAndLoadLog(relPath);
    }

    function renderResultPill(resLabel) {
      if (resLabel === 'WIN') return '<span class="ww-outcome-pill ww-res-win">WIN</span>';
      if (resLabel === 'LOSS') return '<span class="ww-outcome-pill ww-res-loss">LOSS</span>';
      return '<span class="ww-outcome-pill ww-res-draw">DRAW</span>';
    }

    function formatSigned(val) {
      const n = Number(val || 0);
      return `${n >= 0 ? '+' : ''}${n.toFixed(2)}`;
    }

    function loadAgentDetail(agentName, filterOpponent = '') {
      if (!lbWrap) return;
      if (lbStatus) {
        lbStatus.textContent = filterOpponent
          ? `Loading ${agentName} vs ${filterOpponent} matches...`
          : `Loading bot-vs-bot record for ${agentName}...`;
      }
      const query = filterOpponent
        ? `?opponent=${encodeURIComponent(filterOpponent)}&limit=200`
        : '?limit=200';
      fetch(`/api/leaderboard/${encodeURIComponent(agentName)}${query}`)
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(data => {
          const ag = data.agent || {};
          const h2h = data.head_to_head || [];
          const matches = data.recent_matches || [];
          const wins = Number(ag.wins || 0);
          const draws = Number(ag.draws || 0);
          const losses = Number(ag.losses || 0);
          const nGames = Number(ag.n_games || (wins + draws + losses));
          const mu = Number(ag.mu || 25);
          const sigma = Number(ag.sigma || (25 / 3));
          const ratingScore = ag.score != null ? Number(ag.score) : (mu - 3 * sigma);
          const winRate = nGames > 0 ? (wins / nGames * 100).toFixed(1) : '0.0';
          const avgTurns = nGames > 0 ? (Number(ag.total_turns || 0) / nGames).toFixed(1) : '0.0';

          if (lbStatus) {
            lbStatus.textContent = `${agentName}: ${wins}W / ${draws}D / ${losses}L across ${h2h.length} bot opponent${h2h.length === 1 ? '' : 's'}.`;
          }

          let html = `
            <div class="ww-agent-detail-header">
              <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                <button class="ww-btn ww-btn-xs" id="ww-lb-back-btn">← All Bots</button>
                <strong style="font-size:1.05rem;">🤖 ${escapeHtml(agentName)}</strong>
              </div>
              <span style="font-size:0.78rem; color:var(--text-muted);">Bot vs Bot Tournament &amp; Arena Matches</span>
            </div>
            <div class="ww-agent-stat-grid">
              <div class="ww-stat-pill"><span>TrueSkill (μ-3σ)</span><strong style="color:var(--accent);">${ratingScore.toFixed(2)}</strong></div>
              <div class="ww-stat-pill"><span>Skill μ / σ</span><strong>${mu.toFixed(1)} <small style="font-weight:400; color:var(--text-muted);">±${sigma.toFixed(2)}</small></strong></div>
              <div class="ww-stat-pill"><span>Record (W/D/L)</span><strong>${wins} / ${draws} / ${losses}</strong></div>
              <div class="ww-stat-pill"><span>Win Rate</span><strong>${winRate}%</strong></div>
              <div class="ww-stat-pill"><span>Avg Turns</span><strong>${avgTurns}</strong></div>
            </div>

            <div class="ww-human-history-header" style="margin-top:8px;">
              <strong>Head-to-Head vs Other Bots</strong>
              <span style="font-size:0.74rem; color:var(--text-muted);">Click an opponent to filter matches</span>
            </div>
          `;

          if (!h2h.length) {
            html += '<div class="ww-logs-empty">No bot-vs-bot matches recorded yet for this agent.</div>';
          } else {
            html += `
              <table class="ww-summary-table ww-leaderboard-table">
                <thead>
                  <tr>
                    <th>Opponent Bot</th>
                    <th style="text-align:center;">GP</th>
                    <th style="text-align:center;">W / D / L</th>
                    <th style="text-align:right;">Win %</th>
                    <th style="text-align:center;">As P0</th>
                    <th style="text-align:center;">As P1</th>
                    <th style="text-align:right;">Avg Turns</th>
                    <th style="text-align:right;">Actions</th>
                  </tr>
                </thead>
                <tbody>
            `;
            h2h.forEach(row => {
              const isSelected = filterOpponent === row.opponent;
              const replayBtn = row.log_rel_path
                ? `<button class="ww-btn ww-btn-xs ww-lb-replay-btn" data-log="${escapeHtml(row.log_rel_path)}" title="Open series replay">📂 Replay</button>`
                : '';
              html += `
                <tr${isSelected ? ' style="background:rgba(59,130,246,0.08);"' : ''}>
                  <td>
                    <button class="ww-lb-row-btn ww-lb-filter-btn" data-opp="${escapeHtml(row.opponent)}">${escapeHtml(row.opponent)}</button>
                  </td>
                  <td style="text-align:center;">${row.games}</td>
                  <td style="text-align:center; font-weight:600;">
                    <span style="color:#10b981;">${row.wins}</span> /
                    <span style="color:var(--text-muted);">${row.draws}</span> /
                    <span style="color:#ef4444;">${row.losses}</span>
                  </td>
                  <td style="text-align:right; font-weight:600;">${Number(row.win_rate || 0).toFixed(1)}%</td>
                  <td style="text-align:center; color:var(--text-muted);">${row.p0_wins}-${row.p0_losses}</td>
                  <td style="text-align:center; color:var(--text-muted);">${row.p1_wins}-${row.p1_losses}</td>
                  <td style="text-align:right; color:var(--text-muted);">${Number(row.avg_turns || 0).toFixed(1)}</td>
                  <td style="text-align:right; white-space:nowrap;">
                    <button class="ww-btn ww-btn-xs ww-lb-filter-btn" data-opp="${escapeHtml(row.opponent)}">${isSelected ? 'Filtered' : 'Matches'}</button>
                    ${replayBtn}
                  </td>
                </tr>
              `;
            });
            html += '</tbody></table>';
          }

          html += `
            <div class="ww-human-history-header">
              <strong>Recorded Bot-vs-Bot Matches (${matches.length}${filterOpponent ? ` vs ${escapeHtml(filterOpponent)}` : ''})</strong>
              <div style="display:flex; align-items:center; gap:6px;">
                <select id="ww-lb-opp-filter" class="ww-select" style="padding:3px 8px; font-size:0.76rem;">
                  <option value="">All Opponents</option>
                  ${h2h.map(r => `<option value="${escapeHtml(r.opponent)}"${filterOpponent === r.opponent ? ' selected' : ''}>vs ${escapeHtml(r.opponent)} (${r.wins}W-${r.losses}L)</option>`).join('')}
                </select>
                ${filterOpponent ? '<button class="ww-btn ww-btn-xs" id="ww-lb-clear-filter">Show All</button>' : ''}
              </div>
            </div>
          `;

          if (!matches.length) {
            html += '<div class="ww-logs-empty">No individual matches recorded yet.</div>';
          } else {
            html += `
              <div style="max-height:260px; overflow-y:auto; border:1px solid var(--border); border-radius:8px;">
                <table class="ww-summary-table ww-leaderboard-table">
                  <thead>
                    <tr>
                      <th>Result</th>
                      <th>Side</th>
                      <th>Opponent</th>
                      <th style="text-align:right;">Turns</th>
                      <th style="text-align:right;">Score Δ</th>
                      <th>Date</th>
                      <th style="text-align:right;">Replay</th>
                    </tr>
                  </thead>
                  <tbody>
            `;
            matches.forEach(m => {
              const sideBadge = m.bot_side === 0
                ? '<span style="color:var(--p0-primary); font-weight:700;">P0</span>'
                : '<span style="color:var(--p1-primary); font-weight:700;">P1</span>';
              const delta = Number(m.score_delta || 0);
              const deltaColor = delta > 0 ? '#10b981' : (delta < 0 ? '#ef4444' : 'var(--text-muted)');
              const ts = m.timestamp ? String(m.timestamp).replace('T', ' ').slice(0, 16) : '-';
              const replayBtn = m.log_rel_path
                ? `<button class="ww-btn ww-btn-xs ww-lb-replay-btn" data-log="${escapeHtml(m.log_rel_path)}">📂 Series</button>`
                : '<span style="color:var(--text-subtle); font-size:0.72rem;">—</span>';
              html += `
                <tr>
                  <td>${renderResultPill(m.result)}</td>
                  <td>${sideBadge}</td>
                  <td style="font-weight:600;">${escapeHtml(m.opponent)}</td>
                  <td style="text-align:right;">${m.turns}</td>
                  <td style="text-align:right; color:${deltaColor}; font-weight:600;">${formatSigned(delta)}</td>
                  <td style="color:var(--text-muted); font-size:0.74rem;">${escapeHtml(ts)}</td>
                  <td style="text-align:right;">${replayBtn}</td>
                </tr>
              `;
            });
            html += '</tbody></table></div>';
          }

          lbWrap.innerHTML = html;

          const backBtn = document.getElementById('ww-lb-back-btn');
          if (backBtn) backBtn.addEventListener('click', () => loadLeaderboard());

          const clearBtn = document.getElementById('ww-lb-clear-filter');
          if (clearBtn) clearBtn.addEventListener('click', () => loadAgentDetail(agentName, ''));

          const oppSelect = document.getElementById('ww-lb-opp-filter');
          if (oppSelect) {
            oppSelect.addEventListener('change', () => {
              loadAgentDetail(agentName, oppSelect.value);
            });
          }

          lbWrap.querySelectorAll('.ww-lb-filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
              const opp = btn.getAttribute('data-opp') || '';
              loadAgentDetail(agentName, filterOpponent === opp ? '' : opp);
            });
          });

          lbWrap.querySelectorAll('.ww-lb-replay-btn').forEach(btn => {
            btn.addEventListener('click', () => {
              openSeriesReplay(btn.getAttribute('data-log'));
            });
          });
        })
        .catch(() => {
          if (lbStatus) lbStatus.textContent = `Could not load bot-vs-bot history for ${agentName}.`;
        });
    }

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
          if (lbStatus) lbStatus.textContent = `${data.length} agents ranked. Click any bot to inspect its W/D/L records against every other bot.`;
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
                  <th style="text-align:right;">vs Bots</th>
                </tr>
              </thead>
              <tbody>
          `;
          data.forEach(entry => {
            const rankBadge = entry.rank === 1 ? '🥇 #1' : (entry.rank === 2 ? '🥈 #2' : (entry.rank === 3 ? '🥉 #3' : `#${entry.rank}`));
            html += `
              <tr>
                <td style="font-weight:700;">${rankBadge}</td>
                <td>
                  <button class="ww-lb-row-btn ww-lb-agent-btn" data-agent="${escapeHtml(entry.name)}">${escapeHtml(entry.name)}</button>
                </td>
                <td style="text-align:right; font-weight:700; color:var(--accent);">${entry.score.toFixed(2)}</td>
                <td style="text-align:right; color:var(--text-muted);">${entry.mu.toFixed(2)}</td>
                <td style="text-align:right; color:var(--text-subtle);">${entry.sigma.toFixed(2)}</td>
                <td style="text-align:center;">${entry.wins} / ${entry.draws} / ${entry.losses}</td>
                <td style="text-align:right;">${entry.win_rate.toFixed(1)}%</td>
                <td style="text-align:right; color:var(--text-muted);">${entry.avg_turns.toFixed(1)}</td>
                <td style="text-align:right;">
                  <button class="ww-btn ww-btn-xs ww-lb-agent-btn" data-agent="${escapeHtml(entry.name)}">Details</button>
                </td>
              </tr>
            `;
          });
          html += '</tbody></table>';
          lbWrap.innerHTML = html;

          lbWrap.querySelectorAll('.ww-lb-agent-btn').forEach(btn => {
            btn.addEventListener('click', () => {
              const name = btn.getAttribute('data-agent');
              if (name) loadAgentDetail(name);
            });
          });
        })
        .catch(() => {
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

      if (humanClient && humanClient.isActiveUnfinished()) {
        if (e.key === '1') {
          e.preventDefault();
          humanClient.setMode('move');
          return;
        }
        if (e.key === '2') {
          e.preventDefault();
          humanClient.setMode('H');
          return;
        }
        if (e.key === '3') {
          e.preventDefault();
          humanClient.setMode('V');
          return;
        }
        if (e.key === 'r' || e.key === 'R') {
          e.preventDefault();
          humanClient.toggleWallOrientation();
          return;
        }
      }

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
