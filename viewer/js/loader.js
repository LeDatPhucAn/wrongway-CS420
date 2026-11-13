/**
 * Log loader, file drag-and-drop, server log browser, and playlist / series management.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};

  class LogLoader {
    constructor(controller, elements) {
      this.controller = controller;
      this.elements = elements;

      this.playlist = []; // Array of matchData objects
      this.currentMatchIndex = 0;
      this.serverLogs = [];
      this.recentLogs = this._loadRecentLogs();

      this._bindEvents();
    }

    _bindEvents() {
      // File input picker
      if (this.elements.fileInput) {
        this.elements.fileInput.addEventListener('change', (e) => {
          const files = Array.from(e.target.files);
          if (files.length > 0) {
            this.loadFiles(files);
          }
          e.target.value = '';
        });
      }

      // Drag and drop on window
      window.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        document.body.classList.add('ww-drag-active');
      });

      window.addEventListener('dragleave', (e) => {
        if (!e.relatedTarget || e.relatedTarget === document.documentElement) {
          document.body.classList.remove('ww-drag-active');
        }
      });

      window.addEventListener('drop', (e) => {
        e.preventDefault();
        document.body.classList.remove('ww-drag-active');
        const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.json'));
        if (files.length > 0) {
          this.loadFiles(files);
        }
      });

      // Server logs browser button & refresh
      if (this.elements.logsBtn) {
        this.elements.logsBtn.addEventListener('click', () => {
          this.openLogsModal();
        });
      }

      if (this.elements.logsRefreshBtn) {
        this.elements.logsRefreshBtn.addEventListener('click', () => {
          this.fetchServerLogs();
        });
      }

      if (this.elements.logsSearchInput) {
        this.elements.logsSearchInput.addEventListener('input', () => {
          this.renderServerLogsList();
        });
      }

      if (this.elements.logsSortSelect) {
        this.elements.logsSortSelect.addEventListener('change', () => {
          this.renderServerLogsList();
        });
      }

      // Series match navigation buttons
      if (this.elements.prevMatchBtn) {
        this.elements.prevMatchBtn.addEventListener('click', () => this.prevMatch());
      }
      if (this.elements.nextMatchBtn) {
        this.elements.nextMatchBtn.addEventListener('click', () => this.nextMatch());
      }

      // Controller change updates URL hash
      this.controller.on('change', (idx) => {
        this._updateHash();
      });
    }

    _loadRecentLogs() {
      try {
        const stored = localStorage.getItem('ww_recent_logs');
        return stored ? JSON.parse(stored) : [];
      } catch (err) {
        return [];
      }
    }

    _saveRecentLog(name) {
      if (!name) return;
      try {
        this.recentLogs = [name, ...this.recentLogs.filter(n => n !== name)].slice(0, 10);
        localStorage.setItem('ww_recent_logs', JSON.stringify(this.recentLogs));
      } catch (err) {}
    }

    _updateHash() {
      const turn = this.controller.currentIndex;
      const matchIdx = this.currentMatchIndex;
      const params = new URLSearchParams();
      if (turn > 0) params.set('turn', turn);
      if (this.playlist.length > 1) params.set('match', matchIdx + 1);
      const str = params.toString();
      window.location.hash = str ? `#${str}` : '';
    }

    _escapeHtml(text) {
      return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    _formatLogEntry(log) {
      const p1 = this._escapeHtml(log.player1 || 'P0');
      const p2 = this._escapeHtml(log.player2 || 'P1');
      const name = this._escapeHtml(log.name || '');
      const count = log.total_turns || 0;

      if (log.type === 'tournament') {
        return {
          icon: '🏅',
          title: `${p1} <span class="ww-text-muted">·</span> ${p2}`,
          name,
          countLabel: `${count} matches`,
          badge: log.winner
            ? `<span class="ww-log-winner">Leader: ${this._escapeHtml(log.winner)}</span>`
            : `<span class="ww-log-draw">Tournament</span>`,
        };
      }
      if (log.type === 'series') {
        return {
          icon: '📚',
          title: `${p1} <span class="ww-text-muted">vs</span> ${p2}`,
          name,
          countLabel: `${count} matches`,
          badge: `<span class="ww-log-winner">Series</span>`,
        };
      }
      return {
        icon: '📄',
        title: `${p1} <span class="ww-text-muted">vs</span> ${p2}`,
        name,
        countLabel: `${count} turns`,
        badge: log.winner
          ? `<span class="ww-log-winner">Won: ${this._escapeHtml(log.winner)}</span>`
          : `<span class="ww-log-draw">Draw</span>`,
      };
    }

    _findSeriesLog(sourceName, agentA, agentB) {
      const idx = (sourceName || '').lastIndexOf('/');
      const baseDir = idx >= 0 ? sourceName.slice(0, idx + 1) : '';
      const candidates = [
        `${baseDir}series/${agentA}_vs_${agentB}.json`,
        `${baseDir}series/${agentB}_vs_${agentA}.json`,
      ];
      for (const cand of candidates) {
        const match = this.serverLogs.find(l => l.name === cand);
        if (match) return match.name;
      }
      const suffixA = `/${agentA}_vs_${agentB}.json`;
      const suffixB = `/${agentB}_vs_${agentA}.json`;
      const fallback = this.serverLogs.find(
        l => l.name === `${agentA}_vs_${agentB}.json` ||
             l.name === `${agentB}_vs_${agentA}.json` ||
             l.name.endsWith(suffixA) ||
             l.name.endsWith(suffixB)
      );
      return fallback ? fallback.name : null;
    }

    openTournamentModal(data, sourceName = '') {
      const modal = this.elements.tournamentModal;
      const body = this.elements.tournamentContent;
      if (!modal || !body) {
        alert('Tournament summary view is unavailable.');
        return;
      }

      const cfg = data.config || {};
      const side = data.side_wins || {};
      const standings = data.standings || [];
      const h2h = data.head_to_head || {};
      const total = cfg.total_matches || (data.matches ? data.matches.length : 0);
      const p0Wins = side.P0 || 0;
      const p1Wins = side.P1 || 0;
      const draws = side.draw || 0;
      const pct = n => (total > 0 ? ((n / total) * 100).toFixed(1) : '0.0');

      const hasTrueSkill = standings.some(r => typeof r.trueskill_score === 'number');

      let html = `
        <div class="ww-tourney-meta">
          <span><strong>${standings.length}</strong> bots</span>
          <span><strong>${total}</strong> matches (${this._escapeHtml(cfg.games_per_pair ?? '?')} per pair)</span>
          <span>P0 wins: <strong>${p0Wins}</strong> (${pct(p0Wins)}%)</span>
          <span>P1 wins: <strong>${p1Wins}</strong> (${pct(p1Wins)}%)</span>
          <span>Draws: <strong>${draws}</strong> (${pct(draws)}%)</span>
          <span>Elapsed: <strong>${this._escapeHtml(data.elapsed_seconds ?? '?')}s</strong></span>
        </div>
        <div class="ww-tourney-section-title">Standings (3 pts win / 1 pt draw / 0 pts loss)</div>
        <div class="ww-tourney-scroll">
          <table class="ww-tourney-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Bot</th>
                <th>GP</th>
                <th>W/D/L</th>
                <th>Pts</th>
                <th>Win%</th>
                <th>P0 (W-L)</th>
                <th>P1 (W-L)</th>
                <th>AvgT</th>
                <th>TO/Err</th>
                ${hasTrueSkill ? '<th>TrueSkill</th>' : ''}
              </tr>
            </thead>
            <tbody>
      `;

      standings.forEach(row => {
        const badge = row.rank === 1 ? '🥇 #1' : (row.rank === 2 ? '🥈 #2' : (row.rank === 3 ? '🥉 #3' : `#${row.rank}`));
        const issues = (row.timeouts || 0) + (row.errors || 0) + (row.invalids || 0);
        html += `
          <tr>
            <td><strong>${badge}</strong></td>
            <td><strong>${this._escapeHtml(row.name)}</strong></td>
            <td>${row.games ?? 0}</td>
            <td>${row.wins ?? 0}/${row.draws ?? 0}/${row.losses ?? 0}</td>
            <td><strong>${row.points ?? 0}</strong></td>
            <td>${Number(row.win_rate ?? 0).toFixed(1)}%</td>
            <td>${row.p0_wins ?? 0}-${row.p0_losses ?? 0}</td>
            <td>${row.p1_wins ?? 0}-${row.p1_losses ?? 0}</td>
            <td>${Number(row.avg_turns ?? 0).toFixed(1)}</td>
            <td class="${issues > 0 ? 'ww-val-error' : ''}">${row.timeouts || 0}/${(row.errors || 0) + (row.invalids || 0)}</td>
            ${hasTrueSkill ? `<td><strong>${Number(row.trueskill_score ?? 0).toFixed(2)}</strong></td>` : ''}
          </tr>
        `;
      });

      html += `</tbody></table></div>`;

      const ordered = standings.map(r => r.name);
      if (ordered.length > 1 && Object.keys(h2h).length > 0) {
        html += `
          <div class="ww-tourney-section-title">Head-to-Head Matrix (click any matchup to open its series replay)</div>
          <div class="ww-tourney-scroll">
            <table class="ww-tourney-table ww-h2h-table">
              <thead>
                <tr>
                  <th>Bot</th>
                  ${ordered.map((_n, idx) => `<th title="${this._escapeHtml(ordered[idx])}">#${idx + 1}</th>`).join('')}
                </tr>
              </thead>
              <tbody>
        `;

        ordered.forEach((a, idx) => {
          html += `<tr><td>#${idx + 1} ${this._escapeHtml(a)}</td>`;
          ordered.forEach(b => {
            if (a === b) {
              html += `<td class="ww-text-muted">--</td>`;
            } else {
              const rec = (h2h[a] && h2h[a][b]) || { wins: 0, losses: 0, draws: 0 };
              const txt = rec.draws ? `${rec.wins}-${rec.losses}-${rec.draws}` : `${rec.wins}-${rec.losses}`;
              const seriesLog = this._findSeriesLog(sourceName, a, b);
              if (seriesLog) {
                html += `<td><button class="ww-h2h-btn" data-series-log="${this._escapeHtml(seriesLog)}" title="Watch ${this._escapeHtml(a)} vs ${this._escapeHtml(b)}">${txt}</button></td>`;
              } else {
                html += `<td>${txt}</td>`;
              }
            }
          });
          html += `</tr>`;
        });

        html += `</tbody></table></div>`;
      }

      body.innerHTML = html;
      body.querySelectorAll('.ww-h2h-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const target = btn.getAttribute('data-series-log');
          if (target) {
            modal.classList.remove('ww-modal-open');
            this.fetchAndLoadLog(target);
          }
        });
      });

      modal.classList.add('ww-modal-open');
    }

    loadFiles(files) {
      if (!files || !files.length) return;

      const readPromises = files.map(file => {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (e) => {
            try {
              const data = JSON.parse(e.target.result);
              data._fileName = file.name;
              resolve(data);
            } catch (err) {
              reject(new Error(`Failed to parse ${file.name}: ${err.message}`));
            }
          };
          reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
          reader.readAsText(file);
        });
      });

      Promise.all(readPromises).then(results => {
        const matches = [];
        let tournamentData = null;
        for (let i = 0; i < results.length; i++) {
          const item = results[i];
          if (item.type === 'tournament' && Array.isArray(item.standings)) {
            tournamentData = item;
          } else if (item.type === 'series' && Array.isArray(item.matches)) {
            item.matches.forEach((m, idx) => {
              m._fileName = `${item._fileName} #${idx + 1}`;
              matches.push(m);
            });
          } else if (item.initial_state && item.turns) {
            matches.push(item);
          }
        }

        if (matches.length > 0) {
          this.setPlaylist(matches, 0);
          this._saveRecentLog(matches[0]._fileName || 'Custom File');
        } else if (tournamentData) {
          this.openTournamentModal(tournamentData, tournamentData._fileName || 'summary.json');
          this._saveRecentLog(tournamentData._fileName || 'summary.json');
        } else {
          alert('No valid Wrongway match data found in selected file(s).');
        }
      }).catch(err => {
        alert(err.message);
      });
    }

    loadJsonData(data, name = 'Match') {
      if (data.type === 'tournament' && Array.isArray(data.standings)) {
        this.openTournamentModal(data, name);
      } else if (data.type === 'series' && Array.isArray(data.matches)) {
        this.setPlaylist(data.matches, 0, data.summary);
      } else if (data.initial_state && data.turns) {
        data._fileName = name;
        this.setPlaylist([data], 0);
      } else {
        alert('File does not match Wrongway replay format.');
      }
      this._saveRecentLog(name);
    }

    setPlaylist(matches, startMatchIndex = 0, seriesSummary = null) {
      this.playlist = matches;
      this.currentMatchIndex = Math.max(0, Math.min(startMatchIndex, matches.length - 1));

      this._updateSeriesUI(seriesSummary);
      this._loadCurrentMatchFromPlaylist();
    }

    _loadCurrentMatchFromPlaylist() {
      const match = this.playlist[this.currentMatchIndex];
      if (!match) return;

      // Check URL hash for start turn
      let startTurn = 0;
      if (window.location.hash) {
        const hashParams = new URLSearchParams(window.location.hash.slice(1));
        const t = parseInt(hashParams.get('turn'), 10);
        if (!isNaN(t)) startTurn = t;
      }

      this.controller.loadMatch(match, false, startTurn);
      this._updateSeriesUI();
    }

    nextMatch() {
      if (this.currentMatchIndex < this.playlist.length - 1) {
        this.currentMatchIndex++;
        this._loadCurrentMatchFromPlaylist();
      }
    }

    prevMatch() {
      if (this.currentMatchIndex > 0) {
        this.currentMatchIndex--;
        this._loadCurrentMatchFromPlaylist();
      }
    }

    _updateSeriesUI(summary = null) {
      const bar = this.elements.seriesBar;
      if (!bar) return;

      if (this.playlist.length > 1) {
        bar.style.display = 'flex';
        const label = bar.querySelector('.ww-series-label');
        if (label) {
          label.textContent = `Match ${this.currentMatchIndex + 1} of ${this.playlist.length}`;
        }
        if (this.elements.prevMatchBtn) {
          this.elements.prevMatchBtn.disabled = this.currentMatchIndex === 0;
        }
        if (this.elements.nextMatchBtn) {
          this.elements.nextMatchBtn.disabled = this.currentMatchIndex === this.playlist.length - 1;
        }
      } else {
        bar.style.display = 'none';
      }
    }

    openLogsModal() {
      if (!this.elements.logsModal) return;
      this.elements.logsModal.classList.add('ww-modal-open');
      this.fetchServerLogs();
    }

    closeLogsModal() {
      if (!this.elements.logsModal) return;
      this.elements.logsModal.classList.remove('ww-modal-open');
    }

    fetchServerLogs() {
      const statusEl = this.elements.logsStatus;
      if (statusEl) statusEl.textContent = 'Scanning logs directory...';

      fetch('/api/logs')
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(logs => {
          this.serverLogs = logs || [];
          if (statusEl) statusEl.textContent = `${this.serverLogs.length} logs found`;
          this.renderServerLogsList();
        })
        .catch(err => {
          if (statusEl) {
            statusEl.innerHTML = `<span class="ww-text-muted">Server API unavailable (running offline or via file://). Use <strong>Open File</strong> or drag & drop.</span>`;
          }
          this.renderRecentLogs();
        });
    }

    renderRecentLogs() {
      const container = this.elements.logsList;
      if (!container) return;
      if (!this.recentLogs.length) return;

      container.innerHTML = `
        <div class="ww-logs-section-title">Recently Viewed</div>
        <div class="ww-recent-logs-list">
          ${this.recentLogs.map(name => `<div class="ww-recent-log-item">📄 ${this._escapeHtml(name)}</div>`).join('')}
        </div>
      `;
    }

    renderServerLogsList() {
      const container = this.elements.logsList;
      if (!container) return;

      const query = (this.elements.logsSearchInput ? this.elements.logsSearchInput.value : '').toLowerCase().trim();
      const sortBy = this.elements.logsSortSelect ? this.elements.logsSortSelect.value : 'mtime';

      let filtered = this.serverLogs.filter(log => {
        if (!query) return true;
        const nameMatch = log.name && log.name.toLowerCase().includes(query);
        const typeMatch = log.type && log.type.toLowerCase().includes(query);
        const p1Match = log.player1 && log.player1.toLowerCase().includes(query);
        const p2Match = log.player2 && log.player2.toLowerCase().includes(query);
        const winnerMatch = log.winner && log.winner.toLowerCase().includes(query);
        return nameMatch || typeMatch || p1Match || p2Match || winnerMatch;
      });

      // Sort
      filtered.sort((a, b) => {
        if (sortBy === 'mtime') return (b.mtime || 0) - (a.mtime || 0);
        if (sortBy === 'turns') return (b.total_turns || 0) - (a.total_turns || 0);
        if (sortBy === 'name') return (a.name || '').localeCompare(b.name || '');
        return 0;
      });

      if (!filtered.length) {
        container.innerHTML = `<div class="ww-logs-empty">No logs match your filter.</div>`;
        return;
      }

      container.innerHTML = '';
      const fragment = document.createDocumentFragment();

      filtered.forEach(log => {
        const item = document.createElement('div');
        item.className = 'ww-log-item';

        const dateStr = log.mtime ? new Date(log.mtime * 1000).toLocaleString() : '';
        const formatted = this._formatLogEntry(log);

        item.innerHTML = `
          <div class="ww-log-info">
            <div class="ww-log-title">${formatted.icon} ${formatted.title}</div>
            <div class="ww-log-file">${formatted.name}</div>
          </div>
          <div class="ww-log-meta">
            ${formatted.badge}
            <span class="ww-log-turns">${formatted.countLabel}</span>
            <span class="ww-log-date">${dateStr}</span>
          </div>
        `;

        item.addEventListener('click', () => {
          this.fetchAndLoadLog(log.name);
          this.closeLogsModal();
        });

        fragment.appendChild(item);
      });

      container.appendChild(fragment);

      // Also populate the welcome / empty state logs list if visible
      if (this.elements.emptyLogsList) {
        this.elements.emptyLogsList.innerHTML = '';
        if (!this.serverLogs.length) {
          this.elements.emptyLogsList.innerHTML = '<div class="ww-text-muted">No logs found in logs/ directory.</div>';
        } else {
          this.serverLogs.forEach(log => {
            const card = document.createElement('div');
            card.className = 'ww-empty-log-card';
            const formatted = this._formatLogEntry(log);
            card.innerHTML = `
              <div class="ww-empty-log-icon">${formatted.icon}</div>
              <div class="ww-empty-log-info">
                <div class="ww-empty-log-title">${formatted.title}</div>
                <div class="ww-empty-log-sub">${formatted.name} · ${formatted.countLabel}</div>
              </div>
              <div>${formatted.badge}</div>
            `;
            card.addEventListener('click', () => {
              this.fetchAndLoadLog(log.name);
            });
            this.elements.emptyLogsList.appendChild(card);
          });
        }
      }
    }

    fetchAndLoadLog(filename) {
      fetch(`/api/logs/${encodeURIComponent(filename)}`)
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(data => {
          this.loadJsonData(data, filename);
        })
        .catch(err => {
          alert(`Failed to load log ${filename}: ${err.message}`);
        });
    }

    checkInitialUrl() {
      // Auto-fetch available logs from logs/ directory
      this.fetchServerLogs();

      // Check query params if served: ?file=demo.json&paused=1&speed=2
      const params = new URLSearchParams(window.location.search);
      const file = params.get('file');
      const speed = parseFloat(params.get('speed'));
      const paused = params.get('paused') === '1' || params.get('paused') === 'true';

      if (!isNaN(speed) && speed > 0) {
        this.controller.setSpeed(speed);
      }

      if (file) {
        const cleanName = file.replace(/^logs\//, '');
        fetch(`/api/logs/${encodeURIComponent(cleanName)}`)
          .then(res => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
          })
          .then(data => {
            this.loadJsonData(data, cleanName);
            if (!paused && data.type !== 'tournament') {
              this.controller.play();
            }
          })
          .catch(() => {
            // If API fetch fails (e.g. running from file://), attempt to load demo log or show file picker
          });
      }
    }
  }

  window.WW.LogLoader = LogLoader;
})(window);
