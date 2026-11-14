/**
 * Interactive Human vs Bot gameplay client and per-bot Win/Loss stats UI.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};

  function escapeHtml(str) {
    return String(str ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  class HumanMatchClient {
    constructor(controller, boardRenderer, options = {}) {
      this.controller = controller;
      this.boardRenderer = boardRenderer;
      this.onFlipChange = options.onFlipChange || null;
      this.onLoadReplayLog = options.onLoadReplayLog || null;

      this.active = false;
      this.busy = false;
      this.mode = 'move'; // 'move' | 'H' | 'V'
      this.session = null;
      this.botsCache = [];
      this._loadingIntoController = false;

      this._bindElements();
      this._bindEvents();
    }

    _bindElements() {
      this.el = {
        btnPlayBot: document.getElementById('ww-btn-play-bot'),
        btnHumanStats: document.getElementById('ww-btn-human-stats'),
        btnEmptyPlay: document.getElementById('ww-btn-empty-play'),
        btnEmptyStats: document.getElementById('ww-btn-empty-human-stats'),

        playModal: document.getElementById('ww-play-bot-modal'),
        humanNameInput: document.getElementById('ww-play-human-name'),
        botSelect: document.getElementById('ww-play-bot-select'),
        botPreview: document.getElementById('ww-play-bot-preview'),
        playError: document.getElementById('ww-play-bot-error'),
        btnCancelPlay: document.getElementById('ww-btn-cancel-play'),
        btnStartPlay: document.getElementById('ww-btn-start-play'),

        liveBar: document.getElementById('ww-live-bar'),
        controlsCard: document.getElementById('ww-controls-card'),
        liveStatus: document.getElementById('ww-live-status'),
        liveBotRecord: document.getElementById('ww-live-bot-record'),
        liveError: document.getElementById('ww-live-error'),
        liveActiveControls: document.getElementById('ww-live-active-controls'),
        livePostActions: document.getElementById('ww-live-post-actions'),
        liveResultBanner: document.getElementById('ww-live-result-banner'),

        btnModeMove: document.getElementById('ww-mode-move'),
        btnModeH: document.getElementById('ww-mode-h'),
        btnModeV: document.getElementById('ww-mode-v'),
        btnFlipWall: document.getElementById('ww-btn-flip-wall'),
        btnResign: document.getElementById('ww-btn-resign'),

        btnRematch: document.getElementById('ww-btn-rematch'),
        btnSwapRematch: document.getElementById('ww-btn-swap-rematch'),
        btnLiveStats: document.getElementById('ww-btn-live-stats'),
        btnExitLive: document.getElementById('ww-btn-exit-live'),

        statsModal: document.getElementById('ww-human-stats-modal'),
        statsSummary: document.getElementById('ww-human-stats-summary'),
        statsTableWrap: document.getElementById('ww-human-stats-table-wrap'),
        historyFilter: document.getElementById('ww-human-history-filter'),
        historyWrap: document.getElementById('ww-human-history-wrap'),
        btnResetStats: document.getElementById('ww-btn-reset-human-stats')
      };
    }

    _bindEvents() {
      if (this.el.btnPlayBot) {
        this.el.btnPlayBot.addEventListener('click', () => this.openPlayModal());
      }
      if (this.el.btnEmptyPlay) {
        this.el.btnEmptyPlay.addEventListener('click', () => this.openPlayModal());
      }
      if (this.el.btnHumanStats) {
        this.el.btnHumanStats.addEventListener('click', () => this.openStatsModal());
      }
      if (this.el.btnEmptyStats) {
        this.el.btnEmptyStats.addEventListener('click', () => this.openStatsModal());
      }
      if (this.el.btnLiveStats) {
        this.el.btnLiveStats.addEventListener('click', () => this.openStatsModal());
      }
      if (this.el.btnCancelPlay) {
        this.el.btnCancelPlay.addEventListener('click', () => this.closePlayModal());
      }
      if (this.el.btnStartPlay) {
        this.el.btnStartPlay.addEventListener('click', () => this.startFromModal());
      }
      if (this.el.botSelect) {
        this.el.botSelect.addEventListener('change', () => this._updateBotPreview());
      }

      if (this.el.btnModeMove) {
        this.el.btnModeMove.addEventListener('click', () => this.setMode('move'));
      }
      if (this.el.btnModeH) {
        this.el.btnModeH.addEventListener('click', () => this.setMode('H'));
      }
      if (this.el.btnModeV) {
        this.el.btnModeV.addEventListener('click', () => this.setMode('V'));
      }
      if (this.el.btnFlipWall) {
        this.el.btnFlipWall.addEventListener('click', () => this.toggleWallOrientation());
      }
      if (this.el.btnResign) {
        this.el.btnResign.addEventListener('click', () => this.resignMatch());
      }
      if (this.el.btnRematch) {
        this.el.btnRematch.addEventListener('click', () => this.rematch(false));
      }
      if (this.el.btnSwapRematch) {
        this.el.btnSwapRematch.addEventListener('click', () => this.rematch(true));
      }
      if (this.el.btnExitLive) {
        this.el.btnExitLive.addEventListener('click', () => this.exitLiveMode());
      }
      if (this.el.historyFilter) {
        this.el.historyFilter.addEventListener('change', () => this._renderStatsModalData(this._lastStatsPayload));
      }
      if (this.el.btnResetStats) {
        this.el.btnResetStats.addEventListener('click', () => this.resetStats());
      }

      // If another match log is loaded externally, exit live mode cleanly
      this.controller.on('load', () => {
        if (!this._loadingIntoController && this.active) {
          this.exitLiveMode(true);
        }
      });

      // Only allow board interaction when viewing the latest live frame
      this.controller.on('change', () => {
        if (this.active) {
          this._syncBoardInteractive();
        }
      });
    }

    isActiveUnfinished() {
      return this.active && this.session && !this.session.done;
    }

    setMode(mode) {
      if (!this.active || !this.session || this.session.done) return;
      if ((mode === 'H' || mode === 'V') && !this._humanHasWalls()) {
        this._showLiveError('No walls remaining — only pawn moves available.');
        mode = 'move';
      } else {
        this._showLiveError('');
      }
      this.mode = mode;
      this._updateModeButtons();
      this._syncBoardInteractive();
    }

    toggleWallOrientation() {
      if (!this.active || !this.session || this.session.done) return;
      if (!this._humanHasWalls()) {
        this._showLiveError('No walls remaining.');
        return;
      }
      this.setMode(this.mode === 'H' ? 'V' : 'H');
    }

    _humanHasWalls() {
      if (!this.session || !this.session.state) return false;
      const w = this.session.state.walls_remaining;
      return w && w[this.session.human_side] > 0;
    }

    async openPlayModal(preselectBot = null) {
      if (!this.el.playModal) return;
      this.el.playModal.classList.add('ww-modal-open');
      if (this.el.playError) this.el.playError.style.display = 'none';

      try {
        const res = await fetch('/api/human/bots');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        this.botsCache = await res.json();
        if (this.el.botSelect) {
          const prev = preselectBot || this.el.botSelect.value || 'optimized_bot';
          this.el.botSelect.innerHTML = '';
          this.botsCache.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item.name;
            const h = item.human_stats;
            const rec = h.games > 0 ? `You ${h.human_wins}W-${h.human_losses}L-${h.draws}D` : '0 games';
            const ai = item.ai_rank ? ` · AI #${item.ai_rank}` : '';
            opt.textContent = `${item.name} (${rec}${ai})`;
            this.el.botSelect.appendChild(opt);
          });
          if (this.botsCache.some(b => b.name === prev)) {
            this.el.botSelect.value = prev;
          }
          this._updateBotPreview();
        }
      } catch (err) {
        if (this.el.playError) {
          this.el.playError.textContent = 'Start server via "python serve.py" to play against bots.';
          this.el.playError.style.display = 'block';
        }
      }
    }

    closePlayModal() {
      if (this.el.playModal) this.el.playModal.classList.remove('ww-modal-open');
    }

    _updateBotPreview() {
      if (!this.el.botSelect || !this.el.botPreview) return;
      const name = this.el.botSelect.value;
      const info = this.botsCache.find(b => b.name === name);
      if (!info) {
        this.el.botPreview.innerHTML = '';
        return;
      }
      const h = info.human_stats;
      const aiText = info.ai_rank
        ? `AI Leaderboard: <strong>#${info.ai_rank}</strong> (${info.ai_score.toFixed(1)} pts, ${info.ai_win_rate.toFixed(1)}% vs bots)`
        : 'AI Leaderboard: Unranked';
      this.el.botPreview.innerHTML = `
        <div>${aiText}</div>
        <div style="margin-top:4px;">
          vs Humans (${h.games} games):
          <strong style="color:var(--p0-primary);">Bot ${h.bot_wins}W</strong> /
          <strong style="color:var(--p1-primary);">You ${h.human_wins}W</strong> /
          <span>${h.draws}D</span>
        </div>
      `;
    }

    async startFromModal() {
      const bot = this.el.botSelect ? this.el.botSelect.value : '';
      if (!bot) return;
      const humanName = this.el.humanNameInput ? (this.el.humanNameInput.value.trim() || 'Human') : 'Human';
      const sideInput = document.querySelector('input[name="ww-human-side"]:checked');
      const humanSide = sideInput ? parseInt(sideInput.value, 10) : 0;

      if (this.el.btnStartPlay) this.el.btnStartPlay.disabled = true;
      if (this.el.playError) this.el.playError.style.display = 'none';

      try {
        await this.startMatch(bot, humanSide, humanName);
        this.closePlayModal();
      } catch (err) {
        if (this.el.playError) {
          this.el.playError.textContent = err.message || 'Failed to start match';
          this.el.playError.style.display = 'block';
        }
      } finally {
        if (this.el.btnStartPlay) this.el.btnStartPlay.disabled = false;
      }
    }

    async startMatch(botName, humanSide = 0, humanName = 'Human') {
      this.busy = true;
      this.mode = 'move';
      const res = await fetch('/api/human/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bot: botName,
          human_side: humanSide,
          human_name: humanName
        })
      });
      const data = await res.json();
      if (!res.ok) {
        this.busy = false;
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      this.active = true;
      this.busy = false;
      this.mode = 'move';
      if (this.onFlipChange) {
        this.onFlipChange(humanSide === 1);
      }
      this._applySessionData(data);
    }

    async rematch(swapSides = false) {
      if (!this.session) return;
      const nextSide = swapSides ? (1 - this.session.human_side) : this.session.human_side;
      try {
        await this.startMatch(this.session.bot_name, nextSide, this.session.human_name);
      } catch (err) {
        this._showLiveError(err.message);
      }
    }

    async submitAction(action) {
      if (!this.active || !this.session || this.session.done || this.busy) return;
      this.busy = true;
      this._showLiveError('');
      this._updateLiveBar();
      this._syncBoardInteractive();

      try {
        const res = await fetch(`/api/human/match/${encodeURIComponent(this.session.session_id)}/move`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(action)
        });
        const data = await res.json();
        if (!res.ok) {
          this._showLiveError(data.error || `Illegal move (HTTP ${res.status})`);
          this.busy = false;
          this._updateLiveBar();
          this._syncBoardInteractive();
          return;
        }
        this.busy = false;
        this.mode = 'move';
        this._applySessionData(data);
      } catch (err) {
        this.busy = false;
        this._showLiveError(err.message || 'Network error sending move');
        this._updateLiveBar();
        this._syncBoardInteractive();
      }
    }

    async resignMatch() {
      if (!this.active || !this.session || this.session.done || this.busy) return;
      this.busy = true;
      this._updateLiveBar();
      this._syncBoardInteractive();
      try {
        const res = await fetch(`/api/human/match/${encodeURIComponent(this.session.session_id)}/resign`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}'
        });
        const data = await res.json();
        this.busy = false;
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        this._applySessionData(data);
      } catch (err) {
        this.busy = false;
        this._showLiveError(err.message || 'Error resigning');
        this._updateLiveBar();
        this._syncBoardInteractive();
      }
    }

    exitLiveMode(keepControlsShown = false) {
      this.active = false;
      this.busy = false;
      this.boardRenderer.setInteractive(null);
      if (this.el.liveBar) this.el.liveBar.style.display = 'none';
      if (this.el.controlsCard && (!keepControlsShown || this.controller.matchData)) {
        this.el.controlsCard.style.display = 'flex';
      }
    }

    _applySessionData(data) {
      this.session = data;
      const replay = data.replay_data;
      if (replay) {
        replay._suppressAutoSummary = true;
        this._loadingIntoController = true;
        try {
          const targetIdx = (replay.turns || []).length;
          this.controller.loadMatch(replay, false, targetIdx);
        } finally {
          this._loadingIntoController = false;
        }
      }
      this._updateModeButtons();
      this._updateLiveBar();
      this._syncBoardInteractive();
    }

    _updateModeButtons() {
      const hasWalls = this._humanHasWalls();
      const map = {
        move: this.el.btnModeMove,
        H: this.el.btnModeH,
        V: this.el.btnModeV
      };
      Object.entries(map).forEach(([k, btn]) => {
        if (!btn) return;
        btn.classList.toggle('ww-mode-active', this.mode === k);
        if (k === 'H' || k === 'V') btn.disabled = !hasWalls || this.busy;
        else btn.disabled = this.busy;
      });
      if (this.el.btnFlipWall) this.el.btnFlipWall.disabled = !hasWalls || this.busy;
      if (this.el.btnResign) this.el.btnResign.disabled = this.busy;
    }

    _updateLiveBar() {
      if (!this.el.liveBar || !this.session) return;
      this.el.liveBar.style.display = 'flex';
      if (this.el.controlsCard) {
        this.el.controlsCard.style.display = this.session.done ? 'flex' : 'none';
      }

      const br = this.session.bot_record;
      if (this.el.liveBotRecord && br) {
        this.el.liveBotRecord.textContent =
          `${br.bot_name} vs Humans: Bot ${br.bot_wins}W · You ${br.human_wins}W · ${br.draws}D`;
      }

      if (!this.session.done) {
        if (this.el.liveActiveControls) this.el.liveActiveControls.style.display = 'flex';
        if (this.el.livePostActions) this.el.livePostActions.style.display = 'none';
        const humanTurn = this.session.state.current_player === this.session.human_side;
        if (this.el.liveStatus) {
          if (this.busy) {
            this.el.liveStatus.textContent = `⏳ ${this.session.bot_name} is thinking...`;
            this.el.liveStatus.className = 'ww-live-status ww-live-thinking';
          } else if (humanTurn) {
            const modeLabel = this.mode === 'move' ? 'Select move' : `Place ${this.mode} wall`;
            this.el.liveStatus.textContent = `🟢 Your Turn (${modeLabel})`;
            this.el.liveStatus.className = 'ww-live-status ww-live-your-turn';
          } else {
            this.el.liveStatus.textContent = `⏳ ${this.session.bot_name}'s Turn`;
            this.el.liveStatus.className = 'ww-live-status ww-live-thinking';
          }
        }
      } else {
        if (this.el.liveActiveControls) this.el.liveActiveControls.style.display = 'none';
        if (this.el.livePostActions) this.el.livePostActions.style.display = 'flex';
        const w = this.session.winner;
        let cls = 'ww-result-draw';
        let txt = '🤝 Draw Match';
        if (w === this.session.human_side) {
          cls = 'ww-result-win';
          txt = `🏆 You defeated ${this.session.bot_name}! (${this.session.reason || 'goal'})`;
        } else if (w === this.session.bot_side) {
          cls = 'ww-result-loss';
          txt = `🤖 ${this.session.bot_name} wins! (${this.session.reason || 'goal'})`;
        }
        if (this.el.liveStatus) {
          this.el.liveStatus.textContent = '🏁 Match Recorded';
          this.el.liveStatus.className = 'ww-live-status ww-live-done';
        }
        if (this.el.liveResultBanner) {
          this.el.liveResultBanner.className = `ww-live-result-banner ${cls}`;
          this.el.liveResultBanner.textContent = txt;
        }
      }
    }

    _syncBoardInteractive() {
      if (!this.active || !this.session || this.session.done) {
        this.boardRenderer.setInteractive(null);
        return;
      }
      const atLatestFrame = this.controller.currentIndex === this.controller.frames.length - 1;
      const isHumanTurn = this.session.state.current_player === this.session.human_side;
      this.boardRenderer.setInteractive({
        enabled: atLatestFrame && isHumanTurn && !this.busy,
        busy: this.busy,
        humanSide: this.session.human_side,
        mode: this.mode,
        legalMoves: this.session.legal_moves || [],
        legalWalls: this.session.legal_walls || [],
        onMove: (toCell) => this.submitAction({ type: 'move', to: toCell }),
        onWall: (wallAct) => this.submitAction({
          type: 'wall',
          orientation: wallAct.orientation,
          row: wallAct.row,
          col: wallAct.col
        }),
        onToggleOrientation: () => this.toggleWallOrientation()
      });
    }

    _showLiveError(msg) {
      if (!this.el.liveError) return;
      if (msg) {
        this.el.liveError.textContent = msg;
        this.el.liveError.style.display = 'inline-block';
      } else {
        this.el.liveError.textContent = '';
        this.el.liveError.style.display = 'none';
      }
    }

    async openStatsModal() {
      if (!this.el.statsModal) return;
      this.el.statsModal.classList.add('ww-modal-open');
      if (this.el.statsSummary) this.el.statsSummary.innerHTML = 'Loading records...';
      try {
        const res = await fetch('/api/human/stats');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        this._lastStatsPayload = data;
        this._populateHistoryFilter(data.bots || []);
        this._renderStatsModalData(data);
      } catch (err) {
        if (this.el.statsSummary) {
          this.el.statsSummary.innerHTML = '<div class="ww-logs-empty">Start server via <code>python serve.py</code> to view live Human vs Bot records.</div>';
        }
        if (this.el.statsTableWrap) this.el.statsTableWrap.innerHTML = '';
        if (this.el.historyWrap) this.el.historyWrap.innerHTML = '';
      }
    }

    _populateHistoryFilter(bots) {
      if (!this.el.historyFilter) return;
      const prev = this.el.historyFilter.value;
      this.el.historyFilter.innerHTML = '<option value="">All Bots</option>';
      bots.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b.bot_name;
        opt.textContent = b.bot_name;
        this.el.historyFilter.appendChild(opt);
      });
      if (bots.some(b => b.bot_name === prev)) {
        this.el.historyFilter.value = prev;
      }
    }

    _renderStatsModalData(data) {
      if (!data) return;
      const bots = data.bots || [];
      const recent = data.recent_matches || [];

      const totGames = bots.reduce((a, b) => a + b.games, 0);
      const totBotWins = bots.reduce((a, b) => a + b.bot_wins, 0);
      const totHumanWins = bots.reduce((a, b) => a + b.human_wins, 0);
      const totDraws = bots.reduce((a, b) => a + b.draws, 0);

      if (this.el.statsSummary) {
        this.el.statsSummary.innerHTML = `
          <div class="ww-stat-pill"><span>Matches Played</span><strong>${totGames}</strong></div>
          <div class="ww-stat-pill"><span>You Won</span><strong style="color:var(--p1-primary);">${totHumanWins}</strong></div>
          <div class="ww-stat-pill"><span>Bots Won</span><strong style="color:var(--p0-primary);">${totBotWins}</strong></div>
          <div class="ww-stat-pill"><span>Draws</span><strong>${totDraws}</strong></div>
        `;
      }

      if (this.el.statsTableWrap) {
        let html = `
          <table class="ww-summary-table ww-leaderboard-table">
            <thead>
              <tr>
                <th>Bot</th>
                <th style="text-align:right;">Games</th>
                <th style="text-align:right;">Bot Wins</th>
                <th style="text-align:right;">Bot Losses (You W)</th>
                <th style="text-align:right;">Draws</th>
                <th style="text-align:right;">Bot Win%</th>
                <th style="text-align:right;">You Win%</th>
                <th style="text-align:right;">Action</th>
              </tr>
            </thead>
            <tbody>
        `;
        bots.forEach(b => {
          html += `
            <tr>
              <td style="font-weight:600;">${escapeHtml(b.bot_name)}</td>
              <td style="text-align:right;">${b.games}</td>
              <td style="text-align:right; color:var(--p0-primary); font-weight:600;">${b.bot_wins}</td>
              <td style="text-align:right; color:var(--p1-primary); font-weight:600;">${b.bot_losses}</td>
              <td style="text-align:right;">${b.draws}</td>
              <td style="text-align:right;">${b.games > 0 ? b.bot_win_rate.toFixed(1) + '%' : '—'}</td>
              <td style="text-align:right;">${b.games > 0 ? b.human_win_rate.toFixed(1) + '%' : '—'}</td>
              <td style="text-align:right;">
                <button class="ww-btn ww-btn-sm ww-btn-quick-play" data-bot="${escapeHtml(b.bot_name)}">⚔ Play</button>
              </td>
            </tr>
          `;
        });
        html += '</tbody></table>';
        this.el.statsTableWrap.innerHTML = html;

        this.el.statsTableWrap.querySelectorAll('.ww-btn-quick-play').forEach(btn => {
          btn.addEventListener('click', () => {
            const botName = btn.getAttribute('data-bot');
            this.el.statsModal.classList.remove('ww-modal-open');
            this.openPlayModal(botName);
          });
        });
      }

      if (this.el.historyWrap) {
        const filterBot = this.el.historyFilter ? this.el.historyFilter.value : '';
        const filtered = filterBot ? recent.filter(m => m.bot_name === filterBot) : recent;
        if (!filtered.length) {
          this.el.historyWrap.innerHTML = '<div class="ww-logs-empty">No Human vs Bot matches recorded yet.</div>';
          return;
        }
        let html = `
          <table class="ww-summary-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Player</th>
                <th>Bot</th>
                <th>Side</th>
                <th>Result (You / Bot)</th>
                <th style="text-align:right;">Turns</th>
                <th>Reason</th>
                <th style="text-align:right;">Replay</th>
              </tr>
            </thead>
            <tbody>
        `;
        filtered.forEach(m => {
          const sideTxt = m.human_side === 0 ? 'P0 (Red)' : 'P1 (Blue)';
          const badge = m.outcome_for_human === 'win'
            ? '<span class="ww-outcome-pill ww-outcome-win">YOU WON / BOT LOST</span>'
            : (m.outcome_for_human === 'loss'
              ? '<span class="ww-outcome-pill ww-outcome-loss">BOT WON / YOU LOST</span>'
              : '<span class="ww-outcome-pill ww-outcome-draw">DRAW</span>');
          const replayBtn = m.log_path
            ? `<button class="ww-btn ww-btn-sm ww-btn-load-human-replay" data-log="${escapeHtml(m.log_path)}">📂 View</button>`
            : '—';
          html += `
            <tr>
              <td style="font-size:0.75rem; color:var(--text-muted);">${escapeHtml((m.played_at || '').slice(0, 19))}</td>
              <td>${escapeHtml(m.human_name)}</td>
              <td style="font-weight:600;">${escapeHtml(m.bot_name)}</td>
              <td>${sideTxt}</td>
              <td>${badge}</td>
              <td style="text-align:right;">${m.turns}</td>
              <td>${escapeHtml(m.reason || '')}</td>
              <td style="text-align:right;">${replayBtn}</td>
            </tr>
          `;
        });
        html += '</tbody></table>';
        this.el.historyWrap.innerHTML = html;

        this.el.historyWrap.querySelectorAll('.ww-btn-load-human-replay').forEach(btn => {
          btn.addEventListener('click', () => {
            const logPath = btn.getAttribute('data-log');
            this.el.statsModal.classList.remove('ww-modal-open');
            this.exitLiveMode();
            if (this.onLoadReplayLog) {
              this.onLoadReplayLog(logPath);
            }
          });
        });
      }
    }

    async resetStats() {
      if (!window.confirm('Clear all recorded Human vs Bot win/loss match history?')) {
        return;
      }
      const res = await fetch('/api/human/stats/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
      });
      if (res.ok) {
        await this.openStatsModal();
      }
    }
  }

  window.WW.HumanMatchClient = HumanMatchClient;
})(window);
