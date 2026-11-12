/**
 * ReplayController: manages playback state, frame navigation, wall owners, and statistics.
 */
(function(window) {
  'use strict';

  window.WW = window.WW || {};
  const { rules } = window.WW;

  class ReplayController {
    constructor() {
      this.matchData = null;
      this.frames = [];
      this.turns = [];
      this.currentIndex = 0;
      this.playing = false;
      this.speed = 1.0;
      this.loop = false;

      this.wallOwners = new Map(); // "H:r:c" -> player id
      this.cumulativeStats = []; // stats[frameIdx][player] = { ok, pass, invalid, timeout, error }
      this.pathLengths = []; // pathLengths[frameIdx] = [p0Dist, p1Dist]
      this.timings = [[], []]; // timings[player] = array of floats

      this._listeners = new Map();
      this._lastTick = 0;
      this._rafId = null;

      this._tick = this._tick.bind(this);
    }

    on(event, callback) {
      if (!this._listeners.has(event)) {
        this._listeners.set(event, []);
      }
      this._listeners.get(event).push(callback);
      return () => {
        const arr = this._listeners.get(event);
        if (arr) {
          const idx = arr.indexOf(callback);
          if (idx !== -1) arr.splice(idx, 1);
        }
      };
    }

    _emit(event, ...args) {
      const arr = this._listeners.get(event);
      if (arr) {
        for (let i = 0; i < arr.length; i++) {
          arr[i](...args);
        }
      }
    }

    loadMatch(matchData, autoPlay = false, startIndex = 0) {
      this.pause();
      this.matchData = matchData;
      this.frames = [];
      this.turns = matchData.turns || [];
      this.wallOwners.clear();
      this.cumulativeStats = [];
      this.pathLengths = [];
      this.timings = [[], []];

      if (!matchData.initial_state) {
        throw new Error('Invalid match data: initial_state missing.');
      }

      // Build frames list:
      // Frame 0 is initial_state
      this.frames.push(matchData.initial_state);

      // Track wall placements across all turns
      for (let t = 0; t < this.turns.length; t++) {
        const turn = this.turns[t];
        this.frames.push(turn.state);

        if (turn.status === 'ok' && turn.action && turn.action.type === 'wall') {
          const key = `${turn.action.orientation}:${turn.action.row}:${turn.action.col}`;
          this.wallOwners.set(key, turn.player);
        }

        if (typeof turn.elapsed === 'number') {
          this.timings[turn.player].push(turn.elapsed);
        }
      }

      // Precompute cumulative stats per frame
      const runningStats = [
        { ok: 0, pass: 0, invalid: 0, timeout: 0, error: 0 },
        { ok: 0, pass: 0, invalid: 0, timeout: 0, error: 0 }
      ];
      this.cumulativeStats.push(JSON.parse(JSON.stringify(runningStats)));

      for (let t = 0; t < this.turns.length; t++) {
        const turn = this.turns[t];
        const status = turn.status;
        if (runningStats[turn.player][status] !== undefined) {
          runningStats[turn.player][status]++;
        }
        this.cumulativeStats.push(JSON.parse(JSON.stringify(runningStats)));
      }

      // Precompute path lengths per frame
      for (let f = 0; f < this.frames.length; f++) {
        const st = this.frames[f];
        const p0 = st.positions ? st.positions[0] : [8, 4];
        const p1 = st.positions ? st.positions[1] : [0, 4];
        const d0Map = rules.computeDistanceMap(0, st.walls.horizontal, st.walls.vertical);
        const d1Map = rules.computeDistanceMap(8, st.walls.horizontal, st.walls.vertical);
        this.pathLengths.push([
          d0Map[p0[0]][p0[1]],
          d1Map[p1[0]][p1[1]]
        ]);
      }

      this.currentIndex = Math.max(0, Math.min(startIndex, this.frames.length - 1));

      this._emit('load', matchData);
      this._emit('change', this.currentIndex, null, this.currentFrame, this.currentTurn);

      if (autoPlay) {
        this.play();
      }
    }

    get currentFrame() {
      return this.frames[this.currentIndex] || null;
    }

    get previousFrame() {
      return this.currentIndex > 0 ? this.frames[this.currentIndex - 1] : null;
    }

    get currentTurn() {
      return this.currentIndex > 0 ? this.turns[this.currentIndex - 1] : null;
    }

    get totalFrames() {
      return this.frames.length;
    }

    get totalTurns() {
      return this.turns.length;
    }

    seek(index) {
      const prev = this.currentIndex;
      const target = Math.max(0, Math.min(index, this.frames.length - 1));
      if (prev !== target) {
        this.currentIndex = target;
        this._emit('change', this.currentIndex, prev, this.currentFrame, this.currentTurn);
      }
    }

    next() {
      if (this.currentIndex < this.frames.length - 1) {
        this.seek(this.currentIndex + 1);
      } else if (this.loop) {
        this.seek(0);
      } else {
        this.pause();
      }
    }

    prev() {
      if (this.currentIndex > 0) {
        this.seek(this.currentIndex - 1);
      }
    }

    first() {
      this.seek(0);
    }

    last() {
      this.seek(this.frames.length - 1);
    }

    setSpeed(speed) {
      this.speed = Math.max(0.25, Math.min(speed, 16.0));
      this._emit('speedChange', this.speed);
    }

    setLoop(loop) {
      this.loop = !!loop;
      this._emit('loopChange', this.loop);
    }

    play() {
      if (this.playing) return;
      if (this.currentIndex >= this.frames.length - 1) {
        this.seek(0);
      }
      this.playing = true;
      this._lastTick = performance.now();
      this._emit('playState', true);
      this._rafId = requestAnimationFrame(this._tick);
    }

    pause() {
      if (!this.playing) return;
      this.playing = false;
      if (this._rafId) {
        cancelAnimationFrame(this._rafId);
        this._rafId = null;
      }
      this._emit('playState', false);
    }

    togglePlay() {
      if (this.playing) {
        this.pause();
      } else {
        this.play();
      }
    }

    _tick(now) {
      if (!this.playing) return;

      const baseInterval = 600 / this.speed;
      const elapsed = now - this._lastTick;

      if (elapsed >= baseInterval) {
        this._lastTick = now;
        if (this.currentIndex < this.frames.length - 1) {
          this.seek(this.currentIndex + 1);
        } else if (this.loop) {
          this.seek(0);
        } else {
          this.pause();
          return;
        }
      }

      this._rafId = requestAnimationFrame(this._tick);
    }
  }

  window.WW.ReplayController = ReplayController;
})(window);
