"""Run one-file student agents in persistent, time-limited Python processes.

An agent exports ``choose_action(observation)`` and returns a JSON-compatible
action. Module globals survive successful turns and ordinary exceptions. A
timeout or process crash discards the worker; the next turn starts a fresh one.
Imports have a separate five-second limit, excluded from the action deadline.

This isolates faults, not access to files or the network. Use trusted classroom
submissions or an external OS/container sandbox for untrusted code.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Optional
import weakref


MAX_MESSAGE_BYTES = 1_048_576
_LIVE_AGENTS: weakref.WeakSet = weakref.WeakSet()
_NO_REQUEST = object()


def _encode_message(value, label):
    """Reject oversized messages while encoding, without building a huge copy."""
    encoded = bytearray()
    for part in json.JSONEncoder(allow_nan=False).iterencode(value):
        chunk = part.encode("utf-8")
        if len(encoded) + len(chunk) + 1 > MAX_MESSAGE_BYTES:
            raise ValueError(f"{label} exceeds the message limit")
        encoded.extend(chunk)
    encoded.extend(b"\n")
    return encoded


@dataclass(frozen=True)
class ActionResult:
    action: Any
    status: str
    elapsed: float
    error: Optional[str] = None


class _WorkerError(Exception):
    pass


class StudentAgent:
    """A persistent student worker; call ``close()`` or use a ``with`` block.

    ``act`` returns an ``ActionResult`` with status ``ok``, ``timeout`` or
    ``error``. ``elapsed`` is parent-measured action time, including IPC but
    including observation encoding, excluding startup. If startup fails, it
    reports startup time instead.
    Both deadlines use wall-clock time and work without Unix signal handlers.
    The optional seed initializes Python's ``random`` module in each worker.
    """

    def __init__(self, path, timeout=1.0, seed=None, startup_timeout=5.0):
        for name, value in (("timeout", timeout), ("startup_timeout", startup_timeout)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")
        self.path = Path(path).resolve()
        self.timeout = float(timeout)
        self.startup_timeout = float(startup_timeout)
        self.seed = seed
        self._process = None
        self._closed = False
        self._lock = threading.Lock()
        _LIVE_AGENTS.add(self)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("StudentAgent is closed")
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _exchange(self, payload, timeout):
        """Bound writes as well as reads; each exchange owns its response queue."""
        deadline = time.monotonic() + timeout
        process = self._process
        replies = queue.Queue(maxsize=1)

        def exchange():
            try:
                if payload is not _NO_REQUEST:
                    try:
                        encoded = _encode_message(payload, "Observation")
                    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
                        raise _WorkerError(f"Invalid observation: {exc}") from exc
                    process.stdin.write(encoded)
                    process.stdin.flush()
                raw = process.stdout.readline(MAX_MESSAGE_BYTES + 1)
                if not raw:
                    raise _WorkerError("Agent process exited without returning a response")
                if len(raw) > MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
                    raise _WorkerError("Agent response exceeds the message limit")
                response = json.loads(raw)
                if not isinstance(response, dict):
                    raise _WorkerError("Invalid agent response")
                replies.put((True, response))
            except Exception as exc:
                replies.put((False, exc))

        threading.Thread(target=exchange, daemon=True, name="wrongway-agent-io").start()
        try:
            success, value = replies.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError from None
        if time.monotonic() > deadline:
            raise TimeoutError
        if not success:
            raise _WorkerError(str(value)) from value
        return value

    def _start(self):
        started = time.monotonic()
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
            str(self.path), json.dumps(self.seed),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=-1,
            close_fds=True,
            start_new_session=(os.name == "posix"),
        )
        remaining = self.startup_timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError
        response = self._exchange(_NO_REQUEST, remaining)
        if response.get("status") != "ready":
            raise _WorkerError(response.get("error", "Agent startup failed"))

    def _stop_worker(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if os.name == "posix":
                # The worker owns a new session. Also stop subprocesses that
                # inherited that process group, even if the worker just exited.
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except OSError:
            # macOS can report EPERM for an already-exiting process group.
            # Recheck/reap the child and fall back to its PID if still alive.
            # Don't worry, this repo still be able to run on Windows or Linux. ;)
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    def act(self, observation):
        """Call the student's function once; failed calls have ``action=None``."""
        with self._lock:
            if self._closed:
                raise RuntimeError("StudentAgent is closed")
            if self._process is not None and self._process.poll() is not None:
                self._stop_worker()
            if self._process is None:
                started = time.monotonic()
                try:
                    self._start()
                except TimeoutError:
                    elapsed = time.monotonic() - started
                    self._stop_worker()
                    return ActionResult(None, "timeout", elapsed,
                                        f"Agent import exceeded {self.startup_timeout:g}s startup limit")
                except (OSError, _WorkerError) as exc:
                    elapsed = time.monotonic() - started
                    self._stop_worker()
                    return ActionResult(None, "error", elapsed, f"Agent startup failed: {exc}")

            started = time.monotonic()
            try:
                response = self._exchange(observation, self.timeout)
                elapsed = time.monotonic() - started
                if response.get("status") == "ok":
                    return ActionResult(response.get("action"), "ok", elapsed)
                if response.get("status") == "error":
                    return ActionResult(None, "error", elapsed, response.get("error", "Agent error"))
                raise _WorkerError("Invalid agent response status")
            except TimeoutError:
                elapsed = time.monotonic() - started
                self._stop_worker()
                return ActionResult(None, "timeout", elapsed,
                                    f"choose_action exceeded {self.timeout:g}s action limit")
            except _WorkerError as exc:
                elapsed = time.monotonic() - started
                self._stop_worker()
                return ActionResult(None, "error", elapsed, str(exc))

    def close(self):
        """Stop the child process and release its pipes; safe to call repeatedly."""
        with self._lock:
            self._closed = True
            self._stop_worker()
            _LIVE_AGENTS.discard(self)


def _close_live_agents():
    for agent in list(_LIVE_AGENTS):
        agent.close()


atexit.register(_close_live_agents)


def _error_text(exc):
    try:
        return f"{type(exc).__name__}: {exc}"[:2000]
    except BaseException:
        return type(exc).__name__


def _worker(path, seed):
    # Duplicate the protocol descriptors before redirecting normal I/O at the
    # OS level: print(), os.write(1, ...), and library output cannot corrupt it.
    protocol_in = os.fdopen(os.dup(sys.stdin.fileno()), "rb")
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    with open(os.devnull, "r+b", buffering=0) as null:
        for descriptor in (0, 1, 2):
            os.dup2(null.fileno(), descriptor)

    def send(response):
        encoded = _encode_message(response, "Action")
        protocol_out.write(encoded)
        protocol_out.flush()

    try:
        random.seed(seed)
        sys.path.insert(0, str(Path(path).parent))
        os.chdir(Path(path).parent)
        spec = importlib.util.spec_from_file_location("wrongway_submission", path)
        if spec is None or spec.loader is None:
            raise ImportError("Submission must be a Python source file")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        choose_action = getattr(module, "choose_action", None)
        if not callable(choose_action):
            raise TypeError("Submission must define choose_action(observation)")
    except BaseException as exc:
        send({"status": "error", "error": _error_text(exc)})
        return
    send({"status": "ready"})

    while True:
        line = protocol_in.readline(MAX_MESSAGE_BYTES + 1)
        if not line:
            return
        try:
            if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                raise ValueError("Observation exceeds the message limit")
            observation = json.loads(line)
            action = choose_action(observation)
            send({"status": "ok", "action": action})
        except BaseException as exc:
            send({"status": "error", "error": _error_text(exc)})


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Use StudentAgent from arena.py; this file is an internal worker.")
    _worker(sys.argv[2], json.loads(sys.argv[3]))
