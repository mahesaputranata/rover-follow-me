"""
run_recorder.py - records a ros2 bag for every vision/follow session and serves finished runs.

Lives next to rover_server.py (same folder) and is used like this:

    from run_recorder import recorder as run_recorder
    run_recorder.start("FOLLOW_ME")   # when a mode starts
    run_recorder.stop()               # when it stops (no-op if nothing is recording)
    run_recorder.list_runs(20)        # for the "list_runs" websocket action
    run_recorder.export_run("last")   # for the "get_run" websocket action -> tgz + http path

Finished runs live in ~/runs/run_<date>_<time>_<mode>/ (bag/, config/, meta.json).
A tiny read-only HTTP file server on port 8000 serves ~/runs so the PC can download a
run as a .tgz without pushing megabytes through the websocket.

Stopping is non-blocking: the bag is finalised in a background thread, so stopping the
rover is never delayed by it.
"""
import atexit
import functools
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("run_recorder")

RUNS_DIR = Path(os.environ.get("ROVER_RUNS_DIR", str(Path.home() / "runs")))
EXPORT_DIR = RUNS_DIR / "_export"
TOPICS = ["/bbox_error", "/cmd_vel", "/rover/state",
          "/esp32/telemetry", "/esp32/warnings", "/esp32/odom"]      # add "/scan" with the lidar
CONFIG_FILES = [
    Path.home() / "rover_ws/src/rover_bridge/rover_bridge/vision/config/settings.py",
    Path.home() / "rover_ws/src/rover_control/src/lqr_node.cpp",
]
KEEP_LAST = 50          # older run_* folders are deleted unless they contain a file named KEEP
HTTP_PORT = int(os.environ.get("ROVER_RUNS_HTTP_PORT", "8000"))


class RunRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._run_dir = None
        self._meta = {}
        self._finalizer = None
        self.http_port = HTTP_PORT
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self._start_file_server()
        atexit.register(self._shutdown)
        # `pkill rover_server.py` sends SIGTERM, which skips atexit: finalise the bag first.
        try:
            if signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
                signal.signal(signal.SIGTERM, lambda *_: (self._shutdown(), sys.exit(0)))
        except (ValueError, OSError):       # not the main thread, or unsupported
            pass

    # ------------------------------------------------------------------ http
    def _start_file_server(self):
        try:
            handler = functools.partial(SimpleHTTPRequestHandler, directory=str(RUNS_DIR))
            server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), handler)
            threading.Thread(target=server.serve_forever, daemon=True, name="RunFileServer").start()
            log.info("[RUNS] serving %s on http port %d (read-only, LAN)", RUNS_DIR, HTTP_PORT)
        except OSError as e:
            log.warning("[RUNS] file server not started: %s", e)

    # ------------------------------------------------------------- recording
    def start(self, mode, label=None):
        """Begin recording a run. Stops (and finalises) any previous one first."""
        with self._lock:
            self._stop_locked()
            name = datetime.now().strftime("run_%Y-%m-%d_%H%M%S") + "_" + (label or str(mode).lower())
            run_dir = RUNS_DIR / name
            try:
                (run_dir / "config").mkdir(parents=True)
                for f in CONFIG_FILES:
                    if f.exists():
                        shutil.copy(f, run_dir / "config" / f.name)
                logf = open(run_dir / "recorder.log", "w")
                self._proc = subprocess.Popen(
                    ["ros2", "bag", "record", "-o", str(run_dir / "bag")] + TOPICS,
                    stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            except Exception as e:
                log.error("[RUNS] could not start recording: %s", e)
                self._proc = None
                return None
            self._run_dir = run_dir
            self._meta = {"run_id": name, "mode": str(mode), "started_at": time.time(), "topics": TOPICS}
            log.info("[RUNS] recording -> %s", run_dir)
            self._prune()
            return name

    def stop(self):
        """Stop recording. Returns immediately; the bag is finalised in the background."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        if self._proc is None:
            return
        proc, run_dir, meta = self._proc, self._run_dir, dict(self._meta)
        self._proc = None
        self._run_dir = None
        t = threading.Thread(target=self._finalize, args=(proc, run_dir, meta),
                             daemon=True, name="RunFinalize")
        self._finalizer = t
        t.start()

    def _finalize(self, proc, run_dir, meta):
        try:
            try:
                os.killpg(proc.pid, signal.SIGINT)      # rosbag2 needs SIGINT to write metadata.yaml
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=3)
            except ProcessLookupError:
                pass
        except Exception as e:
            log.warning("[RUNS] finalize problem: %s", e)
        meta["stopped_at"] = time.time()
        meta["duration_s"] = round(meta["stopped_at"] - meta["started_at"], 1)
        try:
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        except Exception as e:
            log.warning("[RUNS] could not write meta.json: %s", e)
        log.info("[RUNS] saved %s (%.0f s)", meta["run_id"], meta["duration_s"])

    def _wait_finalized(self, timeout=15):
        t = self._finalizer
        if t is not None and t.is_alive():
            t.join(timeout)

    def _shutdown(self):
        self.stop()
        self._wait_finalized()

    # ----------------------------------------------------------------- store
    def _prune(self):
        try:
            runs = sorted(p for p in RUNS_DIR.glob("run_*") if p.is_dir())
            for old in runs[:-KEEP_LAST]:
                if (old / "KEEP").exists() or old == self._run_dir:
                    continue
                shutil.rmtree(old, ignore_errors=True)
                (EXPORT_DIR / f"{old.name}.tgz").unlink(missing_ok=True)
                log.info("[RUNS] pruned %s", old.name)
        except Exception as e:
            log.warning("[RUNS] prune failed: %s", e)

    def _dir_size_mb(self, d):
        return round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1)

    def list_runs(self, limit=20):
        out = []
        for d in sorted((p for p in RUNS_DIR.glob("run_*") if p.is_dir()), reverse=True)[:limit]:
            entry = {"run_id": d.name, "size_mb": self._dir_size_mb(d),
                     "recording": d == self._run_dir}
            meta = d / "meta.json"
            if meta.exists():
                try:
                    m = json.loads(meta.read_text())
                    entry.update({k: m.get(k) for k in ("mode", "started_at", "duration_s")})
                except Exception:
                    pass
            out.append(entry)
        return out

    def export_run(self, run_id="last"):
        """Make ~/runs/_export/<run_id>.tgz and return where to download it."""
        self._wait_finalized()
        with self._lock:
            active = self._run_dir
        if run_id in (None, "", "last"):
            runs = [p for p in sorted(RUNS_DIR.glob("run_*"), reverse=True) if p.is_dir() and p != active]
            if not runs:
                return {"error": "no finished runs"}
            run_dir = runs[0]
        else:
            run_dir = RUNS_DIR / Path(str(run_id)).name        # no path tricks
        if not run_dir.is_dir():
            return {"error": f"run not found: {run_id}"}
        if run_dir == active:
            return {"error": "run is still recording; stop it first"}
        tgz = EXPORT_DIR / f"{run_dir.name}.tgz"
        if not tgz.exists():
            with tarfile.open(tgz, "w:gz") as tar:
                tar.add(run_dir, arcname=run_dir.name)
        return {"run_id": run_dir.name, "url_path": f"/_export/{tgz.name}",
                "http_port": self.http_port, "size_mb": round(tgz.stat().st_size / 1e6, 1)}


recorder = RunRecorder()
