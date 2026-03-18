"""
generated_log.py
================
NeuroScalpel — Persistent Generated Log System

Every pipeline run appends a structured, human-readable log to:
    logs/neuroscalpel_<date>.log        ← daily rotating file
    logs/latest.log                     ← always the last run

Each log entry contains:
  - Timestamp + Session ID
  - Phase-by-phase events (Phase 1 → Phase 5)
  - TARGET LOCKED coordinates
  - Edit result (method, weights changed, success)
  - Duration per phase

The GeneratedLogger is a singleton wired into MainWindow's pipeline signals.
It fires automatically — no manual calls needed from the UI layer.

Usage (auto-wired in MainWindow):
    from core.generated_log import gen_log
    gen_log.start_session(session_id, model_name, order)
    gen_log.log_phase(1, "Trick prompt generated", extra={"trick_prompt": "..."})
    gen_log.log_target(layer=7, point=42)
    gen_log.log_edit("ROME+LyapLock", weights=["transformer.h.7.mlp.c_proj.weight"], success=True)
    gen_log.close_session()
"""

from __future__ import annotations

import json
import logging
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_SEPARATOR = "─" * 72

# ── ANSI-free colour codes stored as plain labels in the log file ─────────────
_PHASE_LABELS = {
    1: "PHASE-1 │ DIAGNOSIS",
    2: "PHASE-2 │ NEURAL-SCAN",
    3: "PHASE-3 │ TARGET-LOCK",
    4: "PHASE-4 │ VISUALISE",
    5: "PHASE-5 │ ROME-EDIT",
}

logger = logging.getLogger("NeuroScalpel.GeneratedLog")


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class GeneratedLogger:
    """
    Singleton persistent log writer for NeuroScalpel pipeline runs.

    All output goes to:
      logs/neuroscalpel_YYYYMMDD.log   (daily rotating)
      logs/latest.log                  (symlink / overwrite for quick access)
    """

    def __init__(self):
        self._session_id: Optional[str] = None
        self._model_name: str = ""
        self._order: str = ""
        self._start_time: float = 0.0
        self._phase_times: Dict[int, float] = {}
        self._events: List[str] = []
        self._fh = None           # file handle
        self._latest_fh = None   # latest.log handle

    # ── Session lifecycle ────────────────────────────────────────────────────

    def start_session(self, session_id: str, model_name: str, order: str):
        """Opens log files and writes the session header."""
        self._close_handles()

        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        self._model_name = model_name
        self._order = order
        self._start_time = time.monotonic()
        self._phase_times.clear()
        self._events.clear()

        date_tag = datetime.datetime.now().strftime("%Y%m%d")
        log_path = _LOGS_DIR / f"neuroscalpel_{date_tag}.log"
        latest_path = _LOGS_DIR / "latest.log"

        try:
            self._fh = open(log_path, "a", encoding="utf-8")
            self._latest_fh = open(latest_path, "w", encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not open log file: {e}")
            return

        header = (
            f"\n{_SEPARATOR}\n"
            f"  NeuroScalpel  |  Session Start  |  {_ts()}\n"
            f"  Session ID : {session_id}\n"
            f"  Model      : {model_name}\n"
            f"  Order      : {order[:120]}\n"
            f"{_SEPARATOR}\n"
        )
        self._write(header)
        logger.info(f"GeneratedLog open -> {log_path.name}")

    def close_session(self, success: bool = True):
        """Writes session footer with total duration and closes handles."""
        if not self._fh:
            return
        elapsed = time.monotonic() - self._start_time
        status = "COMPLETED" if success else "ABORTED"
        footer = (
            f"\n{_SEPARATOR}\n"
            f"  Session {status}  |  {_ts()}  |  Total: {elapsed:.1f}s\n"
            f"{_SEPARATOR}\n\n"
        )
        self._write(footer)
        self._close_handles()
        logger.info(f"GeneratedLog closed. Duration: {elapsed:.1f}s")

    # ── Phase events ─────────────────────────────────────────────────────────

    def log_phase_start(self, phase: int, task_index: int = 0,
                        task_total: int = 1):
        """Called when a pipeline phase begins."""
        self._phase_times[phase] = time.monotonic()
        label = _PHASE_LABELS.get(phase, f"PHASE-{phase}")
        line = (
            f"\n  [{_ts()}]  {label}  "
            f"(Task {task_index + 1}/{task_total})\n"
        )
        self._write(line)

    def log_phase_event(self, phase: int, message: str,
                        data: Optional[Dict[str, Any]] = None):
        """Logs a single event within a phase, with optional structured data."""
        label = _PHASE_LABELS.get(phase, f"PHASE-{phase}")
        msg = f"    [{_ts()}]  {label}  >>  {message}"
        if data:
            msg += "\n      " + json.dumps(data, ensure_ascii=False)[:300]
        msg += "\n"
        self._write(msg)

    def log_phase_end(self, phase: int, result: str = "OK"):
        """Called when a phase finishes. Writes duration."""
        start = self._phase_times.get(phase, self._start_time)
        dt = time.monotonic() - start
        label = _PHASE_LABELS.get(phase, f"PHASE-{phase}")
        line = f"    [{_ts()}]  {label}  <<  {result}  [{dt:.2f}s]\n"
        self._write(line)

    # ── Specific structured events ────────────────────────────────────────────

    def log_tasks_parsed(self, tasks: list):
        """Logs the task list parsed from Phase 1."""
        self._write(f"\n  TASK QUEUE ({len(tasks)} task(s)):\n")
        for t in tasks:
            self._write(
                f"    [{t.index + 1}] subject='{t.subject}'  "
                f"wrong='{t.wrong_value}'  correct='{t.correct_value}'\n"
                f"         prompt='{t.trick_prompt[:100]}'\n"
            )

    def log_scan_result(self, critical_layer: str, max_deviation: float,
                        total_layers: int):
        """Logs Phase 2 tensor scan summary."""
        self._write(
            f"  SCAN RESULT:\n"
            f"    Critical Layer : {critical_layer}\n"
            f"    Max Deviation  : {max_deviation:.6f}\n"
            f"    Total Layers   : {total_layers}\n"
        )

    def log_target(self, layer: int, point: int, analysis_snippet: str = ""):
        """Logs the Phase 3 TARGET LOCKED result."""
        self._write(
            f"\n  *** TARGET LOCKED ***\n"
            f"    Layer  : {layer}\n"
            f"    Neuron : {point}\n"
        )
        if analysis_snippet:
            snippet = analysis_snippet.strip()[-400:]
            self._write(f"  Analysis Snippet:\n    {snippet}\n")

    def log_edit(self, method: str, weights: List[str],
                 success: bool, notes: str = ""):
        """Logs the Phase 5 edit result."""
        status = "SUCCESS" if success else "FAILED"
        self._write(
            f"\n  EDIT RESULT: {status}\n"
            f"    Method          : {method}\n"
            f"    Weights Changed : {len(weights)}\n"
        )
        for w in weights:
            self._write(f"      - {w}\n")
        if notes:
            self._write(f"    Notes : {notes}\n")

    def log_error(self, phase: int, error: str):
        """Logs an error in any phase."""
        label = _PHASE_LABELS.get(phase, f"PHASE-{phase}")
        self._write(
            f"\n  !! ERROR in {label}\n"
            f"     {error[:500]}\n"
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _write(self, text: str):
        for fh in (self._fh, self._latest_fh):
            if fh:
                try:
                    fh.write(text)
                    fh.flush()
                except Exception:
                    pass

    def _close_handles(self):
        for fh in (self._fh, self._latest_fh):
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        self._fh = None
        self._latest_fh = None


# ── Global singleton ──────────────────────────────────────────────────────────
gen_log = GeneratedLogger()
