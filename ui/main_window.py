"""
main_window.py
==============
NeuroScalpel – Main Application Window (Full Pipeline Orchestrator)

5-Phase Pipeline
----------------
Phase 1  — DeepSeek Surgeon Mind: parse user order, detect single or
            multiple issues, return structured JSON task list.
Phase 2  — PyTorch Neural Scan: load model, attach FFN hooks, run forward
            pass with the trick prompt, extract tensor deviations.
Phase 3  — DeepSeek Target Lock: analyse PyTorch telemetry, identify exact
            layer + vector point, emit "TARGET LOCKED: Layer [X], Point [Y]".
Phase 4  — Visualise: parse the TARGET LOCKED line, highlight the offending
            neuron in the 3D view with a pulsing red marker.
Phase 5  — ROME + LyapLock Edit: on user confirmation ("APPLY ROME EDIT"),
            perform the real rank-1 weight update, then run LyapLock to
            prevent catastrophic forgetting. Log everything to session DB.

Multi-task queue
----------------
If Phase 1 returns a list of tasks, the pipeline loops: after Phase 3/4
completes for task N, Phase 5 can be applied, then Phase 1–4 runs for
task N+1.  Progress is shown in the dashboard.

Session DB
----------
Every pipeline start creates a new session via SessionManager. All phases
write their results to the session's SQLite database.
"""

import re
import sys
import json
import logging
import traceback
import time
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from core.model_backend import ModelManager, apply_real_edit
from core.session_manager import session_manager
from core.task_queue import TaskQueue, EditTask
from core.generated_log import gen_log
from core.point_and_layer_detect import CoreAnomalyDetector
from ui.panels.feature_extractor import FeatureExtractorPanel
from ui.panels.dashboard import DashboardPanel
from ui.visualizer.point_cloud_3d import PointCloud3DWidget
from ui.panels.order_terminal import OrderTerminalWindow

from core.nvidia_agent import (
    DEEPSEEK_MODEL,
    default_chat_params_stream,
    nvidia_openai_client,
    stream_delta_reasoning_and_content,
)

logger = logging.getLogger("NeuroScalpel.MainWindow")


def _display_method(method: str) -> str:
    """
    Sanitize method strings for end-user visibility.
    Removes internal technique names like ROME / LyapLock from the UI.
    """
    m = method or ""
    m = m.replace("ROME_and_LyapLock_success", "UPDATE_and_STABILIZATION_success")
    m = m.replace("ROME_success_LyapLock_failed", "UPDATE_success_STABILIZATION_failed")
    m = m.replace("ROME_only", "UPDATE_only")
    m = m.replace("ROME", "UPDATE")
    m = m.replace("LyapLock", "STABILIZATION")
    m = m.replace("+", " + ")
    return " ".join(m.split())


# ===========================================================================
# PHASE 1 – DeepSeek Diagnosis (multi-task aware)
# ===========================================================================
class Phase1Thread(QThread):
    """
    Sends the user order to DeepSeek and asks it to return a structured
    JSON task list. Supports single or multiple issues in one order.

    Emits
    -----
    stream_mind(str, str)       – streaming thought text + colour
    phase1_complete(str)        – raw JSON string from DeepSeek
    error_occurred(str)
    """
    stream_mind    = pyqtSignal(str, str)
    phase1_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    SYSTEM_PROMPT = (
        "You are the 'Surgeon Mind', an autonomous AI agent embedded within "
        "'NeuroScalpel' — a tool that tracks information flow across an LLM's "
        "transformer layers using PyTorch hooks to surgically correct hallucinations "
        "without fine-tuning.\n\n"
        "The user will describe one OR MORE incorrect facts that the target model believes.\n\n"
        "Your EXACT task:\n"
        "1. Identify every distinct factual error in the user's message.\n"
        "2. For EACH error, create a targeted test question (trick_prompt) that forces "
        "the model to reproduce the error, enabling our tracking modules to locate it.\n"
        "3. Return ONLY a valid JSON array. No markdown, no code blocks, no extra text.\n\n"
        "Each element of the array MUST have exactly these keys:\n"
        "  \"analysis\"     : brief explanation of the error\n"
        "  \"trick_prompt\" : exact question to feed the model (forces the wrong answer)\n"
        "  \"subject\"      : the entity being corrected (e.g. 'Egypt')\n"
        "  \"wrong_value\"  : what the model currently (wrongly) says\n"
        "  \"correct_value\": what the correct answer is\n\n"
        "If there is only one issue, still return a JSON array with one element.\n"
        "Example: [{\"analysis\":\"...\",\"trick_prompt\":\"...\","
        "\"subject\":\"...\",\"wrong_value\":\"...\",\"correct_value\":\"...\"}]"
    )

    def __init__(self, order_text: str, parent=None):
        super().__init__(parent)
        self.order_text = order_text

    def run(self):
        self.stream_mind.emit("\n>> PHASE 1: DIAGNOSIS INITIALIZED <<\n", "#00f3ff")
        self.stream_mind.emit("> Analysing order for distinct issues...\n\n", "#0088aa")
        client = nvidia_openai_client()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"USER ORDER:\n{self.order_text}"}
        ]
        attempts = 3
        for i in range(1, attempts + 1):
            try:
                self.stream_mind.emit(f"\n>> Surgeon Mind processing... (attempt {i}/{attempts})\n", "#bc13fe")
                completion = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=1200,
                    timeout=120,
                    stream=True,
                    **default_chat_params_stream(),
                )
                content_chunks = []
                reasoning_chunks = []
                for chunk in completion:
                    reasoning, content = stream_delta_reasoning_and_content(chunk)
                    if reasoning is not None:
                        self.stream_mind.emit(reasoning, "#6688aa")
                        reasoning_chunks.append(reasoning)
                    if content is not None:
                        self.stream_mind.emit(content, "#bc13fe")
                        content_chunks.append(content)
                raw = "".join(content_chunks).strip()
                if not raw:
                    raw = "".join(reasoning_chunks).strip()
                if raw:
                    self.stream_mind.emit("\n\n>> JSON ready — Passing to Task Queue...\n", "#00f3ff")
                    self.phase1_complete.emit(raw)
                    return
                raise RuntimeError("Empty Phase 1 response.")
            except Exception as e:
                self.stream_mind.emit(
                    f"\n[PHASE 1 WARNING] attempt {i} failed: {e}\n",
                    "#ffaa00",
                )
                if i < attempts:
                    time.sleep(2 * i)
                    continue
                # Final fallback: non-stream request
                try:
                    self.stream_mind.emit("\n[PHASE 1] fallback mode (non-stream) ...\n", "#ffaa00")
                    resp = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=messages,
                        temperature=0.2,
                        top_p=0.95,
                        max_tokens=600,
                        timeout=90,
                        stream=False,
                        **default_chat_params_stream(),
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    if raw:
                        self.stream_mind.emit("\n>> JSON ready — Passing to Task Queue...\n", "#00f3ff")
                        self.phase1_complete.emit(raw)
                        return
                    raise RuntimeError("Empty fallback response.")
                except Exception as final_e:
                    msg = f"\n[PHASE 1 CRITICAL FAILURE] {final_e}\n{traceback.format_exc()}\n"
                    self.stream_mind.emit(msg, "#ff003c")
                    self.error_occurred.emit(str(final_e))
                    return


# ===========================================================================
# PHASE 2 – PyTorch Neural Scan
# ===========================================================================
class Phase2Thread(QThread):
    """
    Loads the model, attaches FFN hooks, runs the trick-prompt forward pass,
    and extracts tensor deviation telemetry.
    """
    stream_core     = pyqtSignal(str, str)
    phase2_complete = pyqtSignal(dict)
    error_occurred  = pyqtSignal(str)

    def __init__(self, trick_prompt: str, model_name: str = "openai-community/gpt2",
                 task_index: int = 0, model_manager: ModelManager = None,
                 forced_layer_idx: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.trick_prompt = trick_prompt
        self.model_name   = model_name
        self.task_index   = task_index
        self.model_manager = model_manager
        self.forced_layer_idx = forced_layer_idx

    def run(self):
        try:
            self.stream_core.emit(
                f"\n>> PHASE 2: NEURAL SCAN  [Task #{self.task_index + 1}] <<\n", "#00f3ff"
            )
            detector = CoreAnomalyDetector(model_name=self.model_name)

            if self.model_manager and self.model_manager.model is not None and self.model_manager.tokenizer is not None:
                detector.adopt_loaded_model(
                    model=self.model_manager.model,
                    tokenizer=self.model_manager.tokenizer,
                    model_name=self.model_manager.model_name or self.model_name,
                )
                self.stream_core.emit("[SYS] Reusing loaded model from memory.\n", "#00ff00")
            else:
                ok = detector.load_model(log_callback=self.stream_core.emit)
                if not ok:
                    self.stream_core.emit("[ERR] Model load failed. Aborting phase 2.\n", "#ff003c")
                    self.error_occurred.emit("Model load failure.")
                    return

            detector.attach_hooks(log_callback=self.stream_core.emit)
            analysis = detector.probe_and_analyze(
                self.trick_prompt,
                forced_layer_idx=self.forced_layer_idx,
                log_callback=self.stream_core.emit
            )
            # Do not free shared model if we are reusing ModelManager model.
            if not self.model_manager:
                detector.cleanup()

            if analysis:
                analysis["task_index"] = self.task_index

            self.stream_core.emit("\n>> Phase 2 Complete. Handing to Agent...\n", "#00f3ff")
            self.phase2_complete.emit(analysis or {"error": "no analysis", "task_index": self.task_index})

        except Exception as e:
            msg = f"\n[PHASE 2 CRITICAL FAILURE] {e}\n{traceback.format_exc()}\n"
            self.stream_core.emit(msg, "#ff003c")
            self.error_occurred.emit(str(e))


# ===========================================================================
# PHASE 3 – DeepSeek Target Lock + Thinking Protocol
# ===========================================================================
class Phase3Thread(QThread):
    """
    Feeds Phase 2 telemetry + original order to DeepSeek for deep analysis.
    MUST conclude with "TARGET LOCKED: Layer [X], Vector Point [Y]".
    """
    stream_mind     = pyqtSignal(str, str)
    phase3_complete = pyqtSignal(str)    # emits the full response text
    error_occurred  = pyqtSignal(str)

    SYSTEM_PROMPT = (
        "You are the 'Surgeon Mind', the analytical core of NeuroScalpel. "
        "You receive raw PyTorch tensor telemetry from live FFN hooks executed during "
        "a forward pass of the target LLM, along with Phase 2 neuron-targeting results "
        "computed via three independent methods: max FFN activation, ROME-style k* projection, "
        "and gradient sensitivity (\u2202loss/\u2202hidden).\n\n"
        "Review the telemetry and CONFIRM or CORRECT the Phase 2 (layer, neuron) identification. "
        "Provide a concise mechanistic explanation of WHY that neuron misfires for this fact.\n\n"
        "CRITICAL DIRECTIVE \u2014 PHASE 4 TARGET LOCK:\n"
        "You MUST conclude your entire response with this EXACT line and nothing after it:\n"
        "TARGET LOCKED: Layer [X], Vector Point [Y]\n"
        "Where X = integer layer index and Y = integer neuron index. "
        "Use the Phase 2 values unless you have strong evidence they require correction. "
        "Do NOT add punctuation, spaces, or extra text after that line."
    )

    def __init__(self, order_text: str, analysis_dict: dict,
                 task: "EditTask" = None, parent=None):
        super().__init__(parent)
        self.order_text   = order_text
        self.analysis_dict = analysis_dict
        self.task = task

    def run(self):
        try:
            self.stream_mind.emit("\n>> PHASE 3: POST-SCAN ANALYSIS + TARGET LOCK <<\n", "#00f3ff")
            self.stream_mind.emit(">> Ingesting PyTorch telemetry. Executing cognitive sub-routines...\n", "#0088aa")
            client = nvidia_openai_client()

            # Build user message — include Phase 2 neuron results for DeepSeek confirmation
            task_context = ""
            if self.task:
                task_context = (
                    f"\nCURRENT TASK: Correct '{self.task.wrong_value}' \u2192 '{self.task.correct_value}' "
                    f"for subject '{self.task.subject}'.\n"
                )

            phase2_neuron    = self.analysis_dict.get("critical_neuron", -1)
            phase2_layer     = self.analysis_dict.get("critical_layer", "?")
            phase2_layer_idx = self.analysis_dict.get("critical_layer_idx", -1)
            phase2_method    = self.analysis_dict.get("consensus_method", "?")
            phase2_agree     = self.analysis_dict.get("agreement_pct", 0)
            neuron_block = (
                f"\n=== PHASE 2 NEURON TARGETING RESULTS ===\n"
                f"Critical layer : {phase2_layer} (idx={phase2_layer_idx})\n"
                f"Critical neuron: {phase2_neuron}\n"
                f"Consensus method: {phase2_method} (agreement={phase2_agree:.0f}%)\n"
                f"Please CONFIRM or CORRECT these values.\n"
            ) if phase2_neuron >= 0 else ""

            raw_report = self.analysis_dict.get("raw_report", "")
            if "=== FULL HIDDEN-STATE SCAN" in raw_report:
                raw_report = raw_report.split("=== FULL HIDDEN-STATE SCAN")[0].rstrip()

            user_msg = (
                f"USER ORDER: {self.order_text}{task_context}{neuron_block}\n\n"
                f"RAW PYTORCH TENSOR LOG:\n{raw_report}"
            )

            attempts = 2
            for i in range(1, attempts + 1):
                try:
                    completion = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user",   "content": user_msg}
                        ],
                        temperature=0.5,
                        top_p=0.95,
                        max_tokens=1200,
                        timeout=120,
                        stream=True,
                        **default_chat_params_stream(),
                    )
                    self.stream_mind.emit(f"\n[ TARGET LOCK ANALYSIS attempt {i}/{attempts} ]\n", "#333355")
                    full_text = []
                    reasoning_text = []
                    for chunk in completion:
                        reasoning, content = stream_delta_reasoning_and_content(chunk)
                        if reasoning is not None and str(reasoning).strip():
                            self.stream_mind.emit(reasoning, "#6688aa")
                            reasoning_text.append(reasoning)
                        if content is not None and str(content).strip():
                            is_locked = "TARGET LOCKED" in content
                            color = "#ff2244" if is_locked else "#00f3ff"
                            self.stream_mind.emit(content, color)
                            full_text.append(content)

                    full_response = "".join(full_text).strip()
                    if not full_response:
                        full_response = "".join(reasoning_text).strip()
                    if full_response:
                        self.stream_mind.emit("\n\n>> OPERATION LOGIC SEQUENCE FULLY TERMINATED <<\n", "#bc13fe")
                        self.phase3_complete.emit(full_response)
                        return
                    raise RuntimeError("Empty Phase 3 response.")
                except Exception as e:
                    self.stream_mind.emit(f"\n[PHASE 3 WARNING] attempt {i} failed: {e}\n", "#ffaa00")
                    if i < attempts:
                        time.sleep(2 * i)
                        continue
                    # Fallback to non-stream mode before failing hard.
                    self.stream_mind.emit("\n[PHASE 3] fallback mode (non-stream) ...\n", "#ffaa00")
                    resp = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user",   "content": user_msg}
                        ],
                        temperature=0.2,
                        top_p=0.95,
                        max_tokens=600,
                        timeout=90,
                        stream=False,
                        **default_chat_params_stream(),
                    )
                    msg = resp.choices[0].message
                    raw = (getattr(msg, "content", None) or "").strip()
                    if not raw:
                        raw = (getattr(msg, "reasoning_content", None) or "").strip()
                    if raw:
                        self.stream_mind.emit("\n\n>> OPERATION LOGIC SEQUENCE FULLY TERMINATED <<\n", "#bc13fe")
                        self.phase3_complete.emit(raw)
                        return
                    raise RuntimeError("Empty Phase 3 fallback response.")

        except Exception as e:
            msg = f"\n[PHASE 3 CRITICAL FAILURE] {e}\n{traceback.format_exc()}\n"
            self.stream_mind.emit(msg, "#ff003c")
            self.error_occurred.emit(str(e))


# ===========================================================================
# PHASE 5 – ROME + LyapLock Edit Thread
# ===========================================================================
class Phase5EditThread(QThread):
    """
    Runs the real ROME rank-1 edit followed by LyapLock stabilisation
    in a background thread so the UI stays responsive.
    """
    stream_edit     = pyqtSignal(str, str)
    phase5_complete = pyqtSignal(dict)   # {"success": bool, "method": str, ...}
    error_occurred  = pyqtSignal(str)

    def __init__(self, model_manager: ModelManager, task: "EditTask",
                 parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.task = task

    def run(self):
        neuron_hint = getattr(self.task, "target_point", -1)
        self.stream_edit.emit(
            f"\n>> PHASE 5: NEURAL EDIT  [Task #{self.task.index + 1}] <<\n",
            "#bc13fe"
        )
        self.stream_edit.emit(
            f">> Subject      : {self.task.subject}\n"
            f">> Wrong Value  : {self.task.wrong_value}\n"
            f">> Correct Value: {self.task.correct_value}\n"
            f">> Prompt       : {self.task.trick_prompt}\n"
            f">> Target Layer : {self.task.target_layer}\n"
            f">> Target Neuron: {neuron_hint}\n",
            "#0088aa"
        )

        try:
            result = apply_real_edit(
                model_manager=self.model_manager,
                subject=self.task.subject,
                prompt_template=self.task.trick_prompt,
                target_new=self.task.correct_value,
                target_old=self.task.wrong_value,
                layer_hint=self.task.target_layer,
                neuron_hint=neuron_hint if neuron_hint >= 0 else None,
                log_callback=self.stream_edit.emit
            )
            self.stream_edit.emit(
                f"\n>> Phase 5 Complete: {_display_method(result.get('method', ''))}\n",
                "#00ff00",
            )
            self.phase5_complete.emit(result)

        except Exception as e:
            msg = f"\n[PHASE 5 CRITICAL FAILURE] {e}\n{traceback.format_exc()}\n"
            self.stream_edit.emit(msg, "#ff003c")
            self.error_occurred.emit(str(e))


# ===========================================================================
# MAIN WINDOW
# ===========================================================================
class MainWindow(QMainWindow):
    """
    Orchestrates all 5 phases across the pipeline for single or multiple tasks.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroScalpel — LLM Neural Surgery v2.0")
        self.resize(1700, 1020)

        # Backend
        self.backend = ModelManager()
        self.active_model_name: str = ""
        self.last_order_text: str = ""

        # Pipeline state
        self.task_queue = TaskQueue()
        self.current_task: EditTask = None
        self._phase2_attempts_required = 3
        self._phase2_attempts: list[dict] = []
        self._phase2_forced_layer: Optional[int] = None
        self._phase2_pass4_recheck: bool = False

        # Active threads (kept as attrs to prevent GC)
        self.t_phase1: Phase1Thread = None
        self.t_phase2: Phase2Thread = None
        self.t_phase3: Phase3Thread = None
        self.t_phase5: Phase5EditThread = None

        # Spawned sub-windows
        self.active_sub_windows: list = []
        self.win_mind: OrderTerminalWindow = None
        self.win_logs: OrderTerminalWindow = None
        self.win_edit: OrderTerminalWindow = None

        self._init_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("MainSplitter")

        self.feature_panel = FeatureExtractorPanel(self.backend)
        self.visualizer    = PointCloud3DWidget()
        self.visualizer.setObjectName("VisualizerCanvas")
        self.dashboard     = DashboardPanel()

        self.splitter.addWidget(self.feature_panel)
        self.splitter.addWidget(self.visualizer)
        self.splitter.addWidget(self.dashboard)
        self.splitter.setSizes([310, 1080, 310])
        self.splitter.setHandleWidth(2)

        layout.addWidget(self.splitter)

    def _connect_signals(self):
        # Feature panel
        self.feature_panel.modelLoaded.connect(self._on_model_loaded)

        # Visualizer
        self.visualizer.pointSelected.connect(self._on_point_selected)
        self.visualizer.pointMoved.connect(self._on_point_dragged)
        self.visualizer.targetNeuronVisualized.connect(self._on_target_visualized)

        # Dashboard
        self.dashboard.startWordRequested.connect(self._on_start_word_requested)
        self.dashboard.applyRomeRequested.connect(self._on_apply_rome_requested)
        self.dashboard.targetLockEdited.connect(self._on_target_lock_edited)
        self.dashboard.runAbliterationRequested.connect(self._on_run_abliteration)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _on_model_loaded(self, model_id: str):
        self.active_model_name = self.backend.model_name or model_id
        self.feature_panel.update_readout(
            f"Model loaded: {model_id}\nExtracting layered 3D geometry..."
        )

        # Try layered geometry first (requires model in memory)
        if self.backend.model is not None:
            pts, labels, lmap = self.backend.get_layer_neuron_geometry(
                num_neurons_per_layer=None,
                log_callback=self.feature_panel.update_readout_colored
            )
            if len(pts) > 1:
                self.visualizer.load_layered_model(pts, labels, lmap)
                self.feature_panel.update_readout(
                    f"Layered 3D view: {len(pts)} neurons across {len(lmap)} layers ✅"
                )
                return

        # Fallback: flat PCA embedding cloud
        pts, ids = self.backend.get_real_weights(
            model_id=model_id, num_points=None,
            log_callback=self.feature_panel.update_readout_colored
        )
        self.visualizer.load_points(pts, ids)
        self.feature_panel.update_readout(
            f"Flat embedding cloud: {len(ids)} vectors ✅"
        )

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def _on_start_word_requested(self, order_text: str):
        if not order_text.strip():
            self.feature_panel.update_readout("ERROR: Empty order.")
            return
        ready, reason = self._is_model_ready()
        if not ready:
            self.feature_panel.update_readout(
                "ERROR: No model loaded.\n"
                f"{reason}\n"
                "Load a model first before starting a pipeline run."
            )
            return

        self.last_order_text = order_text
        self._stop_all_threads()
        self.visualizer.reset_target()
        self.dashboard.clear_target_lock()

        # Create session in DB + open generated log
        try:
            session_id = session_manager.create_session(
                model_name=self.active_model_name or "unknown",
                order_text=order_text
            )
            self.feature_panel.update_readout(f"Session created: {session_id}")
            gen_log.start_session(session_id, self.active_model_name or "unknown", order_text)
        except Exception as e:
            self.feature_panel.update_readout(f"Session DB warning: {e}")

        # Spawn terminal windows
        self._spawn_terminal_windows()

        # Log kicks off
        self.win_mind.append_text(
            f">> SESSION STARTED\n>> ORDER: {order_text[:200]}\n", "#00f3ff"
        )

        self._start_phase1(order_text)

    # ------------------------------------------------------------------
    # PHASE 1
    # ------------------------------------------------------------------

    def _start_phase1(self, order_text: str):
        gen_log.log_phase_start(1, 0, 1)
        gen_log.log_phase_event(1, f"Order received: {order_text[:120]}")
        self.feature_panel.update_readout("Phase 1: Sending to Surgeon Mind...")
        self.t_phase1 = Phase1Thread(order_text, parent=self)
        self.t_phase1.stream_mind.connect(self._append_mind_log)
        self.t_phase1.phase1_complete.connect(self._on_phase1_complete)
        self.t_phase1.error_occurred.connect(self._on_error)
        self.t_phase1.start()

    def _on_phase1_complete(self, raw_json: str):
        """Parse tasks from DeepSeek JSON response."""
        gen_log.log_phase_end(1, "Trick prompts formulated")
        tasks = self.task_queue.parse_from_phase1_response(raw_json)
        total = self.task_queue.total

        self.win_mind.append_text(
            f"\n>> {total} task(s) queued.\n{self.task_queue.summary_text()}\n\n",
            "#00ff00"
        )

        gen_log.log_tasks_parsed(tasks)

        # Persist tasks to session DB
        for t in tasks:
            try:
                session_manager.log_task(
                    task_index=t.index,
                    trick_prompt=t.trick_prompt,
                    subject=t.subject,
                    wrong_value=t.wrong_value,
                    correct_value=t.correct_value
                )
            except Exception:
                pass

        self._run_next_task()

    # ------------------------------------------------------------------
    # Task queue runner
    # ------------------------------------------------------------------

    def _run_next_task(self):
        if not self.task_queue.has_next():
            self._on_all_tasks_complete()
            return

        task = self.task_queue.pop_next()
        self.task_queue.advance()
        self.current_task = task

        # Update dashboard queue display
        pos = self.task_queue.current_position - 1  # already advanced
        total = self.task_queue.total
        self.dashboard.update_task_queue(
            self.task_queue.summary_text(), pos, total
        )
        self.feature_panel.update_readout(
            f"Task {pos}/{total}: {task.subject} | {task.wrong_value} → {task.correct_value}"
        )

        # Persist status
        try:
            session_manager.update_task_status(task.index, "running")
        except Exception:
            pass

        self._start_phase2(task)

    def _on_all_tasks_complete(self):
        self.feature_panel.update_readout(
            "All tasks complete ✅\nSession data saved to sessions/ folder."
        )
        self.win_mind.append_text(
            "\n>> ALL TASKS COMPLETED SUCCESSFULLY <<\n", "#00ff00"
        )
        
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        from PyQt6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Post-Edit Evaluation")
        dialog.setFixedSize(480, 220)
        dialog.setStyleSheet("background-color: #1e1e2e; color: #fff; border: 1px solid #555;")
        
        layout = QVBoxLayout(dialog)
        
        lbl_title = QLabel("✅ All Weights Edited Successfully!")
        lbl_title.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold; border: none;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_desc = QLabel(
            "The model weights have been surgically stabilized in-memory.\n\n"
            "Do you want to open a chat window to test the newly modified AI model? "
            "(The main program will remain safely open, and chatting will use 0 extra VRAM)"
        )
        lbl_desc.setStyleSheet("color: #00f3ff; font-size: 13px; border: none;")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_desc)
        
        btn_layout = QHBoxLayout()
        btn_no = QPushButton("No, Close")
        btn_no.setStyleSheet("background-color: #2e2e3e; color: #ff2244; border: 1px solid #ff2244; padding: 10px; font-weight: bold; border-radius: 4px;")
        btn_no.clicked.connect(dialog.reject)
        
        btn_yes = QPushButton("Yes, Chat with AI 🤖")
        btn_yes.setStyleSheet("background-color: #3e1e1e; color: #00ff00; border: 1px solid #00ff00; padding: 10px; font-weight: bold; border-radius: 4px;")
        btn_yes.clicked.connect(dialog.accept)
        
        btn_layout.addWidget(btn_no)
        btn_layout.addWidget(btn_yes)
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                from ui.panels.chat_panel import InternalChatWindow
                self._internal_chat = InternalChatWindow(self.backend)
                self._internal_chat.show()
            except Exception as e:
                self.win_mind.append_text(f"\n>> Failed to load internal chat: {e}\n", "#ff003c")

    # ------------------------------------------------------------------
    # PHASE 2
    # ------------------------------------------------------------------

    def _start_phase2(self, task: EditTask):
        ready, reason = self._is_model_ready()
        if not ready:
            self.feature_panel.update_readout(
                "ERROR: No model loaded.\n"
                f"{reason}\n"
                "Load a model first before starting a pipeline run."
            )
            return
        model = self.active_model_name or self.backend.model_name
        self._phase2_attempts = []
        self._phase2_forced_layer = None
        self._phase2_pass4_recheck = False
        gen_log.log_phase_start(2, task.index, self.task_queue.total)
        gen_log.log_phase_event(2, f"Scanning model '{model}' with prompt: {task.trick_prompt[:80]}")
        self.feature_panel.update_readout(
            f"Phase 2: Neural scan for task #{task.index + 1}...\n"
            f"Strict verification: {self._phase2_attempts_required}/"
            f"{self._phase2_attempts_required} required."
        )
        self._start_phase2_attempt(task=task, model=model, attempt=1)

    def _start_phase2_attempt(self, task: EditTask, model: str, attempt: int):
        force_msg = f" (forced layer={self._phase2_forced_layer})" if self._phase2_forced_layer is not None else ""
        self.win_logs.append_text(
            f"[VERIFY] Phase 2 target verification attempt {attempt}/"
            f"{self._phase2_attempts_required}{force_msg}\n",
            "#00f3ff",
        )
        self.t_phase2 = Phase2Thread(
            trick_prompt=task.trick_prompt,
            model_name=model,
            task_index=task.index,
            model_manager=self.backend,
            forced_layer_idx=self._phase2_forced_layer,
            parent=self
        )
        self.t_phase2.stream_core.connect(self._append_core_log)
        self.t_phase2.phase2_complete.connect(self._on_phase2_complete)
        self.t_phase2.error_occurred.connect(self._on_error)
        self.t_phase2.start()

    def _on_phase2_complete(self, analysis: dict):
        self._phase2_attempts.append(analysis)
        current_attempt = len(self._phase2_attempts)
        if current_attempt < self._phase2_attempts_required:
            model = self.active_model_name or self.backend.model_name
            self._start_phase2_attempt(
                task=self.current_task,
                model=model,
                attempt=current_attempt + 1
            )
            return

        targets = [self._extract_target(a) for a in self._phase2_attempts]
        if len(set(targets)) != 1:
            self.feature_panel.update_readout(
                "Phase 2 verification failed.\n"
                f"Observed targets: {targets}\n"
                "Pipeline stopped before edit."
            )
            if self.current_task:
                try:
                    session_manager.update_task_status(self.current_task.index, "failed")
                except Exception:
                    pass
            return

        locked_target = targets[0]
        io_ref_layer = self._pass4_io_referee(self._phase2_attempts)
        if io_ref_layer is None:
            self.feature_panel.update_readout(
                "Phase 2 verification failed.\n"
                "Pass 4 (layer I/O referee) could not compute a valid layer.\n"
                "Pipeline stopped before edit."
            )
            if self.current_task:
                try:
                    session_manager.update_task_status(self.current_task.index, "failed")
                except Exception:
                    pass
            return
        if io_ref_layer != locked_target[0]:
            if not self._phase2_pass4_recheck:
                self._phase2_pass4_recheck = True
                self._phase2_forced_layer = io_ref_layer
                self._phase2_attempts = []
                model = self.active_model_name or self.backend.model_name
                self.feature_panel.update_readout(
                    "Phase 2 pass4 selected a different layer.\n"
                    f"Re-running 3/3 verification constrained to layer {io_ref_layer}..."
                )
                self._start_phase2_attempt(task=self.current_task, model=model, attempt=1)
                return
            self.feature_panel.update_readout(
                "Phase 2 verification failed after pass4-constrained recheck.\n"
                f"Expected forced layer {self._phase2_forced_layer}, got targets {targets}.\n"
                "Pipeline stopped before edit."
            )
            if self.current_task:
                try:
                    session_manager.update_task_status(self.current_task.index, "failed")
                except Exception:
                    pass
            return

        analysis = self._phase2_attempts[-1]
        task_idx = analysis.get("task_index", 0)

        crit      = analysis.get("critical_layer", "unknown")
        crit_idx  = analysis.get("critical_layer_idx", -1)
        neuron    = analysis.get("critical_neuron", -1)
        dev       = analysis.get("max_magnitude", 0.0)
        if locked_target:
            crit_idx, neuron = locked_target
            crit = f"layer.{crit_idx}"

        # Store both layer and neuron on the current_task for Phase 5
        if self.current_task:
            if crit_idx >= 0:
                self.current_task.target_layer = crit_idx
            if neuron >= 0:
                self.current_task.target_point = neuron

        try:
            session_manager.log_scan_result(
                task_index=task_idx,
                critical_layer=crit,
                max_deviation=dev,
                raw_report=analysis.get("raw_report", "")
            )
        except Exception:
            pass
        gen_log.log_scan_result(crit, dev, total_layers=0)
        gen_log.log_phase_end(2,
            f"Critical layer: {crit} (idx={crit_idx}) | neuron: {neuron} | dev: {dev:.4f}")

        self.feature_panel.update_readout(
            f"Phase 2 complete.\n"
            f"Strict verification: PASSED ({self._phase2_attempts_required}/"
            f"{self._phase2_attempts_required}) + Pass4 I/O=PASSED (layer {io_ref_layer})\n"
            f"Critical layer: {crit} (idx={crit_idx})\n"
            f"Critical neuron: {neuron}\n"
            f"Max deviation: {dev:.4f}"
        )
        self._start_phase3(analysis, locked_target=locked_target)

    # ------------------------------------------------------------------
    # PHASE 3
    # ------------------------------------------------------------------

    def _start_phase3(self, analysis: dict, locked_target: Optional[Tuple[int, int]] = None):
        gen_log.log_phase_start(3, self.current_task.index if self.current_task else 0, self.task_queue.total)
        self.feature_panel.update_readout("Phase 3: Target Lock analysis...")
        self.t_phase3 = Phase3Thread(
            order_text=self.last_order_text,
            analysis_dict=analysis,
            task=self.current_task,
            parent=self
        )
        self.t_phase3.stream_mind.connect(self._append_mind_log)
        self.t_phase3.phase3_complete.connect(
            lambda full_response: self._on_phase3_complete(full_response, locked_target=locked_target)
        )
        self.t_phase3.error_occurred.connect(
            lambda error: self._on_phase3_error(error, locked_target=locked_target)
        )
        self.t_phase3.start()

    def _on_phase3_complete(self, full_response: str, locked_target: Optional[Tuple[int, int]] = None):
        """
        Parse "TARGET LOCKED: Layer [X], Vector Point [Y]" from Phase 3 output.
        Trigger Phase 4 (visualisation) and enable ROME edit button.
        """
        parsed_layer, parsed_point = self._parse_target_lock(full_response)
        if locked_target:
            layer_idx, vec_pt = locked_target
            if parsed_layer is not None and (parsed_layer, parsed_point) != locked_target:
                self.win_mind.append_text(
                    f"[PHASE 3] Advisory mismatch ignored. Parsed ({parsed_layer}, {parsed_point}) "
                    f"but locked target is ({layer_idx}, {vec_pt}).\n",
                    "#ffaa00",
                )
        else:
            layer_idx, vec_pt = parsed_layer, parsed_point

        if layer_idx is not None and self.current_task:
            self.current_task.target_layer = layer_idx
            self.current_task.target_point = vec_pt
            self.task_queue.set_target(self.current_task.index, layer_idx, vec_pt)

            # Persist target to session DB
            try:
                session_manager.log_target(
                    task_index=self.current_task.index,
                    layer_idx=layer_idx,
                    vector_point=vec_pt,
                    analysis_summary=full_response[-1000:]
                )
                session_manager.update_task_status(self.current_task.index, "done")
            except Exception:
                pass

            # Generated log
            gen_log.log_target(layer_idx, vec_pt, full_response[-400:])
            gen_log.log_phase_end(3, f"TARGET LOCKED Layer {layer_idx} Neuron {vec_pt}")

            # PHASE 4: Visualise target neuron
            self._phase4_visualize(layer_idx, vec_pt)

        self.feature_panel.update_readout(
            f"Phase 3 complete.\nTarget: Layer {layer_idx}, Point {vec_pt}"
        )

    # ------------------------------------------------------------------
    # PHASE 4 — Visualisation (synchronous, just UI updates)
    # ------------------------------------------------------------------

    def _phase4_visualize(self, layer_idx: int, vec_pt: int):
        """Highlights the target neuron in the 3D view."""
        self.visualizer.highlight_target_neuron(layer_idx, vec_pt)
        self.dashboard.set_target_locked(layer_idx, vec_pt)
        self.win_mind.append_text(
            f"\n>> PHASE 4: TARGET VISUALISED\n"
            f"   Layer {layer_idx} | Neuron {vec_pt} → highlighted in 3D view\n",
            "#ff2244"
        )

    def _on_target_visualized(self, layer_idx: int, point_idx: int):
        self.feature_panel.update_readout(
            f"3D target highlighted: Layer {layer_idx}, Neuron {point_idx} 🎯"
        )

    # ------------------------------------------------------------------
    # PHASE 5 — Neural Edit
    # ------------------------------------------------------------------

    def _on_apply_rome_requested(self):
        if self.current_task is None:
            self.feature_panel.update_readout("ERROR: No active task to edit.")
            return
        if self.backend.model is None:
            self.feature_panel.update_readout(
                "ERROR: Model not loaded in memory.\n"
                "Load the same model via the Feature Panel first."
            )
            self.dashboard.btn_rome_edit.setEnabled(True)
            self.dashboard.btn_rome_edit.setText("🧬  APPLY NEURAL EDIT")
            return

        gen_log.log_phase_start(5, self.current_task.index, self.task_queue.total)
        self.feature_panel.update_readout(
            f"Phase 5: Applying neural edit for task #{self.current_task.index + 1}..."
        )

        self.t_phase5 = Phase5EditThread(
            model_manager=self.backend,
            task=self.current_task,
            parent=self
        )
        self.t_phase5.stream_edit.connect(self._append_edit_log)
        self.t_phase5.phase5_complete.connect(self._on_phase5_complete)
        self.t_phase5.error_occurred.connect(self._on_phase5_error)
        self.t_phase5.start()

    def _on_phase5_complete(self, result: dict):
        self.dashboard.btn_rome_edit.setText("[OK] EDIT APPLIED")

        weights = result.get("weights", [])
        method  = result.get("method", "unknown")
        disp_method = _display_method(method)
        success = result.get("success", False)
        notes   = result.get("notes", "")
        post_checks = result.get("post_checks", {})

        try:
            session_manager.log_edit(
                task_index=self.current_task.index if self.current_task else 0,
                method=method,
                weights_changed=json.dumps(weights),
                success=success,
                notes=notes
            )
        except Exception:
            pass

        gen_log.log_edit(method, weights, success, notes)
        gen_log.log_phase_end(5, disp_method if success else "FAILED")

        self.feature_panel.update_readout(
            f"Edit complete\nMethod: {disp_method}\n"
            f"Weights modified: {len(weights)}\n"
            f"Post-checks: {post_checks}"
        )
        self.win_edit.append_text(
            f"\n>> EDIT COMPLETE: {disp_method}\n", "#00ff00"
        )

        # Advance to next queued task if any
        if self.task_queue.has_next():
            self.win_mind.append_text(
                "\n>> Advancing to next queued task...\n", "#00f3ff"
            )
            self._run_next_task()
        else:
            self._on_all_tasks_complete()

    def _on_phase5_error(self, error: str):
        self.dashboard.btn_rome_edit.setEnabled(True)
        self.dashboard.btn_rome_edit.setText("🧬  APPLY NEURAL EDIT")
        self.feature_panel.update_readout(f"Edit failed: {error}")

    def _on_phase3_error(self, error: str, locked_target: Optional[Tuple[int, int]] = None):
        """
        If DeepSeek target-lock analysis fails, continue with the verified
        Phase-2 locked target instead of stalling the pipeline.
        """
        self._append_mind_log(
            f"\n[PHASE 3 WARNING] {error}\n"
            "Falling back to verified Phase-2 target.\n",
            "#ffaa00",
        )
        if locked_target is None:
            self._on_error(error)
            return
        fallback_text = (
            f"TARGET LOCKED: Layer [{locked_target[0]}], Vector Point [{locked_target[1]}]"
        )
        self._on_phase3_complete(fallback_text, locked_target=locked_target)

    # ------------------------------------------------------------------
    # Legacy drag and visualizer stubs
    # ------------------------------------------------------------------

    def _on_point_selected(self, point_id: int, coords: tuple):
        pass  # Legacy feature removed from dashboard

    def _on_point_dragged(self, point_id: int, coords: tuple):
        pass  # Legacy feature removed from dashboard

    # ------------------------------------------------------------------
    # Fast Options
    # ------------------------------------------------------------------

    def _on_run_abliteration(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        from PyQt6.QtCore import Qt
        
        self.feature_panel.update_readout("Running FAST OPTION: \ncompute_refusal_dir.py...\n(Check main python console)")
        if self.backend.model is None:
            self.feature_panel.update_readout("ERROR: No model loaded.")
            return
            
        model_id = self.active_model_name or self.backend.model_name
        
        class AbliterationWorker(QThread):
            log_signal = pyqtSignal(str, str)
            success_signal = pyqtSignal()

            def __init__(self, model_id, parent=None):
                super().__init__(parent)
                self.model_id = model_id

            def run(self):
                import subprocess
                import sys
                import os
                
                script_dir = os.path.abspath("core/Abliteration")
                compute_script = os.path.join(script_dir, "compute_refusal_dir.py")
                
                self.log_signal.emit(f"\n>> Launching {compute_script} in background...\n", "#00f3ff")
                res = subprocess.run([sys.executable, compute_script, self.model_id])
                
                if res.returncode == 0:
                    self.success_signal.emit()
                else:
                    self.log_signal.emit(f"\n>> Compute script failed with code {res.returncode}.\n", "#ff003c")
        
        def on_success():
            self._append_edit_log(f"\n>> Success! Refusal direction computed.\n", "#00ff00")
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Abliteration Vector Setup Complete")
            dialog.setFixedSize(580, 270)
            dialog.setStyleSheet("background-color: #1e1e2e; color: #fff; border: 1px solid #555;")
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            lbl_title = QLabel("✅ Abliteration .pt File Generated Successfully!")
            lbl_title.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold; border: none;")
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_title.setWordWrap(True)
            
            lbl_desc = QLabel(
                "You are now ready to REMOVE the Refusal Vector from the model dynamically.\n\n"
                "How do you want to apply the patch and chat with the AI?"
            )
            lbl_desc.setStyleSheet("color: #00f3ff; font-size: 14px; border: none; padding: 10px;")
            lbl_desc.setWordWrap(True)
            lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            layout.addWidget(lbl_title)
            layout.addWidget(lbl_desc)
            
            btn_layout = QHBoxLayout()
            
            btn_term = QPushButton("Run 'inference.py' in Terminal\n(Standard Console)")
            btn_term.setStyleSheet("background-color: #2e2e3e; color: #00f3ff; border: 1px solid #00f3ff; padding: 10px; font-weight: bold; border-radius: 4px;")
            
            btn_gui = QPushButton("Apply Vector & Open Chat UI\n(Closes NeuroScalpel to Free VRAM)")
            btn_gui.setStyleSheet("background-color: #3e1e1e; color: #00ff00; border: 1px solid #00ff00; padding: 10px; font-weight: bold; border-radius: 4px;")
            
            def on_term_clicked():
                dialog.done(2)
                
            def on_gui_clicked():
                dialog.done(1)
                
            btn_term.clicked.connect(on_term_clicked)
            btn_gui.clicked.connect(on_gui_clicked)
            
            btn_layout.addWidget(btn_term)
            btn_layout.addWidget(btn_gui)
            layout.addLayout(btn_layout)
            
            res = dialog.exec()
            
            if res == 1:
                self._append_edit_log("\n>> Launching Graphic Chat Interface... Removing refusal vector dynamically...\n", "#00ff00")
                import subprocess, sys, os
                from PyQt6.QtWidgets import QApplication
                
                script_dir = os.path.abspath("core/Abliteration")
                chat_script = os.path.join(script_dir, "chat_ui.py")
                
                subprocess.Popen([sys.executable, chat_script, model_id])
                
                # Directly shut down this main application to free all VRAM
                QApplication.instance().quit()
            elif res == 2:
                self._append_edit_log("\n>> Launching inference.py in Terminal... Removing refusal vector...\n", "#00f3ff")
                import subprocess, sys, os
                script_dir = os.path.abspath("core/Abliteration")
                inference_script = os.path.join(script_dir, "inference.py")
                
                cmd_str = f"{sys.executable} {inference_script} {model_id}"
                terminals = [
                    (["gnome-terminal", "--"], cmd_str.split()),
                    (["x-terminal-emulator", "-e"], [cmd_str]),
                    (["konsole", "-e"], [cmd_str]),
                    (["xterm", "-e"], cmd_str.split())
                ]
                
                launched = False
                for term_args, cmd_args in terminals:
                    try:
                        subprocess.Popen(term_args + cmd_args)
                        launched = True
                        break
                    except Exception:
                        pass
                
                if not launched:
                    self._append_edit_log("\n>> Failed: Could not find a supported terminal emulator. Please run manually.\n", "#ff003c")
                    self._append_edit_log(f">> python {inference_script} {model_id}\n", "#ff003c")
            else:
                import os
                script_dir = os.path.abspath("core/Abliteration")
                inference_script = os.path.join(script_dir, "inference.py")
                self._append_edit_log(f"\n>> [SAFE MODE] Test the model later by running:\n>> python {inference_script} {model_id}\n", "#00f3ff")

        self._spawn_terminal_windows()
        self._abliteration_worker = AbliterationWorker(model_id, self)
        self._abliteration_worker.log_signal.connect(self._append_edit_log)
        self._abliteration_worker.success_signal.connect(on_success)
        self._abliteration_worker.start()

    def _on_target_lock_edited(self, new_layer: int, new_point: int):
        if self.current_task:
            self.current_task.target_layer = new_layer
            self.current_task.target_point = new_point
            self.task_queue.set_target(self.current_task.index, new_layer, new_point)
            try:
                session_manager.log_target(
                    task_index=self.current_task.index,
                    layer_idx=new_layer,
                    vector_point=new_point,
                    analysis_summary="Manually updated via dashboard"
                )
            except Exception:
                pass
        self.feature_panel.update_readout(f"Target manually adjusted to Layer {new_layer}, Neuron {new_point}.")
        self.visualizer.highlight_target_neuron(new_layer, new_point)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_target_lock(text: str):
        """
        Extracts layer index and vector point from the Phase 3 output.
        Matches: "TARGET LOCKED: Layer [X], Vector Point [Y]"
        Returns (layer_int, point_int) or (None, 0).
        """
        pattern = r"TARGET LOCKED:?\s*Layer\s*\[?(\d+)\]?,?\s*Vector\s*Point\s*\[?(\d+)\]?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Looser fallback: just grab first two integers after "TARGET LOCKED"
        loose = re.search(r"TARGET LOCKED[^\d]*(\d+)[^\d]+(\d+)", text, re.IGNORECASE)
        if loose:
            return int(loose.group(1)), int(loose.group(2))
        return None, 0

    def _spawn_terminal_windows(self):
        """Creates / recreates the three floating terminal windows."""
        for w in self.active_sub_windows:
            try:
                w.close()
            except Exception:
                pass
        self.active_sub_windows.clear()

        screen = self.screen()
        if screen:
            g = screen.availableGeometry()
            sw, sh = g.width(), g.height()
        else:
            sw, sh = 1920, 1080

        self.win_mind = OrderTerminalWindow(title="AGENT MIND TERMINAL", parent=self)
        self.win_logs = OrderTerminalWindow(title="DEEP CORE LOGS", parent=self)
        self.win_edit = OrderTerminalWindow(title="EDIT ENGINE", parent=self)

        self.win_mind.move(50, 50)
        self.win_mind.resize(520, 440)
        self.win_logs.move(sw - 580, 50)
        self.win_logs.resize(520, 440)
        self.win_edit.move(50, sh - 480)
        self.win_edit.resize(520, 420)

        self.active_sub_windows = [self.win_mind, self.win_logs, self.win_edit]
        for w in self.active_sub_windows:
            w.show()
            w.raise_()
            w.activateWindow()

    def _stop_all_threads(self):
        for t in (self.t_phase1, self.t_phase2, self.t_phase3, self.t_phase5):
            if t and t.isRunning():
                t.terminate()
                t.wait()

    def _on_error(self, error: str):
        self.feature_panel.update_readout(f"PIPELINE ERROR: {error}")
        gen_log.log_error(0, error)
        self._append_mind_log(f"\n[PIPELINE ERROR] {error}\n", "#ff003c")
        self._append_core_log(f"\n[PIPELINE ERROR] {error}\n", "#ff003c")

    def _append_mind_log(self, text: str, color: str = "#00f3ff"):
        if self.win_mind is not None:
            self.win_mind.append_text(text, color)
        else:
            self.feature_panel.update_readout(text.strip())

    def _append_core_log(self, text: str, color: str = "#00f3ff"):
        if self.win_logs is not None:
            self.win_logs.append_text(text, color)
        else:
            self.feature_panel.update_readout(text.strip())

    def _append_edit_log(self, text: str, color: str = "#00f3ff"):
        if self.win_edit is not None:
            self.win_edit.append_text(text, color)
        else:
            self.feature_panel.update_readout(text.strip())

    def _is_model_ready(self) -> Tuple[bool, str]:
        model_name = (self.active_model_name or self.backend.model_name or "").strip()
        if not model_name:
            return False, "No active model name in runtime state."
        if self.backend.model is None or self.backend.tokenizer is None:
            return False, "Model or tokenizer object is missing in memory."
        return True, "ready"

    @staticmethod
    def _extract_target(analysis: dict) -> Tuple[int, int]:
        return int(analysis.get("critical_layer_idx", -1)), int(analysis.get("critical_neuron", -1))

    @staticmethod
    def _pass4_io_referee(analyses: list[dict]) -> Optional[int]:
        """
        Pass 4 verifier: average real FFN input/output deltas across attempts
        and select the layer with maximum mean io_l2_delta.
        """
        if not analyses:
            return None
        layer_sums = {}
        layer_counts = {}
        for a in analyses:
            for row in a.get("layer_io_metrics", []):
                layer = int(row.get("layer", -1))
                score = float(row.get("io_l2_delta", 0.0))
                if layer < 0:
                    continue
                layer_sums[layer] = layer_sums.get(layer, 0.0) + score
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if not layer_sums:
            return None
        means = {
            layer: layer_sums[layer] / max(layer_counts.get(layer, 1), 1)
            for layer in layer_sums
        }
        candidates = {k: v for k, v in means.items() if k >= 1} or means
        return max(candidates, key=candidates.get)

    def closeEvent(self, event):
        self._stop_all_threads()
        gen_log.close_session(success=True)
        session_manager.close()
        for w in self.active_sub_windows:
            try:
                w.close()
            except Exception:
                pass
        event.accept()
