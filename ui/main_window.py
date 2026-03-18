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
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from core.model_backend import ModelManager, apply_real_edit
from core.session_manager import session_manager
from core.task_queue import TaskQueue, EditTask
from core.point_and_layer_detect import CoreAnomalyDetector
from ui.panels.feature_extractor import FeatureExtractorPanel
from ui.panels.dashboard import DashboardPanel
from ui.visualizer.point_cloud_3d import PointCloud3DWidget
from ui.panels.order_terminal import OrderTerminalWindow

from openai import OpenAI

logger = logging.getLogger("NeuroScalpel.MainWindow")

# ---------------------------------------------------------------------------
# NVIDIA / DeepSeek client factory
# ---------------------------------------------------------------------------
_NVIDIA_API_KEY = "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"
_DEEPSEEK_MODEL = "deepseek-ai/deepseek-v3.2"
_BASE_URL       = "https://integrate.api.nvidia.com/v1"


def _nvidia_client() -> OpenAI:
    return OpenAI(base_url=_BASE_URL, api_key=_NVIDIA_API_KEY)


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
        self.stream_mind.emit(">> Analysing order for distinct issues...\n\n", "#0088aa")
        try:
            client = _nvidia_client()
            completion = client.chat.completions.create(
                model=_DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"USER ORDER:\n{self.order_text}"}
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            raw = completion.choices[0].message.content.strip()
            self.stream_mind.emit(
                f">> Phase 1 Response:\n{raw}\n\n>> Passing to Task Queue...\n",
                "#bc13fe"
            )
            self.phase1_complete.emit(raw)
        except Exception as e:
            msg = f"\n[PHASE 1 CRITICAL FAILURE] {e}\n{traceback.format_exc()}\n"
            self.stream_mind.emit(msg, "#ff003c")
            self.error_occurred.emit(str(e))


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
                 task_index: int = 0, parent=None):
        super().__init__(parent)
        self.trick_prompt = trick_prompt
        self.model_name   = model_name
        self.task_index   = task_index

    def run(self):
        try:
            self.stream_core.emit(
                f"\n>> PHASE 2: NEURAL SCAN  [Task #{self.task_index + 1}] <<\n", "#00f3ff"
            )
            detector = CoreAnomalyDetector(model_name=self.model_name)

            ok = detector.load_model(log_callback=self.stream_core.emit)
            if not ok:
                self.stream_core.emit("[ERR] Model load failed. Aborting phase 2.\n", "#ff003c")
                self.error_occurred.emit("Model load failure.")
                return

            detector.attach_hooks(log_callback=self.stream_core.emit)
            analysis = detector.probe_and_analyze(self.trick_prompt,
                                                   log_callback=self.stream_core.emit)
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
        "a forward pass of the target LLM. Your objective: mechanically interpret these "
        "logs to pinpoint the exact transformer layer and vector-space coordinate where "
        "the hallucinated concept is formed.\n\n"
        "Provide a concise, technically precise breakdown of the tensor deviations observed.\n\n"
        "CRITICAL DIRECTIVE — PHASE 4 TARGET LOCK:\n"
        "You MUST conclude your entire response with this EXACT line and nothing after it:\n"
        "TARGET LOCKED: Layer [X], Vector Point [Y]\n"
        "Where X = integer layer index (from the deviation report) and "
        "Y = integer vector point index (use the token position with highest L2 norm or deviation). "
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
            self.stream_mind.emit(
                "\n>> PHASE 3: POST-SCAN ANALYSIS + TARGET LOCK <<\n", "#00f3ff"
            )
            self.stream_mind.emit(
                ">> Ingesting PyTorch telemetry. Executing cognitive sub-routines...\n",
                "#0088aa"
            )
            client = _nvidia_client()

            # Build user message
            task_context = ""
            if self.task:
                task_context = (
                    f"\nCURRENT TASK: Correct '{self.task.wrong_value}' → '{self.task.correct_value}' "
                    f"for subject '{self.task.subject}'.\n"
                )

            user_msg = (
                f"USER ORDER: {self.order_text}{task_context}\n\n"
                f"RAW PYTORCH TENSOR LOG:\n{self.analysis_dict.get('raw_report', '')}"
            )

            completion = client.chat.completions.create(
                model=_DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg}
                ],
                temperature=1,
                top_p=0.95,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )

            self.stream_mind.emit("\n[ THINKING PROTOCOL ENGAGED ]\n", "#333355")

            full_text = []
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    self.stream_mind.emit(reasoning, "#0055aa")
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    is_locked = "TARGET LOCKED" in content
                    color = "#ff2244" if is_locked else "#00f3ff"
                    self.stream_mind.emit(content, color)
                    full_text.append(content)

            full_response = "".join(full_text)
            self.stream_mind.emit(
                "\n\n>> OPERATION LOGIC SEQUENCE FULLY TERMINATED <<\n", "#bc13fe"
            )
            self.phase3_complete.emit(full_response)

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
        self.stream_edit.emit(
            f"\n>> PHASE 5: ROME + LyapLock EDIT  [Task #{self.task.index + 1}] <<\n",
            "#bc13fe"
        )
        self.stream_edit.emit(
            f">> Subject      : {self.task.subject}\n"
            f">> Wrong Value  : {self.task.wrong_value}\n"
            f">> Correct Value: {self.task.correct_value}\n"
            f">> Prompt       : {self.task.trick_prompt}\n"
            f">> Target Layer : {self.task.target_layer}\n",
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
                log_callback=self.stream_edit.emit
            )
            self.stream_edit.emit(
                f"\n>> Phase 5 Complete: {result['method']}\n", "#00ff00"
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
        self.visualizer.pointSelected.connect(self.dashboard.set_selected_point)
        self.visualizer.pointMoved.connect(self._on_point_dragged)
        self.visualizer.targetNeuronVisualized.connect(self._on_target_visualized)

        # Dashboard
        self.dashboard.updateRequested.connect(self._on_update_weights_requested)
        self.dashboard.startWordRequested.connect(self._on_start_word_requested)
        self.dashboard.applyRomeRequested.connect(self._on_apply_rome_requested)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _on_model_loaded(self, model_id: str):
        self.active_model_name = model_id
        self.feature_panel.update_readout(
            f"Model loaded: {model_id}\nExtracting layered 3D geometry..."
        )

        # Try layered geometry first (requires model in memory)
        if self.backend.model is not None:
            pts, labels, lmap = self.backend.get_layer_neuron_geometry(
                num_neurons_per_layer=80,
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
            model_id=model_id, num_points=2500,
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

        self.last_order_text = order_text
        self._stop_all_threads()
        self.visualizer.reset_target()
        self.dashboard.clear_target_lock()

        # Create session in DB
        try:
            session_id = session_manager.create_session(
                model_name=self.active_model_name or "unknown",
                order_text=order_text
            )
            self.feature_panel.update_readout(f"Session created: {session_id}")
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
        self.feature_panel.update_readout("Phase 1: Sending to Surgeon Mind...")
        self.t_phase1 = Phase1Thread(order_text, parent=self)
        self.t_phase1.stream_mind.connect(self.win_mind.append_text)
        self.t_phase1.phase1_complete.connect(self._on_phase1_complete)
        self.t_phase1.error_occurred.connect(self._on_error)
        self.t_phase1.start()

    def _on_phase1_complete(self, raw_json: str):
        """Parse tasks from DeepSeek JSON response."""
        tasks = self.task_queue.parse_from_phase1_response(raw_json)
        total = self.task_queue.total

        self.win_mind.append_text(
            f"\n>> {total} task(s) queued.\n{self.task_queue.summary_text()}\n\n",
            "#00ff00"
        )

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

    # ------------------------------------------------------------------
    # PHASE 2
    # ------------------------------------------------------------------

    def _start_phase2(self, task: EditTask):
        model = self.active_model_name or "openai-community/gpt2"
        self.feature_panel.update_readout(
            f"Phase 2: Neural scan for task #{task.index + 1}..."
        )
        self.t_phase2 = Phase2Thread(
            trick_prompt=task.trick_prompt,
            model_name=model,
            task_index=task.index,
            parent=self
        )
        self.t_phase2.stream_core.connect(self.win_logs.append_text)
        self.t_phase2.phase2_complete.connect(self._on_phase2_complete)
        self.t_phase2.error_occurred.connect(self._on_error)
        self.t_phase2.start()

    def _on_phase2_complete(self, analysis: dict):
        task_idx = analysis.get("task_index", 0)

        # Persist scan result
        try:
            session_manager.log_scan_result(
                task_index=task_idx,
                critical_layer=analysis.get("critical_layer", ""),
                max_deviation=analysis.get("max_magnitude", 0.0),
                raw_report=analysis.get("raw_report", "")
            )
        except Exception:
            pass

        self.feature_panel.update_readout(
            f"Phase 2 complete.\nCritical layer: {analysis.get('critical_layer')}\n"
            f"Max deviation: {analysis.get('max_magnitude', 0):.4f}"
        )
        self._start_phase3(analysis)

    # ------------------------------------------------------------------
    # PHASE 3
    # ------------------------------------------------------------------

    def _start_phase3(self, analysis: dict):
        self.feature_panel.update_readout("Phase 3: Target Lock analysis...")
        self.t_phase3 = Phase3Thread(
            order_text=self.last_order_text,
            analysis_dict=analysis,
            task=self.current_task,
            parent=self
        )
        self.t_phase3.stream_mind.connect(self.win_mind.append_text)
        self.t_phase3.phase3_complete.connect(self._on_phase3_complete)
        self.t_phase3.error_occurred.connect(self._on_error)
        self.t_phase3.start()

    def _on_phase3_complete(self, full_response: str):
        """
        Parse "TARGET LOCKED: Layer [X], Vector Point [Y]" from Phase 3 output.
        Trigger Phase 4 (visualisation) and enable ROME edit button.
        """
        layer_idx, vec_pt = self._parse_target_lock(full_response)

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
    # PHASE 5 — ROME + LyapLock Edit
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
            self.dashboard.btn_rome_edit.setText("🧬  APPLY ROME EDIT")
            return

        self.feature_panel.update_readout(
            f"Phase 5: Applying ROME + LyapLock edit for task #{self.current_task.index + 1}..."
        )

        self.t_phase5 = Phase5EditThread(
            model_manager=self.backend,
            task=self.current_task,
            parent=self
        )
        self.t_phase5.stream_edit.connect(self.win_edit.append_text)
        self.t_phase5.phase5_complete.connect(self._on_phase5_complete)
        self.t_phase5.error_occurred.connect(self._on_phase5_error)
        self.t_phase5.start()

    def _on_phase5_complete(self, result: dict):
        self.dashboard.btn_rome_edit.setText("✅  EDIT APPLIED")

        try:
            session_manager.log_edit(
                task_index=self.current_task.index if self.current_task else 0,
                method=result.get("method", "unknown"),
                weights_changed=json.dumps(result.get("weights", [])),
                success=result.get("success", False),
                notes=result.get("notes", "")
            )
        except Exception:
            pass

        self.feature_panel.update_readout(
            f"Edit complete ✅\nMethod: {result.get('method')}\n"
            f"Weights modified: {len(result.get('weights', []))}"
        )
        self.win_edit.append_text(
            f"\n>> EDIT COMPLETE: {result.get('method')}\n", "#00ff00"
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
        self.dashboard.btn_rome_edit.setText("🧬  APPLY ROME EDIT")
        self.feature_panel.update_readout(f"Edit failed: {error}")

    # ------------------------------------------------------------------
    # Manual weight drag (legacy dashboard feature)
    # ------------------------------------------------------------------

    def _on_point_dragged(self, point_id: int, coords: tuple):
        self.dashboard.set_selected_point(point_id, coords)

    def _on_update_weights_requested(self, point_id: int, new_coords: tuple, bias: tuple):
        self.feature_panel.update_readout(
            f"Manual bias update → Point P{point_id}\nCoords: {new_coords}\nBias: {bias}"
        )
        self.visualizer.update_point_position(point_id, new_coords)

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

    def _stop_all_threads(self):
        for t in (self.t_phase1, self.t_phase2, self.t_phase3, self.t_phase5):
            if t and t.isRunning():
                t.terminate()
                t.wait()

    def _on_error(self, error: str):
        self.feature_panel.update_readout(f"PIPELINE ERROR: {error}")

    def closeEvent(self, event):
        self._stop_all_threads()
        session_manager.close()
        for w in self.active_sub_windows:
            try:
                w.close()
            except Exception:
                pass
        event.accept()
