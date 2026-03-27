"""
pipeline_runner.py
==================
Qt-free pipeline orchestrator for the NeuroScalpel web interface.

Runs the same 5-phase pipeline as MainWindow but using
plain Python threading instead of QThread + pyqtSignal.

Usage (from Flask):
    runner = PipelineRunner(emit_fn=sse_queue.put)
    runner.start_word("The model thinks Paris is in Germany")
    runner.apply_rome()
"""

import re
import json
import logging
import threading
import time
import traceback
from typing import Callable, Optional, Tuple

from core.model_backend   import ModelManager, apply_real_edit
from core.session_manager import session_manager
from core.task_queue      import TaskQueue, EditTask
from core.generated_log   import gen_log
from core.point_and_layer_detect import CoreAnomalyDetector

from core.nvidia_agent import (
    DEEPSEEK_MODEL,
    default_chat_params_stream,
    nvidia_openai_client,
    stream_delta_reasoning_and_content,
)

logger = logging.getLogger("NeuroScalpel.PipelineRunner")

_PHASE1_SYSTEM = (
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

_PHASE3_SYSTEM = (
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


def _parse_target_lock(text: str):
    """Extract (layer_idx, vec_pt) from 'TARGET LOCKED: Layer X, Vector Point Y'."""
    m = re.search(
        r"TARGET LOCKED\s*:\s*Layer\s*\[?(\d+)\]?\s*,\s*Vector\s+Point\s*\[?(\d+)\]?",
        text, re.IGNORECASE
    )
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# ───────────────────────────────────────────────────────────────────────────
class PipelineRunner:
    """
    Thread-safe, Qt-free orchestrator for the 5-phase NeuroScalpel pipeline.

    Parameters
    ----------
    emit_fn : callable(dict)
        Called on every log / status event. The web layer puts these into
        an SSE queue.  Dict shape:
            {"type": "log",    "text": "...", "color": "#00f3ff"}
            {"type": "status", "key": "...", "value": ...}
            {"type": "done"}
    """

    def __init__(self, emit_fn: Callable[[dict], None]):
        self._emit = emit_fn
        self.backend = ModelManager()
        self.active_model_name: str = ""
        self.task_queue = TaskQueue()
        self.current_task: Optional[EditTask] = None
        self.last_order_text: str = ""
        self._lock = threading.Lock()

    @staticmethod
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
        # collapse double spaces (if any)
        return " ".join(m.split())

    # ── Public API ──────────────────────────────────────────────────────────

    def check_connection(self, result_cb: Callable[[bool, str], None]):
        """Async AI connection probe. Calls result_cb(online, message)."""
        def _run():
            try:
                client = nvidia_openai_client()
                client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                    timeout=15,
                    temperature=1,
                    top_p=0.95,
                )
                result_cb(True, "AI Helper ONLINE ✅")
            except Exception as e:
                result_cb(False, f"OFFLINE — {str(e)[:120]}")
        threading.Thread(target=_run, daemon=True).start()

    def load_local_model(self, path: str):
        """Load a local model directory. Non-blocking."""
        def _run():
            self._log(f"Loading local model: {path}", "#00f3ff")
            ok = self.backend.load_local_model(path, log_callback=self._log_cb)
            if ok:
                name = self.backend.model_name or path
                self.active_model_name = name
                self._emit({"type": "status", "key": "model_loaded", "value": name})
                self._log(f"Model '{name}' ready ✅", "#00ff88")
            else:
                self.active_model_name = ""
                self._log("Model load failed ❌", "#ff003c")
                self._emit({"type": "status", "key": "model_loaded", "value": ""})
        threading.Thread(target=_run, daemon=True).start()

    def load_hf_model(self, model_id: str):
        """Download + load a HuggingFace model. Non-blocking."""
        def _run():
            self._log(f"Connecting to HuggingFace: {model_id}", "#00f3ff")
            ok = self.backend.load_hf_model(model_id, log_callback=self._log_cb)
            if ok:
                name = self.backend.model_name or model_id
                self.active_model_name = name
                self._emit({"type": "status", "key": "model_loaded", "value": name})
                self._log(f"Model '{name}' ready ✅", "#00ff88")
            else:
                self.active_model_name = ""
                self._log("HuggingFace model load failed ❌", "#ff003c")
                self._emit({"type": "status", "key": "model_loaded", "value": ""})
        threading.Thread(target=_run, daemon=True).start()

    def start_word(self, order_text: str) -> bool:
        """Kick off the full 5-phase pipeline. Non-blocking."""
        if not order_text.strip():
            self._log("ERROR: Empty order text.", "#ff003c")
            return False
        ready, reason = self._is_model_ready()
        if not ready:
            self._log(
                f"ERROR: No model loaded.\n{reason}\n"
                "Load a model first before starting a pipeline run.",
                "#ff003c"
            )
            return False
        self.last_order_text = order_text
        self.task_queue = TaskQueue()
        self.current_task = None

        try:
            sid = session_manager.create_session(
                model_name=self.active_model_name or "unknown",
                order_text=order_text
            )
            self._emit({"type": "status", "key": "session_id", "value": sid})
            gen_log.start_session(sid, self.active_model_name or "unknown", order_text)
        except Exception as e:
            self._log(f"Session warning: {e}", "#ffaa00")

        threading.Thread(target=self._phase1, args=(order_text,), daemon=True).start()
        return True

    def apply_rome(self) -> bool:
        """Trigger Phase 5 neural edit on current task. Non-blocking."""
        if self.current_task is None:
            self._log("ERROR: No active task for neural edit.", "#ff003c")
            return False
        ready, reason = self._is_model_ready()
        if not ready:
            self._log(
                "ERROR: Model not loaded in memory.\n"
                f"{reason}", "#ff003c"
            )
            return False
        threading.Thread(target=self._phase5, daemon=True).start()
        return True

    # ── Phase 1 ─────────────────────────────────────────────────────────────

    def _phase1(self, order_text: str):
        self._log("\n>> PHASE 1: DIAGNOSIS INITIALIZED <<\n", "#00f3ff")
        self._log("> Analysing order for distinct issues...\n\n", "#0088aa")
        gen_log.log_phase_start(1, 0, 1)
        client = nvidia_openai_client()
        messages = [
            {"role": "system", "content": _PHASE1_SYSTEM},
            {"role": "user",   "content": f"USER ORDER:\n{order_text}"},
        ]
        attempts = 3
        for i in range(1, attempts + 1):
            try:
                self._log(
                    f"\n>> Surgeon Mind processing (attempt {i}/{attempts})...\n",
                    "#bc13fe",
                )
                completion = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    stream=True,
                    timeout=120,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=1200,
                    **default_chat_params_stream(),
                )
                self._log("\n>> Surgeon Mind processing...\n", "#bc13fe")
                content_chunks = []
                reasoning_chunks = []
                for chunk in completion:
                    reasoning, content = stream_delta_reasoning_and_content(chunk)
                    if reasoning is not None:
                        self._log(reasoning, "#6688aa")
                    if content is not None:
                        self._log(content, "#bc13fe")
                        content_chunks.append(content)
                    elif reasoning is not None:
                        reasoning_chunks.append(reasoning)
                raw = "".join(content_chunks).strip()
                if not raw:
                    raw = "".join(reasoning_chunks).strip()
                if not raw:
                    raise RuntimeError("Empty Phase 1 response.")
                self._log("\n\n>> JSON ready — Passing to Task Queue...\n", "#00f3ff")
                self._on_phase1_complete(raw)
                return
            except Exception as e:
                self._log(f"\n[PHASE 1 WARNING] attempt {i} failed: {e}\n", "#ffaa00")
                if i < attempts:
                    time.sleep(2 * i)
                    continue
                # Final fallback: non-stream request
                try:
                    self._log("\n[PHASE 1] fallback mode (non-stream) ...\n", "#ffaa00")
                    resp = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=messages,
                        stream=False,
                        timeout=90,
                        temperature=0.2,
                        top_p=0.95,
                        max_tokens=600,
                        **default_chat_params_stream(),
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    if raw:
                        self._log("\n>> JSON ready — Passing to Task Queue...\n", "#00f3ff")
                        self._on_phase1_complete(raw)
                        return
                    raise RuntimeError("Empty fallback response.")
                except Exception as final_e:
                    self._log(
                        f"\n[PHASE 1 CRITICAL FAILURE] {final_e}\n{traceback.format_exc()}\n",
                        "#ff003c",
                    )
                    return

    def _on_phase1_complete(self, raw_json: str):
        gen_log.log_phase_end(1, "Trick prompts formulated")
        tasks = self.task_queue.parse_from_phase1_response(raw_json)
        total = self.task_queue.total
        self._log(f"\n>> {total} task(s) queued.\n{self.task_queue.summary_text()}\n\n", "#00ff00")
        gen_log.log_tasks_parsed(tasks)

        for t in tasks:
            try:
                session_manager.log_task(
                    task_index=t.index,
                    trick_prompt=t.trick_prompt,
                    subject=t.subject,
                    wrong_value=t.wrong_value,
                    correct_value=t.correct_value,
                )
            except Exception:
                pass

        self._emit({"type": "status", "key": "task_total", "value": total})
        self._run_next_task()

    # ── Task runner ──────────────────────────────────────────────────────────

    def _run_next_task(self):
        if not self.task_queue.has_next():
            self._on_all_complete()
            return

        task = self.task_queue.pop_next()
        self.task_queue.advance()
        self.current_task = task
        pos   = self.task_queue.current_position - 1
        total = self.task_queue.total

        self._emit({"type": "status", "key": "task_current", "value": pos})
        self._emit({"type": "status", "key": "task_summary",
                    "value": self.task_queue.summary_text()})
        self._log(
            f"Task {pos}/{total}: {task.subject} | "
            f"{task.wrong_value} → {task.correct_value}", "#00f3ff"
        )
        try:
            session_manager.update_task_status(task.index, "running")
        except Exception:
            pass

        self._phase2(task)

    def _on_all_complete(self):
        self._log(
            "\n>> ALL TASKS COMPLETED ✅\nSession data saved to sessions/ folder.\n",
            "#00ff88"
        )
        self._emit({"type": "done"})

    # ── Phase 2 ─────────────────────────────────────────────────────────────

    def _phase2(self, task: EditTask):
        ready, reason = self._is_model_ready()
        if not ready:
            self._log(
                f"ERROR: No model loaded.\n{reason}\n"
                "Load a model first before starting a pipeline run.",
                "#ff003c"
            )
            return
        model = self.active_model_name or self.backend.model_name
        required_attempts = 3
        self._log(f"\n>> PHASE 2: NEURAL SCAN  [Task #{task.index + 1}] <<\n", "#00f3ff")
        gen_log.log_phase_start(2, task.index, self.task_queue.total)
        gen_log.log_phase_event(2, f"Scanning '{model}' with: {task.trick_prompt[:80]}")

        def run_attempts(forced_layer_idx: Optional[int] = None):
            analyses_local = []
            targets_local = []
            force_msg = f" (forced layer={forced_layer_idx})" if forced_layer_idx is not None else ""
            for attempt in range(1, required_attempts + 1):
                try:
                    self._log(
                        f"[VERIFY] Phase 2 target verification attempt {attempt}/{required_attempts}{force_msg}\n",
                        "#00f3ff",
                    )
                    detector = CoreAnomalyDetector(model_name=model)
                    if self.backend.model is not None and self.backend.tokenizer is not None:
                        detector.adopt_loaded_model(
                            model=self.backend.model,
                            tokenizer=self.backend.tokenizer,
                            model_name=self.backend.model_name or model,
                        )
                        self._log("[SYS] Reusing loaded model from memory.\n", "#00ff88")
                    else:
                        ok = detector.load_model(log_callback=self._log_cb)
                        if not ok:
                            self._log("[ERR] Model load failed. Aborting phase 2.\n", "#ff003c")
                            return None, None
                    detector.attach_hooks(log_callback=self._log_cb)
                    analysis = detector.probe_and_analyze(
                        task.trick_prompt,
                        forced_layer_idx=forced_layer_idx,
                        log_callback=self._log_cb,
                    )
                    if self.backend.model is None:
                        detector.cleanup()

                    if not analysis:
                        self._log("[ERR] Empty analysis in verification attempt.\n", "#ff003c")
                        return None, None
                    analysis["task_index"] = task.index
                    analyses_local.append(analysis)
                    target = self._extract_target(analysis)
                    targets_local.append(target)
                    self._log(
                        f"[VERIFY] Attempt {attempt} target -> layer={target[0]}, neuron={target[1]}\n",
                        "#0088aa",
                    )
                except Exception as e:
                    self._log(f"\n[PHASE 2 CRITICAL FAILURE] {e}\n{traceback.format_exc()}\n", "#ff003c")
                    return None, None
            return analyses_local, targets_local

        analyses, targets = run_attempts()
        if analyses is None:
            return

        if len(set(targets)) != 1:
            self._log(
                "[VERIFY] ERROR: strict 3/3 verification failed.\n"
                f"Observed targets: {targets}\n"
                "Stopping pipeline before edit.",
                "#ff003c",
            )
            if self.current_task:
                try:
                    session_manager.update_task_status(self.current_task.index, "failed")
                except Exception:
                    pass
            return

        io_ref_layer = self._pass4_io_referee(analyses)
        if io_ref_layer is None:
            self._log(
                "[VERIFY] ERROR: pass 4 (layer I/O referee) could not compute a valid layer.\n"
                "Stopping pipeline before edit.",
                "#ff003c",
            )
            if self.current_task:
                try:
                    session_manager.update_task_status(self.current_task.index, "failed")
                except Exception:
                    pass
            return
        if io_ref_layer != targets[0][0]:
            self._log(
                f"[VERIFY] Pass4 chose layer={io_ref_layer}; re-running 3/3 constrained to that layer.\n",
                "#ffaa00",
            )
            analyses2, targets2 = run_attempts(forced_layer_idx=io_ref_layer)
            if analyses2 is None:
                return
            if len(set(targets2)) != 1 or targets2[0][0] != io_ref_layer:
                self._log(
                    "[VERIFY] ERROR: pass4-constrained recheck failed.\n"
                    f"Observed targets: {targets2}\n"
                    "Stopping pipeline before edit.",
                    "#ff003c",
                )
                if self.current_task:
                    try:
                        session_manager.update_task_status(self.current_task.index, "failed")
                    except Exception:
                        pass
                return
            analyses, targets = analyses2, targets2

        self._log(
            f"\n>> Phase 2 Complete. 3/3 verification passed + Pass4 I/O passed (layer {io_ref_layer}).\n",
            "#00ff88"
        )
        self._on_phase2_complete(analyses[-1], locked_target=targets[0])

    def _on_phase2_complete(self, analysis: dict, locked_target: Optional[Tuple[int, int]] = None):
        task_idx = analysis.get("task_index", 0)
        crit      = analysis.get("critical_layer", "unknown")
        crit_idx  = analysis.get("critical_layer_idx", -1)
        neuron    = analysis.get("critical_neuron", -1)
        dev       = analysis.get("max_magnitude", 0.0)

        # Store neuron on the current task so Phase 5 can use it
        if locked_target:
            crit_idx, neuron = locked_target
            crit = f"layer.{crit_idx}"
        if self.current_task and neuron >= 0:
            self.current_task.target_point = neuron
        if self.current_task and crit_idx >= 0:
            self.current_task.target_layer = crit_idx

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
        gen_log.log_phase_end(2, f"Critical layer: {crit}  |  neuron: {neuron}  |  dev: {dev:.4f}")
        self._log(
            f"Phase 2 complete.\n"
            f"  Critical layer : {crit} (idx={crit_idx})\n"
            f"  Critical neuron: {neuron}\n"
            f"  Deviation      : {dev:.4f}",
            "#00f3ff"
        )

        # Emit target to UI immediately (Phase 4 pre-lock)
        if crit_idx >= 0:
            self._emit({"type": "status", "key": "target_layer",  "value": crit_idx})
        if neuron >= 0:
            self._emit({"type": "status", "key": "target_neuron", "value": neuron})

        self._phase3(analysis, locked_target=locked_target)

    # ── Phase 3 ─────────────────────────────────────────────────────────────

    def _phase3(self, analysis: dict, locked_target: Optional[Tuple[int, int]] = None):
        self._log("\n>> PHASE 3: POST-SCAN ANALYSIS + TARGET LOCK <<\n", "#00f3ff")
        self._log(">> Ingesting PyTorch telemetry...\n", "#0088aa")
        gen_log.log_phase_start(3,
            self.current_task.index if self.current_task else 0,
            self.task_queue.total
        )
        task_ctx = ""
        if self.current_task:
            task_ctx = (
                f"\nCURRENT TASK: Correct '{self.current_task.wrong_value}' → "
                f"'{self.current_task.correct_value}' for subject '{self.current_task.subject}'.\n"
            )
        raw_report = analysis.get("raw_report", "")
        if "=== FULL HIDDEN-STATE SCAN" in raw_report:
            raw_report = raw_report.split("=== FULL HIDDEN-STATE SCAN")[0].rstrip()

        user_msg = (
            f"USER ORDER: {self.last_order_text}{task_ctx}\n\n"
            f"RAW PYTORCH TENSOR LOG:\n{raw_report}"
        )

        client = nvidia_openai_client()
        attempts = 2
        for i in range(1, attempts + 1):
            try:
                completion = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": _PHASE3_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    stream=True,
                    timeout=120,
                    temperature=0.5,
                    top_p=0.95,
                    max_tokens=1200,
                    **default_chat_params_stream(),
                )
                self._log("\n[ THINKING PROTOCOL ENGAGED ]\n", "#333355")
                full = []
                reasoning_full = []
                for chunk in completion:
                    reasoning, content = stream_delta_reasoning_and_content(chunk)
                    if reasoning is not None and str(reasoning).strip():
                        self._log(reasoning, "#6688aa")
                        reasoning_full.append(reasoning)
                    if content is not None and str(content).strip():
                        color = "#ff2244" if "TARGET LOCKED" in content else "#00f3ff"
                        self._log(content, color)
                        full.append(content)
                full_response = "".join(full).strip()
                if not full_response:
                    full_response = "".join(reasoning_full).strip()
                if not full_response:
                    raise RuntimeError("Empty Phase 3 response.")
                self._log("\n\n>> OPERATION LOGIC SEQUENCE FULLY TERMINATED <<\n", "#bc13fe")
                self._on_phase3_complete(full_response, locked_target=locked_target)
                return
            except Exception as e:
                self._log(f"\n[PHASE 3 WARNING] attempt {i} failed: {e}\n", "#ffaa00")
                if i < attempts:
                    time.sleep(2 * i)
                    continue
                # Fallback to non-stream mode
                try:
                    self._log("\n[PHASE 3] fallback mode (non-stream) ...\n", "#ffaa00")
                    resp = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=[
                            {"role": "system", "content": _PHASE3_SYSTEM},
                            {"role": "user",   "content": user_msg},
                        ],
                        stream=False,
                        timeout=90,
                        temperature=0.2,
                        top_p=0.95,
                        max_tokens=600,
                        **default_chat_params_stream(),
                    )
                    msg = resp.choices[0].message
                    raw = (getattr(msg, "content", None) or "").strip()
                    if not raw:
                        raw = (getattr(msg, "reasoning_content", None) or "").strip()
                    if not raw:
                        raw = (getattr(msg, "reasoning", None) or "").strip()
                    if raw:
                        self._log("\n\n>> OPERATION LOGIC SEQUENCE FULLY TERMINATED <<\n", "#bc13fe")
                        self._on_phase3_complete(raw, locked_target=locked_target)
                        return
                    raise RuntimeError("Empty Phase 3 fallback response.")
                except Exception as final_e:
                    self._log(
                        f"\n[PHASE 3 CRITICAL FAILURE] {final_e}\n{traceback.format_exc()}\n",
                        "#ff003c",
                    )
                    return

    def _on_phase3_complete(self, full_response: str, locked_target: Optional[Tuple[int, int]] = None):
        parsed_layer, parsed_point = _parse_target_lock(full_response)
        if locked_target:
            layer_idx, vec_pt = locked_target
            if parsed_layer is not None and (parsed_layer, parsed_point) != locked_target:
                self._log(
                    f"[PHASE 3] Advisory mismatch ignored. Parsed {(parsed_layer, parsed_point)} "
                    f"but locked target is {locked_target}.\n",
                    "#ffaa00",
                )
        else:
            layer_idx, vec_pt = parsed_layer, parsed_point

        if layer_idx is not None and self.current_task:
            self.current_task.target_layer = layer_idx
            self.current_task.target_point = vec_pt
            self.task_queue.set_target(self.current_task.index, layer_idx, vec_pt)

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

            gen_log.log_target(layer_idx, vec_pt, full_response[-400:])
            gen_log.log_phase_end(3, f"TARGET LOCKED Layer {layer_idx} Neuron {vec_pt}")

            self._emit({"type": "status", "key": "target_layer",  "value": layer_idx})
            self._emit({"type": "status", "key": "target_neuron", "value": vec_pt})
            self._emit({"type": "status", "key": "rome_ready",    "value": True})
            self._log(
                f"\n>> PHASE 4: TARGET LOCKED\n   Layer {layer_idx} | Neuron {vec_pt} 🎯\n",
                "#ff2244"
            )

        self._log(
            f"Phase 3 complete.\nTarget: Layer {layer_idx}, Point {vec_pt}", "#00f3ff"
        )

    # ── Phase 5 ─────────────────────────────────────────────────────────────

    def _phase5(self):
        task = self.current_task
        neuron_hint = getattr(task, "target_point", -1)
        self._log(
            f"\n>> PHASE 5: NEURAL EDIT  [Task #{task.index + 1}] <<\n"
            f">> Subject      : {task.subject}\n"
            f">> Wrong Value  : {task.wrong_value}\n"
            f">> Correct Value: {task.correct_value}\n"
            f">> Prompt       : {task.trick_prompt}\n"
            f">> Target Layer : {task.target_layer}\n"
            f">> Target Neuron: {neuron_hint}\n",
            "#bc13fe"
        )
        gen_log.log_phase_start(5, task.index, self.task_queue.total)
        try:
            result = apply_real_edit(
                model_manager=self.backend,
                subject=task.subject,
                prompt_template=task.trick_prompt,
                target_new=task.correct_value,
                target_old=task.wrong_value,
                layer_hint=task.target_layer,
                neuron_hint=neuron_hint if neuron_hint >= 0 else None,
                log_callback=self._log_cb,
            )
            self._log(
                f"\n>> Phase 5 Complete: {self._display_method(result.get('method', ''))}\n",
                "#00ff00",
            )
            self._on_phase5_complete(result)
        except Exception as e:
            self._log(f"\n[PHASE 5 CRITICAL FAILURE] {e}\n{traceback.format_exc()}\n", "#ff003c")

    def _on_phase5_complete(self, result: dict):
        weights = result.get("weights", [])
        method  = result.get("method", "unknown")
        disp_method = self._display_method(method)
        success = result.get("success", False)
        notes   = result.get("notes", "")
        post_checks = result.get("post_checks", {})
        try:
            session_manager.log_edit(
                task_index=self.current_task.index if self.current_task else 0,
                method=method,
                weights_changed=json.dumps(weights),
                success=success,
                notes=notes,
            )
        except Exception:
            pass
        gen_log.log_edit(method, weights, success, notes)
        gen_log.log_phase_end(5, disp_method if success else "FAILED")
        self._emit({"type": "status", "key": "edit_result",
                    "value": {"success": success, "method": disp_method,
                               "weights": len(weights), "post_checks": post_checks}})
        self._log(
            f"Edit complete\nMethod: {disp_method}\nWeights modified: {len(weights)}",
            "#00f3ff",
        )
        if self.task_queue.has_next():
            self._log("\n>> Advancing to next queued task...\n", "#00f3ff")
            self._run_next_task()
        else:
            self._on_all_complete()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _log(self, text: str, color: str = "#00f3ff"):
        logger.info(text.strip())
        self._emit({"type": "log", "text": text, "color": color})

    def _log_cb(self, text: str, color: str = "#00f3ff"):
        """Callback-style shim compatible with ModelManager/CoreAnomalyDetector."""
        self._log(text, color)

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
        means = {k: layer_sums[k] / max(layer_counts.get(k, 1), 1) for k in layer_sums}
        candidates = {k: v for k, v in means.items() if k >= 1} or means
        return max(candidates, key=candidates.get)
