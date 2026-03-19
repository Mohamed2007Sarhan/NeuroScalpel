"""
web/app.py
==========
Flask server for NeuroScalpel Web Interface.

Endpoints
---------
GET  /                  → serve index.html
GET  /api/status        → AI connection check (JSON)
POST /api/load_model    → load local or HuggingFace model
POST /api/start_word    → launch 5-phase pipeline
POST /api/apply_rome    → trigger Phase 5 ROME edit
GET  /api/session       → current session summary
GET  /stream/pipeline   → Server-Sent Events stream (live logs)
"""

import queue
import json
import time
import threading
import logging
from flask import Flask, Response, request, jsonify, render_template, stream_with_context

from core.pipeline_runner import PipelineRunner

logger = logging.getLogger("NeuroScalpel.WebApp")


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "neuroscalpel-web"

    # ── Global state ────────────────────────────────────────────────────────
    # One SSE queue per connected client (stored by client id)
    _client_queues: dict[str, queue.Queue] = {}
    _client_lock = threading.Lock()

    # Single shared pipeline runner whose emit broadcasts to all SSE clients
    def _broadcast(event: dict):
        with _client_lock:
            dead = []
            for cid, q in _client_queues.items():
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(cid)
            for cid in dead:
                del _client_queues[cid]

    runner = PipelineRunner(emit_fn=_broadcast)

    # ── Pages ───────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ── SSE stream ───────────────────────────────────────────────────────────
    @app.route("/stream/pipeline")
    def stream_pipeline():
        cid = str(time.time_ns())
        q: queue.Queue = queue.Queue(maxsize=500)
        with _client_lock:
            _client_queues[cid] = q

        def generate():
            # Send a heartbeat first so the browser knows the connection is open
            yield "event: heartbeat\ndata: connected\n\n"
            try:
                while True:
                    try:
                        event = q.get(timeout=20)
                        yield f"data: {json.dumps(event)}\n\n"
                    except queue.Empty:
                        yield "event: heartbeat\ndata: ping\n\n"
            except GeneratorExit:
                pass
            finally:
                with _client_lock:
                    _client_queues.pop(cid, None)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── REST: AI status check ────────────────────────────────────────────────
    @app.route("/api/status")
    def api_status():
        result = {"online": None, "message": "Checking..."}
        done_ev = threading.Event()

        def cb(online, msg):
            result["online"] = online
            result["message"] = msg
            done_ev.set()

        runner.check_connection(cb)
        done_ev.wait(timeout=20)
        return jsonify(result)

    # ── REST: Load model ─────────────────────────────────────────────────────
    @app.route("/api/load_model", methods=["POST"])
    def api_load_model():
        data = request.get_json(force=True)
        mode = data.get("mode", "hf")       # "local" | "hf"
        path = data.get("path", "").strip()

        if not path:
            return jsonify({"error": "Missing path or model_id"}), 400

        if mode == "local":
            runner.load_local_model(path)
        else:
            runner.load_hf_model(path)

        return jsonify({"status": "loading", "target": path})

    # ── REST: Start pipeline ─────────────────────────────────────────────────
    @app.route("/api/start_word", methods=["POST"])
    def api_start_word():
        data = request.get_json(force=True)
        order = data.get("order", "").strip()
        if not order:
            return jsonify({"error": "Empty order text"}), 400
        runner.start_word(order)
        return jsonify({"status": "pipeline_started"})

    # ── REST: Apply ROME edit ────────────────────────────────────────────────
    @app.route("/api/apply_rome", methods=["POST"])
    def api_apply_rome():
        runner.apply_rome()
        return jsonify({"status": "edit_started"})

    # ── REST: Session info ───────────────────────────────────────────────────
    @app.route("/api/session")
    def api_session():
        try:
            from core.session_manager import session_manager
            summary = session_manager.get_session_summary()
            return jsonify(summary)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── REST: Model 3D geometry ──────────────────────────────────────────────
    @app.route("/api/model_geometry")
    def api_model_geometry():
        """
        Returns the 3D layer-neuron geometry for the currently loaded model.
        Shape: { points: [[x,y,z],...], labels: [layer_idx,...], layer_map: {layer: [indices...]} }
        Falls back to flat PCA embedding if no model is in memory.
        """
        try:
            backend = runner.backend
            if backend.model is not None:
                pts, labels, lmap = backend.get_layer_neuron_geometry(
                    num_neurons_per_layer=80
                )
                return jsonify({
                    "mode": "layered",
                    "points": pts.tolist(),
                    "labels": labels.tolist(),
                    "layer_map": {str(k): v for k, v in lmap.items()},
                    "num_layers": int(labels.max()) + 1 if len(labels) else 0,
                })
            elif backend.model_name:
                pts, ids = backend.get_real_weights(
                    model_id=backend.model_name, num_points=1500
                )
                return jsonify({
                    "mode": "flat",
                    "points": pts.tolist(),
                    "labels": ids.tolist(),
                    "layer_map": {},
                    "num_layers": 1,
                })
            else:
                return jsonify({"mode": "empty", "points": [], "labels": [], "layer_map": {}, "num_layers": 0})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── REST: Highlight target neuron ─────────────────────────────────────────
    @app.route("/api/highlight_target")
    def api_highlight_target():
        """Returns the currently locked layer + neuron for 3D highlighting."""
        task = runner.current_task
        if task and task.target_layer is not None:
            return jsonify({
                "layer":  task.target_layer,
                "neuron": getattr(task, "target_point", 0) or 0,
            })
        return jsonify({"layer": None, "neuron": None})

    return app

