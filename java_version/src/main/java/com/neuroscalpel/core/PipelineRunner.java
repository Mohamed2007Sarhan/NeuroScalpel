package com.neuroscalpel.core;

import com.google.gson.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;

/**
 * PipelineRunner.java
 * ===================
 * Qt-free, JavaFX-compatible 5-phase pipeline orchestrator.
 * Uses Java 11 HttpClient for async API calls to the NVIDIA endpoint.
 *
 * Emits events via a BiConsumer<String, String> callback:
 *   ("log",    "#color")   → text to append to the terminal
 *   ("status", "key=value") → structured state update
 *   ("done",   "")         → all tasks finished
 */
public class PipelineRunner {

    private static final Logger log = LoggerFactory.getLogger(PipelineRunner.class);

    // ── NVIDIA / AI config ──────────────────────────────────────────────────
    private static final String BASE_URL    = "https://integrate.api.nvidia.com/v1";
    private static final String AI_MODEL    = "deepseek-ai/deepseek-v3.2";
    private static final String API_KEY     = System.getenv()
            .getOrDefault("NVIDIA_API_KEY", "YOUR_API_KEY_HERE");

    // ── State ───────────────────────────────────────────────────────────────
    private final ModelManager backend = new ModelManager();
    private final TaskQueue    taskQueue = new TaskQueue();
    private String  activeModelName = "";
    private String  lastOrderText   = "";
    private EditTask currentTask    = null;

    private final BiConsumer<String, String> emitFn;
    private final ExecutorService executor  = Executors.newVirtualThreadPerTaskExecutor();
    private final HttpClient httpClient;

    public PipelineRunner(BiConsumer<String, String> emitFn) {
        this.emitFn = emitFn;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    // ── Public API ──────────────────────────────────────────────────────────

    public ModelManager getBackend() { return backend; }
    public String getActiveModelName() { return activeModelName; }
    public EditTask getCurrentTask() { return currentTask; }

    public void checkConnection(BiConsumer<Boolean, String> cb) {
        executor.submit(() -> {
            try {
                callApi("ping", "ping", 5, false);
                cb.accept(true, "AI Helper ONLINE");
            } catch (Exception e) {
                cb.accept(false, "OFFLINE — " + e.getMessage());
            }
        });
    }

    public void loadLocalModel(String path) {
        executor.submit(() -> {
            emit("log", "> Loading local model: " + path + "\n", "#00f3ff");
            boolean ok = backend.loadLocal(path, this::emitLog);
            if (ok) {
                activeModelName = new java.io.File(path).getName();
                emitStatus("model_loaded", activeModelName);
                emit("log", "Model ready ✅\n", "#00ff88");
            } else {
                emit("log", "Model load failed ❌\n", "#ff003c");
            }
        });
    }

    public void loadHfModel(String modelId) {
        executor.submit(() -> {
            emit("log", "> Downloading from HuggingFace: " + modelId + "\n", "#00f3ff");
            boolean ok = backend.loadHuggingFace(modelId, this::emitLog);
            if (ok) {
                activeModelName = modelId;
                emitStatus("model_loaded", modelId);
                emit("log", "Model '" + modelId + "' ready ✅\n", "#00ff88");
            } else {
                emit("log", "HuggingFace load failed ❌\n", "#ff003c");
            }
        });
    }

    public void startWord(String orderText) {
        if (orderText.isBlank()) { emit("log", "ERROR: empty order.\n", "#ff003c"); return; }
        lastOrderText = orderText;
        taskQueue.reset();
        currentTask = null;
        executor.submit(() -> phase1(orderText));
    }

    public void applyEdit() {
        if (currentTask == null) { emit("log", "ERROR: no active task.\n", "#ff003c"); return; }
        if (!backend.isModelLoaded()) { emit("log", "ERROR: no model loaded.\n", "#ff003c"); return; }
        executor.submit(this::phase5);
    }

    // ── Phase 1: Diagnosis ──────────────────────────────────────────────────

    private void phase1(String orderText) {
        emit("log", "\n>> PHASE 1: DIAGNOSIS INITIALIZED <<\n", "#00f3ff");
        emit("log", "> Analysing order for distinct issues...\n\n", "#0088aa");

        String system = """
            You are the 'Surgeon Mind' embedded in NeuroScalpel.
            Identify every factual error in the user's message.
            Return ONLY a valid JSON array. Each element must have:
              "analysis", "trick_prompt", "subject", "wrong_value", "correct_value"
            No markdown, no code blocks.
            """;
        try {
            String raw = callApi(system, "USER ORDER:\n" + orderText, 4096, false);
            emit("log", "\n>> JSON ready — passing to Task Queue...\n", "#00f3ff");
            onPhase1Complete(raw);
        } catch (Exception e) {
            emit("log", "\n[PHASE 1 FAILURE] " + e.getMessage() + "\n", "#ff003c");
        }
    }

    private void onPhase1Complete(String rawJson) {
        List<EditTask> tasks = taskQueue.parseFromJson(rawJson);
        int total = taskQueue.getTotal();
        emit("log", "\n>> " + total + " task(s) queued.\n" + taskQueue.summaryText() + "\n\n", "#00ff00");
        emitStatus("task_total", String.valueOf(total));
        runNextTask();
    }

    // ── Phase 2: Neural Scan ────────────────────────────────────────────────

    private void runNextTask() {
        if (!taskQueue.hasNext()) { onAllComplete(); return; }
        currentTask = taskQueue.popNext();
        int pos = taskQueue.getCurrentPosition();
        emitStatus("task_current", String.valueOf(pos));
        emitStatus("task_summary", taskQueue.summaryText());
        emit("log", "\nTask " + pos + "/" + taskQueue.getTotal() + ": " +
             currentTask.getSubject() + "\n", "#00f3ff");
        phase2(currentTask);
    }

    private void phase2(EditTask task) {
        if (activeModelName.isBlank()) {
            emit("log", "ERROR: No model loaded. Load a model first.\n", "#ff003c");
            return;
        }
        emit("log", "\n>> PHASE 2: NEURAL SCAN  [Task #" + (task.getIndex()+1) + "] <<\n", "#00f3ff");
        try {
            AnomalyDetector detector = new AnomalyDetector(activeModelName);
            detector.loadModel(this::emitLog);
            detector.attachHooks(this::emitLog);
            Map<String, Object> analysis = detector.probeAndAnalyze(task.getTrickPrompt(), this::emitLog);
            detector.cleanup();
            analysis.put("task_index", task.getIndex());
            emit("log", "\n>> Phase 2 Complete. Handing to Agent...\n", "#00f3ff");
            onPhase2Complete(analysis);
        } catch (Exception e) {
            emit("log", "\n[PHASE 2 FAILURE] " + e.getMessage() + "\n", "#ff003c");
        }
    }

    private void onPhase2Complete(Map<String, Object> analysis) {
        String crit = (String) analysis.getOrDefault("critical_layer", "unknown");
        double dev  = (double) analysis.getOrDefault("max_magnitude", 0.0);
        emit("log", "Phase 2 complete. Critical layer: " + crit +
             "  |  dev: " + String.format("%.4f", dev) + "\n", "#00f3ff");
        phase3(analysis);
    }

    // ── Phase 3: Target Lock ────────────────────────────────────────────────

    private void phase3(Map<String, Object> analysis) {
        emit("log", "\n>> PHASE 3: POST-SCAN ANALYSIS + TARGET LOCK <<\n", "#00f3ff");
        String rawReport = (String) analysis.getOrDefault("raw_report", "");
        String system = """
            You are the 'Surgeon Mind' analytical core.
            Interpret the tensor telemetry to pinpoint the transformer layer and
            vector-space coordinate of the hallucinated concept.
            CRITICAL: You MUST end your response with EXACTLY this line:
            TARGET LOCKED: Layer [X], Vector Point [Y]
            """;
        String taskCtx = currentTask != null
            ? "\nCURRENT TASK: Correct '" + currentTask.getWrongValue() +
              "' → '" + currentTask.getCorrectValue() +
              "' for subject '" + currentTask.getSubject() + "'.\n"
            : "";
        String user = "USER ORDER: " + lastOrderText + taskCtx + "\n\nRAW PYTORCH TENSOR LOG:\n" + rawReport;

        try {
            String response = callApi(system, user, 4096, false);
            emit("log", "\n[ THINKING PROTOCOL ENGAGED ]\n", "#333355");
            String color = response.contains("TARGET LOCKED") ? "#ff2244" : "#00f3ff";
            emit("log", response + "\n", color);
            emit("log", "\n\n>> OPERATION LOGIC SEQUENCE FULLY TERMINATED <<\n", "#bc13fe");
            onPhase3Complete(response);
        } catch (Exception e) {
            emit("log", "\n[PHASE 3 FAILURE] " + e.getMessage() + "\n", "#ff003c");
        }
    }

    private void onPhase3Complete(String response) {
        int[] target = parseTargetLock(response);
        if (target != null && currentTask != null) {
            currentTask.setTargetLayer(target[0]);
            currentTask.setTargetPoint(target[1]);
            emitStatus("target_layer",  String.valueOf(target[0]));
            emitStatus("target_neuron", String.valueOf(target[1]));
            emitStatus("edit_ready",    "true");
            emit("log",
                "\n>> PHASE 4: TARGET LOCKED\n   Layer " + target[0] +
                " | Neuron " + target[1] + " 🎯\n",
                "#ff2244"
            );
        }
    }

    // ── Phase 5: Neural Edit ────────────────────────────────────────────────

    private void phase5() {
        EditTask task = currentTask;
        emit("log",
            "\n>> PHASE 5: NEURAL EDIT  [Task #" + (task.getIndex()+1) + "] <<\n" +
            ">> Subject      : " + task.getSubject() + "\n" +
            ">> Wrong Value  : " + task.getWrongValue() + "\n" +
            ">> Correct Value: " + task.getCorrectValue() + "\n" +
            ">> Target Layer : " + task.getTargetLayer() + "\n",
            "#bc13fe"
        );
        try {
            Map<String, Object> result = backend.applyEdit(task, this::emitLog);
            boolean success = (boolean) result.getOrDefault("success", false);
            String  method  = (String)  result.getOrDefault("method", "unknown");
            int     weights = (int)     result.getOrDefault("weights", 0);

            emit("log", "\n>> Phase 5 Complete: " + method + "\n", "#00ff00");
            emitStatus("edit_result",
                (success ? "success" : "failed") + "|" + method + "|" + weights);

            if (taskQueue.hasNext()) {
                emit("log", "\n>> Advancing to next queued task...\n", "#00f3ff");
                runNextTask();
            } else {
                onAllComplete();
            }
        } catch (Exception e) {
            emit("log", "\n[PHASE 5 FAILURE] " + e.getMessage() + "\n", "#ff003c");
            emitStatus("edit_ready", "true"); // re-enable button
        }
    }

    private void onAllComplete() {
        emit("log", "\n>> ALL TASKS COMPLETED ✅\nSession data saved.\n", "#00ff88");
        emitFn.accept("done", "");
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private String callApi(String system, String user, int maxTokens, boolean stream) throws Exception {
        var body = new JsonObject();
        body.addProperty("model", AI_MODEL);
        body.addProperty("temperature", 1.0);
        body.addProperty("max_tokens", maxTokens);

        var msgs = new JsonArray();
        var s = new JsonObject(); s.addProperty("role","system"); s.addProperty("content", system); msgs.add(s);
        var u = new JsonObject(); u.addProperty("role","user");   u.addProperty("content", user);   msgs.add(u);
        body.add("messages", msgs);

        var request = HttpRequest.newBuilder()
                .uri(URI.create(BASE_URL + "/chat/completions"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + API_KEY)
                .timeout(Duration.ofSeconds(120))
                .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
                .build();

        var response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200)
            throw new RuntimeException("API error " + response.statusCode() + ": " + response.body());

        var json    = JsonParser.parseString(response.body()).getAsJsonObject();
        var choices = json.getAsJsonArray("choices");
        if (choices == null || choices.isEmpty()) return "";
        return choices.get(0).getAsJsonObject()
                      .getAsJsonObject("message")
                      .get("content").getAsString().strip();
    }

    private int[] parseTargetLock(String text) {
        var m = java.util.regex.Pattern
            .compile("TARGET LOCKED\\s*:\\s*Layer\\s*\\[?(\\d+)\\]?\\s*,\\s*Vector\\s+Point\\s*\\[?(\\d+)\\]?",
                java.util.regex.Pattern.CASE_INSENSITIVE)
            .matcher(text);
        return m.find() ? new int[]{ Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)) } : null;
    }

    private void emit(String type, String text, String color) {
        if ("log".equals(type)) {
            log.info("[{}] {}", color, text.strip());
        }
        emitFn.accept(text, color);
    }

    private void emitLog(String text, String color) { emit("log", text, color); }

    private void emitStatus(String key, String value) {
        emitFn.accept("__status__:" + key + "=" + value, "");
    }
}
