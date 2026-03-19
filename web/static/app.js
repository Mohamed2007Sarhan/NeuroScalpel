/**
 * app.js — NeuroScalpel Web Interface
 * Handles SSE streaming, REST API calls, UI state, and Three.js 3D neural visualizer.
 */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────
const termLog           = document.getElementById("termLog");
const connDot           = document.getElementById("connDot");
const connLabel         = document.getElementById("connLabel");
const badgeDot          = document.getElementById("badgeDot");
const badgeText         = document.getElementById("badgeText");
const sessionLabel      = document.getElementById("sessionLabel");
const taskProgressLabel = document.getElementById("taskProgressLabel");
const taskProgressWrap  = document.getElementById("taskProgressWrap");
const taskProgressBar   = document.getElementById("taskProgressBar");
const taskList          = document.getElementById("taskList");
const targetDisplay     = document.getElementById("targetDisplay");
const btnRomeEdit       = document.getElementById("btnRomeEdit");
const modelNameLabel    = document.getElementById("modelNameLabel");
const btnStartWord      = document.getElementById("btnStartWord");
const orderInput        = document.getElementById("orderInput");
const localPathInput    = document.getElementById("localPathInput");
const progressWrap      = document.getElementById("progressWrap");
const progressBar       = document.getElementById("progressBar");
const toast             = document.getElementById("toast");
const vizCanvas         = document.getElementById("vizCanvas");
const vizCanvasWrap     = document.getElementById("vizCanvasWrap");
const vizOverlay        = document.getElementById("vizOverlay");
const vizHint           = document.getElementById("vizHint");
const btnLoad3D         = document.getElementById("btnLoad3D");

// ── State ─────────────────────────────────────────────────────────────────
let taskTotal   = 0;
let taskCurrent = 0;
let romeReady   = false;
let progressAnim = null;

// ── Toast ─────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = "info") {
  toast.textContent = msg;
  toast.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
}

// ── Terminal log ──────────────────────────────────────────────────────────
function appendLog(text, color = "#a8b8cc") {
  const span = document.createElement("span");
  span.style.color = color;
  span.textContent = text;
  termLog.appendChild(span);
  termLog.scrollTop = termLog.scrollHeight;
}

// ── Connection status helpers ─────────────────────────────────────────────
function setConnStatus(state, label) {
  connDot.className  = `pill-dot ${state}`;
  connLabel.textContent = label;
  badgeDot.className = `badge-dot ${state}`;
  badgeText.textContent =
    state === "online"  ? `🟢 ${label}` :
    state === "offline" ? `🔴 ${label}` : `🟡 ${label}`;
}

// ═══════════════════════════════════════════════════════════════════════════
//  THREE.JS 3D NEURAL VISUALIZER
// ═══════════════════════════════════════════════════════════════════════════

let g = {
  scene: null, camera: null, renderer: null,
  pointMesh: null, targetMarker: null,
  animFrameId: null, isDragging: false,
  lastX: 0, lastY: 0,
  rotX: 0.3, rotY: 0,
  zoom: 1.0,
};

// Layer colour gradient: cyan (early) → purple (mid) → red/orange (deep)
function layerColor(fraction) {
  // fraction: 0 = early layer, 1 = last layer
  const stops = [
    [0.00, 0x00, 0xf3, 0xff],  // #00f3ff cyan
    [0.40, 0x00, 0xaa, 0xff],  // mid blue
    [0.65, 0xbc, 0x13, 0xfe],  // #bc13fe purple
    [0.85, 0xff, 0x88, 0x00],  // orange
    [1.00, 0x00, 0xff, 0x88],  // #00ff88 green (final layer)
  ];
  let s0 = stops[0], s1 = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (fraction >= stops[i][0] && fraction <= stops[i + 1][0]) {
      s0 = stops[i]; s1 = stops[i + 1]; break;
    }
  }
  const t = s0[0] === s1[0] ? 0 : (fraction - s0[0]) / (s1[0] - s0[0]);
  const r = Math.round(s0[1] + (s1[1] - s0[1]) * t);
  const g2 = Math.round(s0[2] + (s1[2] - s0[2]) * t);
  const b = Math.round(s0[3] + (s1[3] - s0[3]) * t);
  return (r << 16) | (g2 << 8) | b;
}

function initThree() {
  if (g.renderer) return; // already initialized

  const W = vizCanvasWrap.clientWidth;
  const H = vizCanvasWrap.clientHeight;

  g.scene    = new THREE.Scene();
  g.camera   = new THREE.PerspectiveCamera(50, W / H, 0.1, 1000);
  g.camera.position.set(0, 0, 30);

  g.renderer = new THREE.WebGLRenderer({ canvas: vizCanvas, antialias: true, alpha: true });
  g.renderer.setPixelRatio(window.devicePixelRatio);
  g.renderer.setSize(W, H);
  g.renderer.setClearColor(0x000000, 0);

  // Ambient + point lights for subtle depth shading
  g.scene.add(new THREE.AmbientLight(0x00f3ff, 0.15));
  const pl = new THREE.PointLight(0xbc13fe, 0.8, 100);
  pl.position.set(10, 10, 10);
  g.scene.add(pl);

  // Grid plane reference
  const grid = new THREE.GridHelper(40, 20, 0x00f3ff, 0x0a0f20);
  grid.position.y = -12;
  grid.material.opacity = 0.18;
  grid.material.transparent = true;
  g.scene.add(grid);

  // Rotation via mouse drag
  vizCanvasWrap.addEventListener("mousedown", (e) => {
    g.isDragging = true; g.lastX = e.clientX; g.lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => { g.isDragging = false; });
  vizCanvasWrap.addEventListener("mousemove", (e) => {
    if (!g.isDragging) return;
    const dx = e.clientX - g.lastX;
    const dy = e.clientY - g.lastY;
    g.rotY += dx * 0.006;
    g.rotX += dy * 0.004;
    g.rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, g.rotX));
    g.lastX = e.clientX; g.lastY = e.clientY;
  });
  vizCanvasWrap.addEventListener("wheel", (e) => {
    g.zoom *= e.deltaY > 0 ? 1.08 : 0.93;
    g.zoom = Math.max(0.4, Math.min(4.0, g.zoom));
  });

  // Handle resize
  const ro = new ResizeObserver(() => {
    const W2 = vizCanvasWrap.clientWidth;
    const H2 = vizCanvasWrap.clientHeight;
    g.renderer.setSize(W2, H2);
    g.camera.aspect = W2 / H2;
    g.camera.updateProjectionMatrix();
  });
  ro.observe(vizCanvasWrap);

  animate();
}

function animate() {
  g.animFrameId = requestAnimationFrame(animate);

  if (g.pointMesh) {
    // Gentle auto-rotate when not dragging
    if (!g.isDragging) g.rotY += 0.0015;

    g.pointMesh.rotation.x = g.rotX;
    g.pointMesh.rotation.y = g.rotY;

    if (g.targetMarker) {
      g.targetMarker.rotation.x = g.rotX;
      g.targetMarker.rotation.y = g.rotY;
      // Pulsing scale
      const pulse = 1 + 0.25 * Math.sin(Date.now() / 250);
      g.targetMarker.scale.setScalar(pulse);
    }
  }

  g.camera.position.z = 30 * g.zoom;
  g.renderer.render(g.scene, g.camera);
}

function loadPointCloud(data) {
  // Remove old mesh
  if (g.pointMesh) { g.scene.remove(g.pointMesh); g.pointMesh = null; }
  if (g.targetMarker) { g.scene.remove(g.targetMarker); g.targetMarker = null; }

  const pts = data.points;
  const labels = data.labels;
  const numLayers = data.num_layers || 1;

  if (!pts || pts.length === 0) return;

  const geom   = new THREE.BufferGeometry();
  const verts  = new Float32Array(pts.length * 3);
  const colors = new Float32Array(pts.length * 3);

  const c = new THREE.Color();
  for (let i = 0; i < pts.length; i++) {
    verts[i * 3]     = pts[i][0];
    verts[i * 3 + 1] = pts[i][1];
    verts[i * 3 + 2] = pts[i][2];

    const frac = numLayers > 1 ? labels[i] / (numLayers - 1) : 0.5;
    c.setHex(layerColor(frac));
    colors[i * 3]     = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geom.setAttribute("color",    new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.18,
    vertexColors: true,
    transparent: true,
    opacity: 0.88,
    sizeAttenuation: true,
  });

  g.pointMesh = new THREE.Points(geom, mat);
  g.scene.add(g.pointMesh);

  // Hide overlay
  vizOverlay.style.opacity = "0";
  setTimeout(() => { vizOverlay.style.display = "none"; }, 600);

  vizHint.textContent = `${pts.length} neurons — ${numLayers} layers — drag to rotate`;
}

function highlightTargetNeuron(layerIdx, lmap, pts) {
  if (g.targetMarker) { g.scene.remove(g.targetMarker); g.targetMarker = null; }
  if (!pts || !lmap) return;

  const layerKey = String(layerIdx);
  const indices = lmap[layerKey];
  if (!indices || indices.length === 0) return;

  // Use centroid of the target layer's neurons as marker position
  let cx = 0, cy = 0, cz = 0;
  for (const idx of indices) {
    cx += pts[idx][0]; cy += pts[idx][1]; cz += pts[idx][2];
  }
  cx /= indices.length; cy /= indices.length; cz /= indices.length;

  const sphGeom = new THREE.SphereGeometry(0.55, 12, 12);
  const sphMat  = new THREE.MeshBasicMaterial({
    color: 0xff2244, transparent: true, opacity: 0.85,
  });
  g.targetMarker = new THREE.Mesh(sphGeom, sphMat);
  g.targetMarker.position.set(cx, cy, cz);
  // Inherit current rotation
  g.targetMarker.rotation.x = g.rotX;
  g.targetMarker.rotation.y = g.rotY;

  // Glow ring
  const ringGeom = new THREE.TorusGeometry(0.9, 0.06, 8, 32);
  const ringMat  = new THREE.MeshBasicMaterial({ color: 0xff2244 });
  const ring = new THREE.Mesh(ringGeom, ringMat);
  g.targetMarker.add(ring);

  if (g.pointMesh) {
    g.pointMesh.add(g.targetMarker);
  } else {
    g.scene.add(g.targetMarker);
  }
}

// ── Fetch + render 3D geometry ────────────────────────────────────────────
let _lastGeomData = null;

async function fetchAndRender3D() {
  vizHint.textContent = "Extracting geometry…";
  try {
    const res = await fetch("/api/model_geometry");
    const data = await res.json();
    if (data.error) { vizHint.textContent = `Error: ${data.error}`; return; }
    if (data.mode === "empty") { vizHint.textContent = "No model loaded"; return; }
    _lastGeomData = data;
    initThree();
    loadPointCloud(data);
  } catch (e) {
    vizHint.textContent = `3D error: ${e.message}`;
  }
}

// ── SSE event handling ─────────────────────────────────────────────────────
function handleEvent(evt) {
  if (evt.type === "log") {
    appendLog(evt.text, evt.color || "#a8b8cc");
    return;
  }
  if (evt.type === "done") {
    showToast("All tasks complete! ✅", "success");
    btnStartWord.disabled = false;
    return;
  }
  if (evt.type !== "status") return;

  const { key, value } = evt;

  if (key === "session_id") {
    sessionLabel.textContent = `SESSION: ${String(value).substring(0, 20)}…`;
  }
  if (key === "model_loaded") {
    modelNameLabel.textContent = value;
    showToast(`Model loaded: ${value}`, "success");
    stopProgressAnim();
    btnLoad3D.style.display = "block";
    // Auto-render after model loads
    setTimeout(fetchAndRender3D, 500);
  }
  if (key === "task_total") {
    taskTotal = value;
    updateTaskBar(0, value);
  }
  if (key === "task_current") {
    taskCurrent = value;
    updateTaskBar(value, taskTotal);
  }
  if (key === "task_summary") {
    taskList.textContent = value;
    taskProgressWrap.style.display = "block";
  }
  if (key === "target_layer")  window._tLayer  = value;
  if (key === "target_neuron") window._tNeuron = value;

  if (key === "target_layer" || key === "target_neuron") {
    if (window._tLayer !== undefined && window._tNeuron !== undefined) {
      renderTargetLocked(window._tLayer, window._tNeuron);
      // Highlight in 3D
      if (_lastGeomData && _lastGeomData.layer_map) {
        highlightTargetNeuron(window._tLayer, _lastGeomData.layer_map, _lastGeomData.points);
      }
    }
  }
  if (key === "rome_ready" && value) {
    romeReady = true;
    btnRomeEdit.disabled = false;
    btnRomeEdit.textContent = "🧬 APPLY NEURAL EDIT";
    showToast("Target locked. You can now apply the neural edit!", "");
  }
  if (key === "edit_result") {
    const { success, method, weights } = value;
    showToast(
      success
        ? `Edit applied — ${method} (${weights} weights) ✅`
        : "Edit failed. Check terminal for details. ❌",
      success ? "success" : "error"
    );
    btnRomeEdit.textContent = success ? "[OK] EDIT APPLIED" : "🧬 APPLY NEURAL EDIT";
    if (!success) btnRomeEdit.disabled = false;
    window._tLayer = window._tNeuron = undefined;
    if (!success) renderTargetWaiting();
  }
}

function updateTaskBar(current, total) {
  if (total <= 0) {
    taskProgressLabel.textContent = "No active tasks";
    taskProgressWrap.style.display = "none";
    return;
  }
  taskProgressLabel.textContent = `TASK  ${current} / ${total}`;
  taskProgressWrap.style.display = "block";
  const pct = total > 0 ? Math.round(((current - 1) / total) * 100) : 0;
  taskProgressBar.style.width = `${Math.max(0, pct)}%`;
}

function renderTargetLocked(layer, neuron) {
  targetDisplay.innerHTML =
    `<div class="target-locked">🎯 LAYER&nbsp; ${layer}<br>NEURON #${neuron}</div>`;
}

function renderTargetWaiting() {
  targetDisplay.innerHTML = `<span class="target-waiting">// AWAITING SCAN…</span>`;
  btnRomeEdit.disabled = true;
  btnRomeEdit.textContent = "🧬 APPLY NEURAL EDIT";
}

// ── Open SSE connection ───────────────────────────────────────────────────
function openSSE() {
  const es = new EventSource("/stream/pipeline");
  es.onmessage = (e) => {
    try { handleEvent(JSON.parse(e.data)); }
    catch (_) { /* heartbeat or malformed */ }
  };
  es.onerror = () => {
    appendLog("\n[SSE] Connection lost — reconnecting…\n", "#ff6644");
    setTimeout(openSSE, 3000);
    es.close();
  };
}

// ── AI Connection check ───────────────────────────────────────────────────
async function checkAI() {
  setConnStatus("checking", "CHECKING…");
  document.getElementById("btnConnCheck").disabled = true;
  document.getElementById("btnCheckConn").disabled = true;
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();
    const state = data.online ? "online" : "offline";
    setConnStatus(state, data.message);
    showToast(data.message, data.online ? "success" : "error");
  } catch (e) {
    setConnStatus("offline", "ERROR: " + e.message);
  } finally {
    document.getElementById("btnConnCheck").disabled = false;
    document.getElementById("btnCheckConn").disabled = false;
  }
}

// ── Progress bar animation helper ─────────────────────────────────────────
function startProgressAnim() {
  let val = 0;
  progressWrap.style.display = "block";
  progressBar.style.width = "0%";
  progressAnim = setInterval(() => {
    val = Math.min(val + 2, 90);
    progressBar.style.width = `${val}%`;
  }, 60);
}

function stopProgressAnim() {
  clearInterval(progressAnim);
  progressBar.style.width = "100%";
  setTimeout(() => {
    progressWrap.style.display = "none";
    progressBar.style.width = "0%";
  }, 600);
}

// ── Load model ────────────────────────────────────────────────────────────
document.getElementById("btnLoadLocal").addEventListener("click", async () => {
  const path = localPathInput.value.trim();
  if (!path) { showToast("Enter a local model directory path.", "error"); return; }
  startProgressAnim();
  showToast("Loading local model…", "");
  appendLog(`\n> Loading local model: ${path}\n`, "#00f3ff");
  await fetch("/api/load_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "local", path }),
  });
});

document.getElementById("btnLoadHF").addEventListener("click", async () => {
  const modelId = document.getElementById("hfInput").value.trim();
  if (!modelId) { showToast("Enter a HuggingFace model ID.", "error"); return; }
  startProgressAnim();
  showToast(`Downloading ${modelId}…`, "");
  appendLog(`\n> Connecting to HuggingFace: ${modelId}\n`, "#00f3ff");
  await fetch("/api/load_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "hf", path: modelId }),
  });
});

// ── RENDER 3D button ──────────────────────────────────────────────────────
btnLoad3D.addEventListener("click", fetchAndRender3D);

// ── Start Word ────────────────────────────────────────────────────────────
btnStartWord.addEventListener("click", async () => {
  const order = orderInput.value.trim();
  if (!order) { showToast("Describe the hallucination(s) first.", "error"); return; }

  btnStartWord.disabled = true;
  btnRomeEdit.disabled  = true;
  romeReady = false;
  taskTotal = taskCurrent = 0;
  window._tLayer = window._tNeuron = undefined;
  renderTargetWaiting();
  taskProgressLabel.textContent = "No active tasks";
  taskProgressWrap.style.display = "none";
  taskList.textContent = "";
  showToast("Pipeline started…", "");

  try {
    const res = await fetch("/api/start_word", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ order }),
    });
    if (!res.ok) throw new Error(await res.text());
  } catch (e) {
    showToast("Failed to start pipeline: " + e.message, "error");
    btnStartWord.disabled = false;
  }
});

// ── Apply ROME Edit ───────────────────────────────────────────────────────
btnRomeEdit.addEventListener("click", async () => {
  if (!romeReady) return;
  btnRomeEdit.disabled = true;
  btnRomeEdit.textContent = "⚙️ EDITING…";
  showToast("Applying neural edit…", "");
  await fetch("/api/apply_rome", { method: "POST" });
});

// ── Clear Terminal ────────────────────────────────────────────────────────
document.getElementById("btnClearLog").addEventListener("click", () => {
  termLog.innerHTML = `<span class="log-initline">> Terminal cleared.\n</span>`;
});

// ── Topbar CHECK AI ───────────────────────────────────────────────────────
document.getElementById("btnCheckConn").addEventListener("click", checkAI);
document.getElementById("btnConnCheck").addEventListener("click", checkAI);

// ── Init ──────────────────────────────────────────────────────────────────
openSSE();
// Initialize Three.js immediately so canvas is ready
setTimeout(() => { initThree(); }, 300);
// Auto-check AI status
setTimeout(checkAI, 800);
