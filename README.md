<!--
  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗
  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗
  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║
  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║
  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝
  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
   ███████╗ ██████╗ █████╗ ██╗     ██████╗ ███████╗██╗
   ██╔════╝██╔════╝██╔══██╗██║     ██╔══██╗██╔════╝██║
   ███████╗██║     ███████║██║     ██████╔╝█████╗  ██║
   ╚════██║██║     ██╔══██║██║     ██╔═══╝ ██╔══╝  ██║
   ███████║╚██████╗██║  ██║███████╗██║     ███████╗███████╗
   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚══════╝
-->

<div align="center">

# ⚡ NeuroScalpel

### *Surgical Precision for the Mind of an LLM*

[![Python](https://img.shields.io/badge/Python-3.10%2B-00f3ff?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-hooks--powered-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-bc13fe?style=for-the-badge&logo=qt&logoColor=white)](https://pypi.org/project/PyQt6/)
[![ROME](https://img.shields.io/badge/ROME-rank--1%20editing-00ff88?style=for-the-badge)](https://rome.baulab.info/)

> **Cut. Target. Heal.**  
> NeuroScalpel is a real-time neural surgery toolkit that hunts hallucinations inside large language models, pinpoints the exact layer and neuron responsible, and rewrites the offending weight — without fine-tuning, without guesswork.

---

</div>

## 🧬 What Is This?

Modern LLMs hallucinate. They confidently state wrong facts baked deep into their transformer weights.  
Traditional fixes? Retrain millions of parameters. Hope for the best.

**NeuroScalpel does something different.**

It treats the model like a patient — inserts live **PyTorch tensor hooks** into every FFN layer, traces the moment a wrong belief surfaces mathematically, locks onto the precise neuron coordinate with an autonomous AI agent, and applies a targeted **ROME rank-1 weight edit** to correct it — all while the model stays loaded in memory.

No retraining. No guesswork. Pure surgical precision.

---

## 🌌 The Pipeline — 5 Phases of Neural Surgery

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SURGEON MIND ONLINE                          │
│                                                                     │
│  [USER ORDER] ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4     │
│   "Fix wrong            DeepSeek    PyTorch     Target       3D    │
│    capital"             Diagnosis   Neural      Lock        Viz    │
│                         (JSON)      Scan        (AI)               │
│                                       │              │             │
│                                       ▼              ▼             │
│                              Tensor telemetry   Neuron ID          │
│                              layer deviations   highlighted        │
│                                                                     │
│  ──────────────► Phase 5 ◄─────── (User confirms)                  │
│                   ROME Edit                                         │
│                   LyapLock stabilise                                │
│                   Session saved ✓                                   │
└─────────────────────────────────────────────────────────────────────┘
```

| Phase | Name | What Happens |
|-------|------|-------------|
| **1** | 🧠 Surgeon Mind | DeepSeek AI reads your order, identifies every distinct wrong belief, generates a targeted *trick prompt* designed to reproduce the hallucination |
| **2** | 🔬 Neural Scan | PyTorch hooks attach to every FFN layer — a real forward pass runs, capturing L2 norms, cosine deviations, and hidden states token-by-token |
| **3** | 🎯 Target Lock | DeepSeek ingests the raw tensor telemetry and pinpoints `Layer [X], Vector Point [Y]` — the exact birth-coordinate of the error |
| **4** | 🌐 3D Visualise | The offending neuron is highlighted in the live cyberpunk 3D point cloud with a pulsing red marker |
| **5** | ⚙️ ROME + LyapLock | A rank-1 weight update rewrites the hallucination in-place; LyapLock prevents catastrophic forgetting |

> Multi-task support: describe **multiple hallucinations in a single order** — NeuroScalpel queues them all and runs the full pipeline for each automatically.

---

## ✨ Feature Highlights

- **Live FFN Tensor Hooks** — track `mlp.c_fc` / `mlp.c_proj` activations inside every transformer block in real time
- **Streaming Surgeon Mind** — watch DeepSeek's reasoning tokens stream into the HUD *as it thinks*
- **Layered 3D Geometry** — real PCA-reduced model weights rendered as a neon glowing point cloud, sliced by layer
- **Multi-Task Queue** — pipeline loops automatically for complex orders with several errors to correct
- **Session SQLite DB** — every run is persisted: scan results, target coords, edits applied, all timestamped
- **Generated Log** — human-readable daily log of every pipeline event written to `logs/`
- **JSON Telemetry Dump** — full hidden states + FFN entries exported to `transformer_neuron_log.json`
- **Cyberpunk HUD** — three floating terminal windows stream live diagnostics (Agent Mind / Deep Core / Edit Engine)

---

## 🖥️ UI Overview

```
┌──────────────────┬───────────────────────────────┬──────────────────┐
│  Feature         │                               │   Dashboard      │
│  Extractor       │     3D Neural Visualizer      │   Panel          │
│                  │                               │                  │
│  ▸ Load Model    │   ·  ·    ·           ·  ·   │  Target Lock     │
│  ▸ HF Model ID   │      ·  ·   · ● ·  ·         │  Layer / Neuron  │
│  ▸ Console Log   │   ·     ·     ·   ·    ·  ·  │                  │
│  ▸ AI Status     │      · ·   ·    ·    ·        │  Task Queue      │
│                  │   ·    ·  ·  ●  ·  ·   ·     │                  │
│                  │      ·    ·   ·    ·    ·     │  ⚡ START WORD   │
│                  │                               │  🧬 ROME EDIT    │
└──────────────────┴───────────────────────────────┴──────────────────┘
   [Agent Mind Terminal]  [Deep Core Logs]  [Edit Engine]   ← HUDs
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers pyqt6 pyqtgraph scikit-learn openai datasets
```

> **Note:** NeuroScalpel uses native PyTorch DLL bindings. CPU fallback logic is in place — if CUDA architecture is unsupported, the UI will show a safe error instead of crashing.

### Launch

```bash
python main.py
```

### Workflow

1. **Load a Model**  
   Enter a HuggingFace model ID in the *Feature Extractor* panel (e.g. `openai-community/gpt2`) and hit **LOAD**.  
   The 3D visualizer will build the layered geometry from real model weights.

2. **Write an Order**  
   In the *Dashboard*, type a description of the hallucination(s) the model has:  
   ```
   The model claims the capital of France is Lyon.
   The model believes Einstein invented the telephone.
   ```

3. **Hit START WORD**  
   The 5-phase pipeline fires automatically. Watch the three HUD terminals stream live output.

4. **Confirm the Edit**  
   After Phase 3 locks the target neuron, click **🧬 APPLY ROME EDIT** to rewrite the weight in-place.

---

## 🗂️ Architecture

```
NeuroScalpel/
├── main.py                     ← Entry point, splash, stylesheet, app icon
├── ui/
│   ├── main_window.py          ← Pipeline orchestrator (Phases 1–5 threads)
│   ├── panels/
│   │   ├── feature_extractor.py ← Model loader, console readout, AI status
│   │   ├── dashboard.py         ← Order input, task queue, ROME button
│   │   └── order_terminal.py    ← Floating HUD terminal windows
│   └── visualizer/
│       └── point_cloud_3d.py    ← PyQtGraph 3D layer-slab visualizer
├── core/
│   ├── point_and_layer_detect.py ← CoreAnomalyDetector, FFN hooks, tensor analysis
│   ├── model_backend.py          ← ModelManager, PCA extraction, ROME edit dispatch
│   ├── session_manager.py        ← SQLite WAL session persistence
│   ├── task_queue.py             ← Multi-task queue + EditTask dataclass
│   └── generated_log.py          ← Human-readable pipeline event logging
├── LyapLock/                     ← Lyapunov stability post-edit stabiliser
├── img/
│   └── icon.ico                  ← Application icon
└── sessions/                     ← Per-run SQLite databases
```

---

## 🔬 Technical Deep Dive

### PyTorch Hook Strategy

NeuroScalpel attaches `register_forward_hook` to the `mlp` sub-module of each transformer block. During the trick-prompt forward pass it captures:

- **Pre/Post activations** — hidden state vectors entering and leaving each FFN
- **L2 Norm magnitude** — total energy of each layer's output
- **Cosine similarity drift** — angular deviation from baseline token representations
- **∆ deviation score** — cross-layer delta to identify anomalous spikes

### ROME Rank-1 Edit

The edit is a targeted outer-product weight update:

```
W_new = W_old + (v* - W_old · k*) ⊗ k* / (C · k*)
```

Where `k*` is the critical key-vector at the hallucinated layer and `v*` is the corrected value vector — computed via the model's context with the correct fact.

### LyapLock Stabilisation

Post-edit, LyapLock applies a Lyapunov stability constraint check to verify the edited weight matrix preserves the model's overall attractor dynamics, preventing catastrophic forgetting of unrelated knowledge.

---

## 📋 Requirements

| Package | Purpose |
|---------|---------|
| `torch` | Live tensor hooks, forward passes, weight editing |
| `transformers` | HuggingFace model loading & tokenization |
| `PyQt6` + `pyqtgraph` | Cyberpunk HUD interface & 3D visualizer |
| `scikit-learn` | PCA dimensionality reduction for 3D geometry |
| `openai` | DeepSeek / NVIDIA NIM API client (Phases 1 & 3) |
| `datasets` | HuggingFace datasets support for ROME context |

---

<div align="center">

---

*Built with obsessive attention to neural-level detail.*  
*No neurons were permanently harmed in the making of this tool.*

**⚡ NeuroScalpel — Operate with Precision.**

</div>
