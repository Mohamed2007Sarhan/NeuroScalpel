"""
point_and_layer_detect.py
=========================
NeuroScalpel — Precision Neural Targeting Engine

Phase 2 of the pipeline. Given a *trick prompt*, this module:

  1. Loads the target model with gradient tracking enabled.
  2. Attaches forward hooks to every FFN layer (MLP block) to capture:
        • The hidden state ENTERING the MLP  →  k* (the "key" vector in ROME notation)
        • The hidden state EXITING  the MLP  →  ffn output
  3. Runs a forward pass and computes layer-level deviation scores
     (cosine distance between consecutive hidden states at the last token).
  4. Identifies the CRITICAL LAYER — the one with the highest angular deviation.
  5. Within the critical layer, identifies the CRITICAL NEURON using THREE
     independent methods and combines them for a consensus score:
        a) MAX ACTIVATION — neuron with the largest absolute output magnitude.
        b) GRADIENT SENSITIVITY — torch.autograd.grad w.r.t. loss, isolates
           neurons whose activation shifts most severely impact the logit of
           the *wrong* prediction.
        c) K* PROJECTION — dot-product of k* (key vector) with each column of
           W_in (the MLP up-projection weight): directly mirrors ROME's formula
           k* = hidden_state @ W_in  and picks the neuron with peak projection.
  6. Returns a rich dict:
        {
          "critical_layer"     : "layer.N" (str),
          "critical_layer_idx" : N          (int),
          "critical_neuron"    : M          (int),   ← THE KEY NEW FIELD
          "k_star"             : [...],              ← k* vector (list[float])
          "neuron_scores"      : {...},              ← per-method scores
          "max_magnitude"      : float,
          "raw_report"         : str,
        }

The (layer_idx, neuron_idx) pair is passed downstream to:
    Phase 3 — DeepSeek confirmation agent (TARGET LOCKED: Layer X, Vector Point Y)
    Phase 5 — Neural edit engine (uses layer_hint + neuron_hint for ROME targeting)

Architecture support
--------------------
• GPT-2 family:  model.transformer.h[i].mlp
• LLaMA / Mistral / Gemma / Qwen:  model.model.layers[i].mlp
• Phi / OPT / Bloom:  model.model.decoder.layers[i].fc1/fc2 (via generic fallback)
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("NeuroScalpel.Detector")

# ──────────────────────────────────────────────────────────────────────────────
# Safe imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    PYTORCH_AVAILABLE = True
except Exception as _torch_err:
    PYTORCH_AVAILABLE = False
    logger.warning(f"PyTorch offline: {_torch_err}")


# ──────────────────────────────────────────────────────────────────────────────
# Hook storage structure
# ──────────────────────────────────────────────────────────────────────────────
class _LayerCapture:
    """Stores tensors captured by forward hooks for one layer."""
    __slots__ = ("layer_idx", "k_star", "ffn_output", "attn_output")

    def __init__(self, layer_idx: int):
        self.layer_idx   = layer_idx
        self.k_star      = None   # hidden state ENTERING the MLP (last token)
        self.ffn_output  = None   # hidden state EXITING  the MLP (last token)
        self.attn_output = None   # attention output (for anomaly scoring)


def _make_ffn_hook(capture: _LayerCapture):
    """Returns a forward hook that saves input/output of the FFN block."""
    def hook(module, inp, out):
        try:
            # inp is a tuple; inp[0] is the hidden state [batch, seq, hidden]
            x_in  = inp[0].detach()          # (1, seq, hidden)
            x_out = out if not isinstance(out, tuple) else out[0]
            x_out = x_out.detach()            # (1, seq, hidden)

            # Take the LAST TOKEN position — this is k* in ROME notation
            capture.k_star     = x_in [:, -1, :].float()   # (1, hidden)
            capture.ffn_output = x_out[:, -1, :].float()   # (1, hidden)
        except Exception:
            pass
    return hook


def _make_attn_hook(capture: _LayerCapture):
    """Forward hook for the attention block output."""
    def hook(module, inp, out):
        try:
            x_out = out[0] if isinstance(out, tuple) else out
            capture.attn_output = x_out.detach()[:, -1, :].float()
        except Exception:
            pass
    return hook


# ──────────────────────────────────────────────────────────────────────────────
# Architecture resolvers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_layers(model):
    """
    Returns a list of (attention_module, ffn_module) for each transformer block.
    Supports GPT-2, LLaMA, Mistral, Gemma, Phi, Qwen, OPT, Bloom.
    """
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [(model.transformer.h[i].attn, model.transformer.h[i].mlp)
                for i in range(len(model.transformer.h))]

    # LLaMA / Mistral / Gemma / Qwen style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        def _attn(l): return getattr(l, "self_attn", getattr(l, "attn", None))
        def _mlp(l):  return getattr(l, "mlp", None)
        return [(_attn(l), _mlp(l)) for l in layers]

    # OPT style
    if hasattr(model, "model") and hasattr(model.model, "decoder") \
            and hasattr(model.model.decoder, "layers"):
        layers = model.model.decoder.layers
        def _attn(l): return l.self_attn
        def _mlp(l):  return l   # OPT has no separate mlp module; use the block
        return [(_attn(l), _mlp(l)) for l in layers]

    return []   # unsupported architecture


def _get_win_weight(ffn_module, model_name: str) -> Optional["torch.Tensor"]:
    """
    Extracts the W_in weight matrix (up-projection of the FFN).
    This is needed for computing the k* projection scores per neuron.

    Returns shape (hidden_dim, ffn_intermediate_dim) or None.
    """
    # GPT-2: transformer.h[i].mlp  →  c_fc (W_in)
    if hasattr(ffn_module, "c_fc"):
        return ffn_module.c_fc.weight.detach().float()   # (ffn, hidden) in Conv1D

    # LLaMA / Mistral: mlp.gate_proj or mlp.up_proj
    if hasattr(ffn_module, "gate_proj"):
        return ffn_module.gate_proj.weight.detach().float()  # (ffn, hidden)

    if hasattr(ffn_module, "up_proj"):
        return ffn_module.up_proj.weight.detach().float()    # (ffn, hidden)

    # OPT: fc1
    if hasattr(ffn_module, "fc1"):
        return ffn_module.fc1.weight.detach().float()        # (ffn, hidden)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────
class CoreAnomalyDetector:
    """
    Precision neural targeting engine for NeuroScalpel.
    Identifies the exact (layer, neuron) responsible for a hallucination.
    """

    def __init__(self, model_name: str):
        if not model_name or not model_name.strip():
            raise ValueError("CoreAnomalyDetector requires a non-empty model_name.")
        self.model_name  = model_name
        self.model       = None
        self.tokenizer   = None
        self._hooks      = []
        self._captures: list[_LayerCapture] = []
        self.device      = torch.device("cuda" if PYTORCH_AVAILABLE and
                                         torch.cuda.is_available() else "cpu") \
                           if PYTORCH_AVAILABLE else "cpu"

    def adopt_loaded_model(self, model, tokenizer, model_name: Optional[str] = None):
        """
        Reuse an already-loaded model/tokenizer from ModelManager.
        This avoids re-downloading or re-loading from HuggingFace in Phase 2.
        """
        self.model = model
        self.tokenizer = tokenizer
        if model_name:
            self.model_name = model_name

    # ── Load ────────────────────────────────────────────────────────────────

    def load_model(self, log_callback: Optional[Callable] = None) -> bool:
        _log = log_callback or (lambda t, c: None)
        _log(f"[SYS] Initialising — device: {self.device}\n", "#00f3ff")
        _log(f"[SYS] Architecture target: {self.model_name}\n", "#00f3ff")

        if not PYTORCH_AVAILABLE:
            _log("[ERR] PyTorch unavailable — cannot load real model.\n", "#ff003c")
            return False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                output_hidden_states=True,
                output_attentions=True,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
            # We will need gradients for the gradient sensitivity pass
            # but keep eval mode for all other parameters
            self.model.eval()
            _log("[SYS] Model load complete. Ready for precision probe.\n", "#00ff00")
            return True
        except Exception as e:
            _log(f"[ERR] Model load failure: {e}\n{traceback.format_exc()}\n", "#ff003c")
            return False

    # ── Hooks ───────────────────────────────────────────────────────────────

    def attach_hooks(self, log_callback: Optional[Callable] = None) -> bool:
        _log = log_callback or (lambda t, c: None)
        if not PYTORCH_AVAILABLE or self.model is None:
            _log("[ERR] No model — cannot attach hooks.\n", "#ff003c")
            return False

        self._remove_hooks()
        self._captures.clear()

        layer_pairs = _resolve_layers(self.model)
        if not layer_pairs:
            _log("[WARN] Unsupported architecture — could not resolve layers.\n", "#ffaa00")
            return False

        _log(f"[HOOK] Attaching precision hooks across {len(layer_pairs)} transformer blocks...\n",
             "#00f3ff")

        for i, (attn_mod, ffn_mod) in enumerate(layer_pairs):
            cap = _LayerCapture(i)
            self._captures.append(cap)

            if ffn_mod is not None:
                h = ffn_mod.register_forward_hook(_make_ffn_hook(cap))
                self._hooks.append(h)

            if attn_mod is not None:
                h = attn_mod.register_forward_hook(_make_attn_hook(cap))
                self._hooks.append(h)

        _log(f"[HOOK] {len(self._hooks)} hooks active across {len(self._captures)} layers.\n",
             "#00ff00")
        return True

    # ── Probe ───────────────────────────────────────────────────────────────

    def probe_and_analyze(
        self,
        prompt: str,
        forced_layer_idx: Optional[int] = None,
        log_callback: Optional[Callable] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Full forward pass + neuron-level analysis.
        Returns the targeting dict including (critical_layer_idx, critical_neuron).
        """
        _log = log_callback or (lambda t, c: None)

        if not PYTORCH_AVAILABLE or self.model is None or self.tokenizer is None:
            _log("[ERR] Model not loaded.\n", "#ff003c")
            return None

        _log(f"\n[PROBE] Prompt: '{prompt}'\n", "#bc13fe")

        # ── Tokenise ────────────────────────────────────────────────────────
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        _log(f"[PROBE] Tokens ({len(tokens)}): {tokens}\n", "#0088aa")

        # ── Forward pass (no_grad for speed; grad only on targeted layer later) ──
        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states   # tuple: (n_layers+1) × (1, seq, hidden)
        logits        = outputs.logits          # (1, seq, vocab)

        top_ids = torch.topk(logits[0, -1, :], k=10).indices.tolist()
        predicted_id = top_ids[0]
        predicted_tok = self.tokenizer.decode([predicted_id])
        # Skip whitespace-only top tokens to get a semantically useful probe target.
        for cand_id in top_ids:
            cand_tok = self.tokenizer.decode([cand_id])
            if cand_tok.strip():
                predicted_id = cand_id
                predicted_tok = cand_tok
                break
        _log(f"[OUT] Predicted next token: '{predicted_tok}' (id={predicted_id})\n", "#bc13fe")

        # ── Layer-level deviation scan ───────────────────────────────────────
        _log("\n[SCAN] --- LAYER-BY-LAYER DEVIATION ANALYSIS ---\n", "#bc13fe")

        n_layers          = len(hidden_states) - 1
        max_dev           = -1.0
        critical_layer_i  = 0
        prev_last         = hidden_states[0][:, -1, :].float()
        report_lines      = []
        layer_scores      = []   # (layer_idx, dev_score, l2_norm)
        layer_io_metrics  = []   # dict per layer from real FFN input/output hooks

        for i in range(n_layers):
            cur_last = hidden_states[i + 1][:, -1, :].float()
            l2_norm  = torch.linalg.norm(cur_last, dim=-1).mean().item()

            cos_sim   = F.cosine_similarity(cur_last, prev_last, dim=-1).mean().item()
            dev_score = 1.0 - cos_sim

            layer_label = f"layer.{i}"
            state = "Measured"
            color = "#00f3ff"

            line = (f"  {layer_label:^22} | last_tok | "
                    f"L2: {l2_norm:8.3f} | Dev: {dev_score:6.4f} | {state}\n")
            _log(line, color)
            report_lines.append(line)
            layer_scores.append((i, dev_score, l2_norm))

            cap_i = self._captures[i] if i < len(self._captures) else None
            io_l2_delta = 0.0
            io_cos_dev = 0.0
            if cap_i is not None and cap_i.k_star is not None and cap_i.ffn_output is not None:
                k_in = cap_i.k_star.squeeze(0).float()
                f_out = cap_i.ffn_output.squeeze(0).float()
                io_l2_delta = torch.linalg.norm(f_out - k_in).item()
                io_cos_dev = 1.0 - F.cosine_similarity(
                    f_out.unsqueeze(0), k_in.unsqueeze(0), dim=-1
                ).item()
            layer_io_metrics.append({
                "layer": i,
                "io_l2_delta": round(io_l2_delta, 6),
                "io_cos_dev": round(io_cos_dev, 6),
            })
            prev_last = cur_last

        # State-level report includes all hidden states: layer.0 ... layer.N
        # For GPT-2 small, this naturally shows layer.12 in addition to layer.0.
        state_report_lines = []
        for s_idx in range(len(hidden_states)):
            s_last = hidden_states[s_idx][:, -1, :].float()
            s_l2 = torch.linalg.norm(s_last, dim=-1).mean().item()
            if s_idx == 0:
                trans_dev = 0.0
            else:
                p_last = hidden_states[s_idx - 1][:, -1, :].float()
                trans_dev = 1.0 - F.cosine_similarity(s_last, p_last, dim=-1).mean().item()
            state_report_lines.append(
                f"  state.layer.{s_idx:<2d} | L2={s_l2:8.3f} | Dev(prev->cur)={trans_dev:7.5f}\n"
            )

        if not layer_scores or not layer_io_metrics:
            _log("[ERR] Missing per-layer metrics. Cannot compute critical layer.\n", "#ff003c")
            return None
        # Strict data-driven layer selection:
        # combine normalized hidden-state deviation and normalized FFN IO delta.
        dev_vals = torch.tensor([x[1] for x in layer_scores], dtype=torch.float32)
        io_vals = torch.tensor([x["io_l2_delta"] for x in layer_io_metrics], dtype=torch.float32)
        dev_norm = dev_vals / (dev_vals.max() + 1e-12)
        io_norm = io_vals / (io_vals.max() + 1e-12)
        joint = (dev_norm + io_norm) / 2.0
        critical_layer_i = int(torch.argmax(joint).item())
        max_dev = float(layer_scores[critical_layer_i][1])
        io_critical_layer_i = int(torch.argmax(io_norm).item())
        if forced_layer_idx is not None:
            if forced_layer_idx < 0 or forced_layer_idx >= n_layers:
                _log(f"[ERR] forced_layer_idx={forced_layer_idx} is out of range.\n", "#ff003c")
                return None
            forced_row = next((r for r in layer_scores if r[0] == forced_layer_idx), None)
            if forced_row is None:
                _log(f"[ERR] Missing metrics for forced layer {forced_layer_idx}.\n", "#ff003c")
                return None
            critical_layer_i, max_dev, _ = forced_row
            _log(
                f"[VERIFY] Forced layer mode active: using layer.{forced_layer_idx} for neuron targeting.\n",
                "#ffaa00",
            )

        critical_layer_str = f"layer.{critical_layer_i}"
        _log(f"\n[RESULT] Critical layer: {critical_layer_str}"
             f"  (deviation={max_dev:.4f})\n", "#bc13fe")

        # ── Retrieve k* for the critical layer ──────────────────────────────
        cap = self._captures[critical_layer_i] if critical_layer_i < len(self._captures) else None
        k_star = cap.k_star if (cap is not None and cap.k_star is not None) else None

        if k_star is None:
            _log("[ERR] k* not captured for critical layer. Strict mode blocks fallback.\n", "#ff003c")
            return None

        # ── Resolve FFN module for the critical layer ────────────────────────
        layer_pairs = _resolve_layers(self.model)
        _, critical_ffn = (layer_pairs[critical_layer_i]
                          if critical_layer_i < len(layer_pairs) else (None, None))

        # ── Neuron identification — three independent methods ────────────────
        _log(f"\n[NEURON] Targeting neurons within {critical_layer_str}...\n", "#00f3ff")

        neuron_scores = {}
        critical_neuron = None

        # ── Method A: Max FFN output activation ─────────────────────────────
        ffn_out_vec = cap.ffn_output if (cap is not None and cap.ffn_output is not None) else None
        score_a = None
        if ffn_out_vec is not None:
            abs_vals = ffn_out_vec.squeeze(0).abs()   # (hidden,)
            neuron_a = abs_vals.argmax().item()
            score_a  = abs_vals[neuron_a].item()
            neuron_scores["max_activation"] = {
                "neuron": neuron_a, "score": round(score_a, 6)
            }
            _log(f"  [A] Max-Activation  → neuron {neuron_a:5d}  (|act|={score_a:.4f})\n",
                 "#00f3ff")

        # ── Method B: k* projection onto W_in ───────────────────────────────
        #   ROME formula: project k* through the up-projection weight matrix
        #   dot(k*, W_in[neuron]) gives the pre-activation magnitude per neuron
        score_b = None
        neuron_b = None
        if critical_ffn is not None:
            W_in = _get_win_weight(critical_ffn, self.model_name)
            if W_in is not None:
                # W_in shape: (ffn_dim, hidden) or (hidden, ffn_dim) depending on arch
                # GPT-2 Conv1D transposes: shape is (hidden, ffn_dim)
                kstar_flat = k_star.squeeze(0).to(W_in.device)  # (hidden,)

                if W_in.shape[1] == kstar_flat.shape[0]:   # (ffn_dim, hidden)
                    proj = W_in @ kstar_flat                # (ffn_dim,)
                elif W_in.shape[0] == kstar_flat.shape[0]: # (hidden, ffn_dim) — GPT2 Conv1D
                    proj = W_in.T @ kstar_flat              # (ffn_dim,)
                else:
                    proj = None

                if proj is not None:
                    neuron_b = proj.abs().argmax().item()
                    score_b  = proj.abs()[neuron_b].item()
                    neuron_scores["k_star_projection"] = {
                        "neuron": neuron_b, "score": round(score_b, 6)
                    }
                    _log(f"  [B] k* Projection   → neuron {neuron_b:5d}  "
                         f"(|k*·w_in|={score_b:.4f})\n", "#00f3ff")

        # ── Method C: Gradient sensitivity (∂loss/∂ffn_output) ─────────────
        #   Measures how much each neuron's activation affects the logit
        #   of the predicted (wrong) token — the one we want to suppress.
        score_c = None
        neuron_c = None
        if critical_ffn is not None and cap is not None and cap.ffn_output is not None:
            try:
                # Re-run with retain_graph to get gradients
                # Enable grads only for the FFN output of the critical layer
                cap_rerun = _LayerCapture(critical_layer_i)
                h_tmp = critical_ffn.register_forward_hook(_make_ffn_hook(cap_rerun))
                try:
                    with torch.enable_grad():
                        out2 = self.model(**inputs)
                    h_tmp.remove()

                    # Get the FFN output tensor that was captured
                    ffn_out_grad = cap_rerun.ffn_output  # (1, hidden) — detached

                    # Recompute through a differentiable path:
                    # We need a tensor that has grad. Use the hidden state directly.
                    hs_target = hidden_states[critical_layer_i + 1]  # right after the layer
                    hs_target = hs_target[:, -1, :].float().requires_grad_(True)

                    # Project to vocab to get logit of the wrong prediction
                    # (use lm_head or language_model_head)
                    lm_head = (getattr(self.model, "lm_head", None) or
                               getattr(self.model, "embed_out", None))
                    if lm_head is not None:
                        logit_wrong = lm_head(hs_target)[0, predicted_id]
                        grads = torch.autograd.grad(logit_wrong, hs_target)[0]
                        grad_abs = grads.squeeze(0).abs()  # (hidden,)
                        neuron_c = grad_abs.argmax().item()
                        score_c  = grad_abs[neuron_c].item()
                        neuron_scores["gradient_sensitivity"] = {
                            "neuron": neuron_c, "score": round(score_c, 6)
                        }
                        _log(f"  [C] Gradient Sens.  → neuron {neuron_c:5d}  "
                             f"(|∂loss/∂hidden|={score_c:.4f})\n", "#00f3ff")
                except Exception as eg:
                    h_tmp.remove()
                    _log(f"  [C] Gradient pass skipped: {eg}\n", "#ffaa00")
            except Exception as eg2:
                _log(f"  [C] Gradient setup failed: {eg2}\n", "#ffaa00")

        # ── Consensus neuron ─────────────────────────────────────────────────
        # Strict data-driven choice by normalized score only (no fixed priority).
        method_candidates = []
        if score_a is not None:
            method_candidates.append(("max_activation", neuron_scores["max_activation"]["neuron"], float(score_a)))
        if score_b is not None and neuron_b is not None:
            method_candidates.append(("k_star_projection", neuron_b, float(score_b)))
        if score_c is not None and neuron_c is not None:
            method_candidates.append(("gradient_sensitivity", neuron_c, float(score_c)))

        if not method_candidates:
            _log("[ERR] No neuron method produced a valid target. Strict mode blocks fallback.\n", "#ff003c")
            return None
        score_tensor = torch.tensor([m[2] for m in method_candidates], dtype=torch.float32)
        norm_tensor = score_tensor / (score_tensor.max() + 1e-12)
        best_i = int(torch.argmax(norm_tensor).item())
        method_used, critical_neuron, _ = method_candidates[best_i]

        # Agreement check across methods
        votes = [n for n in [score_a and neuron_scores["max_activation"]["neuron"],
                              neuron_b, neuron_c] if n is not None]
        agreement_pct = (votes.count(critical_neuron) / len(votes) * 100) if votes else 0.0

        _log(f"\n[NEURON] CONSENSUS: neuron {critical_neuron}"
             f"  (method={method_used}, agreement={agreement_pct:.0f}%)\n", "#ff2244")
        _log(f"[TARGET] Layer {critical_layer_i} | Neuron {critical_neuron} 🎯\n", "#ff2244")

        # ── Save JSON log ────────────────────────────────────────────────────
        log_path = Path("logs") / "pipeline" / "neuron_targeting.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        summary_doc = {
            "model": self.model_name,
            "prompt": prompt,
            "tokens": tokens,
            "predicted_next_token": predicted_tok,
            "critical_layer_idx": critical_layer_i,
            "critical_layer_str": critical_layer_str,
            "critical_neuron": critical_neuron,
            "consensus_method": method_used,
            "agreement_pct": round(agreement_pct, 1),
            "k_star_shape": list(k_star.shape) if k_star is not None else [],
            "neuron_scores": neuron_scores,
            "layer_deviations": [
                {"layer": i, "dev": round(d, 6), "l2": round(l, 4)}
                for i, d, l in layer_scores
            ],
        }
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(summary_doc, f, indent=2)
            _log(f"[LOG] Targeting report saved → {log_path}\n", "#00f3ff")
        except Exception as e:
            _log(f"[WARN] Could not save JSON: {e}\n", "#ffaa00")

        # ── Build raw_report for Phase 3 ─────────────────────────────────────
        raw_report = (
            "=== NEURON TARGETING REPORT ===\n"
            + f"MODEL LAYER RANGE (blocks): 0..{n_layers - 1}\n"
            + f"MODEL STATE RANGE  (hidden): 0..{n_layers}\n"
            + "".join(report_lines)
            + "\n=== FULL HIDDEN-STATE SCAN (includes layer.0 .. layer.N) ===\n"
            + "".join(state_report_lines)
            + f"\nCRITICAL LAYER : {critical_layer_str} (idx={critical_layer_i})"
            f"  |  deviation={max_dev:.4f}\n"
            + f"IO REF LAYER   : layer.{io_critical_layer_i} (max io_l2_delta)\n"
            + f"CRITICAL NEURON: {critical_neuron}  (method={method_used},"
            f" agreement={agreement_pct:.0f}%)\n"
            + "\nEQUATIONS:\n"
            + "  Dev(layer_i) = 1 - cos(h_i_last, h_(i-1)_last)\n"
            + "  IO_L2(layer_i) = ||ffn_out_i_last - ffn_in_i_last||_2\n"
            + "  k* projection score(j) = |<k*_i, W_in_i[:, j]>|\n"
            + "  gradient score(j) = |d logit(pred_token) / d hidden_i[j]|\n"
            + "  joint layer score(i) = 0.5 * norm(Dev_i) + 0.5 * norm(IO_L2_i)\n"
            + "\nNEURON SCORES:\n"
            + json.dumps(neuron_scores, indent=2)
            + "\n\nLAYER IO METRICS:\n"
            + json.dumps(layer_io_metrics, indent=2)
            + f"\n\nPREDICTED WRONG TOKEN: '{predicted_tok}' (id={predicted_id})\n"
            + f"MODEL: {self.model_name}\n"
            + f"TOKENS: {tokens}\n"
        )

        return {
            "critical_layer":     critical_layer_str,
            "critical_layer_idx": critical_layer_i,
            "critical_neuron":    critical_neuron,
            "target_lock": {
                "layer_idx": critical_layer_i,
                "neuron_idx": critical_neuron,
                "canonical": f"{critical_layer_i}:{critical_neuron}",
            },
            "k_star":             k_star.squeeze(0).tolist() if k_star is not None else [],
            "neuron_scores":      neuron_scores,
            "consensus_method":   method_used,
            "agreement_pct":      round(agreement_pct, 1),
            "io_critical_layer_idx": io_critical_layer_i,
            "layer_io_metrics":   layer_io_metrics,
            "max_magnitude":      max_dev,
            "raw_report":         raw_report,
        }

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def cleanup(self):
        """Remove all hooks and release model memory."""
        self._remove_hooks()
        if PYTORCH_AVAILABLE and self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
