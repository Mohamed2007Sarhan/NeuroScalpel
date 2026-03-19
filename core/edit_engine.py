"""
edit_engine.py
==============
Bridge between the NeuroScalpel UI and the real ROME / LyapLock algorithms
stored in the LyapLock/ subfolder.

High-level flow
---------------
1.  The model and tokenizer must already be loaded (we receive them).
2.  We build the ROME request dict from the EditTask (subject, prompt, target).
3.  We load the appropriate HyperParams for the model architecture.
4.  We run apply_rome_to_model()  →  the model is surgically edited.
5.  We then run apply_lyaplock_to_model() to stabilise / prevent forgetting.
6.  We return EditResult with success flag and any changed weight names.

CPU-only graceful fallback
--------------------------
If CUDA is unavailable or ROME fails (often needs cached statistics),
the engine logs the error and returns a failed EditResult so the UI
can report this cleanly without crashing.
"""

import sys
import os
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("NeuroScalpel.EditEngine")

# --------------------------------------------------------------------------
# Make sure the LyapLock package root is on sys.path so imports work
# --------------------------------------------------------------------------
_LYAPLOCK_ROOT = Path(__file__).resolve().parent.parent / "LyapLock"
if str(_LYAPLOCK_ROOT) not in sys.path:
    sys.path.insert(0, str(_LYAPLOCK_ROOT))


# --------------------------------------------------------------------------
# Optional imports (ROME / LyapLock need torch + transformers)
# --------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_OK = True
except Exception as _e:
    logger.warning(f"PyTorch unavailable in edit_engine: {_e}")
    TORCH_OK = False

ROME_OK = False
LYAPLOCK_OK = False

if TORCH_OK:
    try:
        from rome.rome_main import apply_rome_to_model
        from rome.rome_hparams import ROMEHyperParams
        ROME_OK = True
        logger.info("ROME algorithm loaded successfully.")
    except Exception as _e:
        logger.warning(f"Could not load ROME: {_e}")

    try:
        from lyaplock.lyaplock_main import apply_lyaplock_to_model
        from lyaplock.LyapLock_hparams import LyapLockHyperParams
        LYAPLOCK_OK = True
        logger.info("LyapLock algorithm loaded successfully.")
    except Exception as _e:
        logger.warning(f"Could not load LyapLock: {_e}")


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class EditRequest:
    """Everything the neural edit engine needs to perform a single fact correction."""
    subject: str              # e.g. "Egypt"
    prompt_template: str      # e.g. "The capital of {} is"  — use {} for subject
    target_new: str           # e.g. "Cairo"
    target_old: str = ""      # optional — the wrong value the model currently says
    layer_hint: Optional[int] = None   # layer index from Phase 2/3 targeting
    neuron_hint: Optional[int] = None  # specific neuron from Phase 2 — sharpens v* computation


@dataclass
class EditResult:
    """Result returned from the edit engine after running ROME + LyapLock."""
    success: bool
    method: str                     # "ROME+LyapLock" | "ROME_only" | "failed"
    weights_changed: list = field(default_factory=list)
    error_message: str = ""
    notes: str = ""


# --------------------------------------------------------------------------
# Default HyperParams builders
# --------------------------------------------------------------------------

def _build_rome_hparams(model_name: str, layer: Optional[int] = None) -> "ROMEHyperParams":
    """
    Returns sensible default ROME HyperParams.
    Tries to load from hparams/ YAML files first, falls back to hard-coded defaults.
    """
    hparams_dir = _LYAPLOCK_ROOT / "hparams" / "ROME"

    # Try to find a matching YAML
    candidate = None
    name_lower = model_name.lower()
    if hparams_dir.exists():
        for f in hparams_dir.glob("*.yaml"):
            if any(kw in name_lower for kw in [f.stem.lower(), "gpt2"]):
                candidate = f
                break
        if candidate is None and list(hparams_dir.glob("*.yaml")):
            candidate = sorted(hparams_dir.glob("*.yaml"))[0]

    if candidate:
        try:
            hp = ROMEHyperParams.from_hparams(candidate)
            if layer is not None:
                hp.layers = [layer]
            logger.info(f"ROME HyperParams loaded from {candidate.name}")
            return hp
        except Exception as e:
            logger.warning(f"Could not load HyperParams YAML: {e}")

    # Hard-coded GPT-2 defaults as fallback
    hp = ROMEHyperParams()
    hp.layers = [layer] if layer is not None else [17]
    hp.fact_token = "subject_last"
    hp.v_num_grad_steps = 20
    hp.v_lr = 5e-1
    hp.v_loss_layer = 23
    hp.v_weight_decay = 0.5
    hp.clamp_norm_factor = 4
    hp.kl_factor = 0.0625
    hp.mom2_adjustment = True
    hp.mom2_update_weight = 15000
    hp.rewrite_module_tmp = "transformer.h.{}.mlp.c_proj"
    hp.layer_module_tmp = "transformer.h.{}"
    hp.mlp_module_tmp = "transformer.h.{}.mlp"
    hp.attn_module_tmp = "transformer.h.{}.attn"
    hp.ln_f_module = "transformer.ln_f"
    hp.lm_head_module = "transformer.wte"
    hp.mom2_dataset = "wikipedia"
    hp.mom2_n_samples = 100000
    hp.mom2_dtype = "float32"
    hp.context_template_length_params = [[5, 10], [10, 10]]
    logger.warning("Using hard-coded GPT-2 ROME HyperParams as fallback.")
    return hp


def _build_lyaplock_defaults(num_layers: int) -> Dict[str, Any]:
    """Returns the kwargs dict required by execute_lyaplock on first run (cnt=0)."""
    return {
        "V": {i: 1.0 for i in range(num_layers)},
        "Z": {i: "D" for i in range(num_layers)},
        "alpha": {i: 1.0 for i in range(num_layers)},
        "a": {i: "1_D" for i in range(num_layers)},
        "b": {i: 0.0 for i in range(num_layers)},
        "zmax": {i: "D" for i in range(num_layers)},
        "D_base": {i: None for i in range(num_layers)},
        "cnt": 0,
    }


# --------------------------------------------------------------------------
# Main engine
# --------------------------------------------------------------------------

class ROMEEditEngine:
    """
    Performs surgical model weight editing using ROME, then stabilises
    the model with LyapLock to prevent catastrophic forgetting.

    This class is stateless — pass model + tokenizer each time.
    """

    @staticmethod
    def apply_edit(
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        request: EditRequest,
        log_callback=None
    ) -> EditResult:
        """
        Runs the full ROME → LyapLock pipeline on the given model in-place.

        Parameters
        ----------
        model       : loaded HuggingFace model (in eval or train mode)
        tokenizer   : matching tokenizer
        request     : EditRequest with subject / prompt / target
        log_callback: optional callable(str, hex_color) for streaming logs to UI

        Returns
        -------
        EditResult
        """

        def _log(msg: str, color: str = "#00f3ff"):
            logger.info(msg.strip())
            if log_callback:
                log_callback(msg, color)

        if not TORCH_OK:
            return EditResult(
                success=False,
                method="failed",
                error_message="PyTorch not available."
            )

        # ----------------------------------------------------------------
        # 1. Build ROME request dict
        # ----------------------------------------------------------------
        # ROME needs the prompt to contain "{}" as placeholder for subject
        prompt_template = request.prompt_template
        if "{}" not in prompt_template and request.subject in prompt_template:
            prompt_template = prompt_template.replace(request.subject, "{}")
        elif "{}" not in prompt_template:
            prompt_template = prompt_template.rstrip(" ") + " {}"

        rome_request = {
            "prompt": prompt_template,
            "subject": request.subject,
            "target_new": {"str": request.target_new},
            "target_true": {"str": request.target_old} if request.target_old else {"str": ""},
        }

        _log(f"\n[EDIT ENGINE] ⚡ Precision neural edit\n"
             f"  Subject      : {request.subject}\n"
             f"  Prompt       : {prompt_template.format(request.subject)}\n"
             f"  Target (new) : {request.target_new}\n"
             f"  Layer hint   : {request.layer_hint}\n"
             f"  Neuron hint  : {request.neuron_hint}\n", "#bc13fe")

        # ----------------------------------------------------------------
        # 2. ROME
        # ----------------------------------------------------------------
        if not ROME_OK:
            _log("[EDIT ENGINE] ⚠  ROME not available. Edit cannot proceed.\n", "#ffaa00")
            return EditResult(
                success=False,
                method="failed",
                error_message="ROME library not importable."
            )

        weights_changed = []
        try:
            model_name = getattr(model.config, "_name_or_path", "unknown")
            hparams = _build_rome_hparams(model_name, layer=request.layer_hint)

            _log(f"[ROME] Targeting layer(s): {hparams.layers}  |  neuron hint: {request.neuron_hint}\n",
                 "#00f3ff")
            _log("[ROME] Computing rank-one update vectors (k*, v*)...\n", "#00f3ff")

            model, weights_copy, _ = apply_rome_to_model(
                model=model,
                tok=tokenizer,
                request=[rome_request],
                hparams=hparams,
                copy=False,
                return_orig_weights=True,
            )

            weights_changed = list(weights_copy.keys())

            # ── Neuron-selective mask ────────────────────────────────────────
            # ROME applies a full rank-1 update: W += u⊗v
            # When we have a precise neuron_hint from Phase 2, we concentrate
            # the update onto that neuron's column: 80% on target, 20% spread.
            if request.neuron_hint is not None and request.neuron_hint >= 0:
                try:
                    import torch as _torch
                    neuron_idx = request.neuron_hint
                    _log(f"[ROME] Applying neuron-selective mask → neuron {neuron_idx}\n",
                         "#00f3ff")
                    for w_name in weights_changed:
                        from util import nethook as _nethook
                        w = _nethook.get_parameter(model, w_name)
                        orig = weights_copy[w_name]         # saved before edit
                        delta = w.detach() - orig           # full ROME delta

                        # Build concentration mask (same shape as delta)
                        mask = _torch.full_like(delta, 0.2)  # 20% everywhere
                        # Concentrate 80% on the neuron dimension
                        # Delta shape is (hidden, ffn) or (ffn, hidden) depending on arch
                        n_neurons = min(delta.shape)          # smaller dim = neuron count
                        n_idx     = max(delta.shape)          # larger dim  = hidden
                        if neuron_idx < delta.shape[0]:       # neuron is row
                            mask[neuron_idx, :] = 0.8
                        if neuron_idx < delta.shape[1]:       # neuron is col
                            mask[:, neuron_idx] = 0.8

                        with _torch.no_grad():
                            w[...] = orig + delta * mask
                    _log(f"[ROME] ✅ Neuron-selective update applied.\n", "#00ff00")
                except Exception as _em:
                    _log(f"[ROME] ⚠  Mask step skipped ({_em}). Full-matrix update kept.\n",
                         "#ffaa00")

            _log(f"[ROME] ✅ Weights updated: {weights_changed}\n", "#00ff00")

        except Exception as e:
            err = traceback.format_exc()
            _log(f"[ROME] ❌ Failed: {e}\n{err}\n", "#ff003c")
            return EditResult(
                success=False,
                method="failed",
                weights_changed=weights_changed,
                error_message=str(e)
            )

        # ----------------------------------------------------------------
        # 3. LyapLock stabilisation pass
        # ----------------------------------------------------------------
        if not LYAPLOCK_OK:
            _log("[LyapLock] ⚠  LyapLock not available. Skipping preservation pass.\n", "#ffaa00")
            return EditResult(
                success=True,
                method="ROME_only",
                weights_changed=weights_changed,
                notes="LyapLock unavailable — catastrophic forgetting not guarded."
            )

        try:
            _log("[LyapLock] 🔒 Starting Lyapunov stability preservation pass...\n", "#bc13fe")

            model_name = getattr(model.config, "_name_or_path", "gpt2")
            lyap_hparams_dir = _LYAPLOCK_ROOT / "hparams" / "LyapLock"
            lyap_hp = None

            if lyap_hparams_dir.exists():
                candidates = sorted(lyap_hparams_dir.glob("*.yaml"))
                if candidates:
                    lyap_hp = LyapLockHyperParams.from_hparams(candidates[0])

            if lyap_hp is None:
                _log("[LyapLock] ⚠  No HyperParams YAML found. Skipping.\n", "#ffaa00")
                return EditResult(
                    success=True,
                    method="ROME_only",
                    weights_changed=weights_changed,
                    notes="LyapLock YAML missing — skipped preservation pass."
                )

            num_layers = len(hparams.layers)
            lyap_kwargs = _build_lyaplock_defaults(num_layers)

            model, _, _ = apply_lyaplock_to_model(
                model=model,
                tok=tokenizer,
                requests=[rome_request],
                hparams=lyap_hp,
                copy=False,
                return_orig_weights=False,
                **lyap_kwargs
            )

            _log("[LyapLock] ✅ Preservation pass complete. Model stability ensured.\n", "#00ff00")

        except Exception as e:
            err = traceback.format_exc()
            _log(f"[LyapLock] ⚠  Preservation pass failed (ROME edit still applied): {e}\n", "#ffaa00")
            return EditResult(
                success=True,
                method="ROME_only",
                weights_changed=weights_changed,
                error_message=str(e),
                notes="LyapLock pass failed after successful ROME edit."
            )

        return EditResult(
            success=True,
            method="ROME+LyapLock",
            weights_changed=weights_changed,
            notes="Full ROME rank-1 edit + LyapLock Lyapunov stability pass applied."
        )
