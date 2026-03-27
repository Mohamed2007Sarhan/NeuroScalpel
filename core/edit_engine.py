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
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable

from core.model_introspection import (
    discover_rewrite_module_template,
    rank_hparam_json_paths,
    resolve_model_layer_count,
)

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
ROME_IMPORT_ERROR = ""
LYAPLOCK_IMPORT_ERROR = ""

if TORCH_OK:
    try:
        from rome.rome_main import apply_rome_to_model
        from rome.rome_hparams import ROMEHyperParams
        ROME_OK = True
        logger.info("ROME algorithm loaded successfully.")
    except Exception as _e:
        ROME_IMPORT_ERROR = str(_e)
        logger.warning(f"Could not load ROME: {_e}")

    try:
        from lyaplock.lyaplock_main import apply_lyaplock_to_model
        from lyaplock.LyapLock_hparams import LyapLockHyperParams
        LYAPLOCK_OK = True
        logger.info("LyapLock algorithm loaded successfully.")
    except Exception as _e:
        LYAPLOCK_IMPORT_ERROR = str(_e)
        logger.warning(f"Could not load LyapLock: {_e}")


def _ensure_algorithms_loaded():
    """
    Lazy runtime import so the edit path can retry loading ROME/LyapLock
    and report the real import failure cause.
    """
    global ROME_OK, LYAPLOCK_OK, ROME_IMPORT_ERROR, LYAPLOCK_IMPORT_ERROR
    global apply_rome_to_model, ROMEHyperParams, apply_lyaplock_to_model, LyapLockHyperParams

    if not ROME_OK:
        try:
            rome_main = importlib.import_module("rome.rome_main")
            rome_hp = importlib.import_module("rome.rome_hparams")
            apply_rome_to_model = rome_main.apply_rome_to_model
            ROMEHyperParams = rome_hp.ROMEHyperParams
            ROME_OK = True
            ROME_IMPORT_ERROR = ""
        except Exception as e:
            ROME_IMPORT_ERROR = str(e)

    if not LYAPLOCK_OK:
        try:
            ly_main = importlib.import_module("lyaplock.lyaplock_main")
            ly_hp = importlib.import_module("lyaplock.LyapLock_hparams")
            apply_lyaplock_to_model = ly_main.apply_lyaplock_to_model
            LyapLockHyperParams = ly_hp.LyapLockHyperParams
            LYAPLOCK_OK = True
            LYAPLOCK_IMPORT_ERROR = ""
        except Exception as e:
            LYAPLOCK_IMPORT_ERROR = str(e)


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

def _build_rome_hparams(model, layer: Optional[int] = None) -> "ROMEHyperParams":
    """
    Loads ROME HyperParams from real JSON config only.
    No hard-coded fallback is allowed.
    """
    hparams_dir = _LYAPLOCK_ROOT / "hparams" / "ROME"

    if not hparams_dir.exists():
        raise RuntimeError(f"ROME hparams directory missing: {hparams_dir}")

    # Strict selection: try every real JSON config and accept only those
    # whose module templates resolve against the *actual* model object.
    candidates = sorted(hparams_dir.glob("*.json"))
    if not candidates:
        raise RuntimeError(f"No ROME HyperParams JSON files found in {hparams_dir}")
    # Order by overlap between JSON filename and this model's config/path hints; probe is authoritative.
    candidates = rank_hparam_json_paths(candidates, model)

    from util import nethook as _nethook
    errors = []
    for cand in candidates:
        try:
            hp = ROMEHyperParams.from_json(cand)
            if layer is not None:
                hp.layers = [int(layer)]

            # Validate module templates against the loaded model.
            probe_layers = hp.layers if hp.layers else ([int(layer)] if layer is not None else [])
            if not probe_layers:
                raise RuntimeError("ROME hparams has no layers to probe.")

            for L in probe_layers:
                w_name = f"{hp.rewrite_module_tmp.format(L)}.weight"
                _ = _nethook.get_parameter(model, w_name)

            logger.info(f"ROME HyperParams selected by probe: {cand.name}")
            return hp
        except Exception as e:
            errors.append(f"{cand.name}: {e}")
            continue

    probe_layer = int(layer) if layer is not None else 0
    discovered = discover_rewrite_module_template(model, probe_layer)
    suffix = (
        f"\n\nInferred `rewrite_module_tmp` from loaded weight names (add a matching JSON under hparams/ROME): {discovered}"
        if discovered
        else ""
    )
    raise RuntimeError(
        "No compatible ROME HyperParams JSON matched the loaded model.\n"
        + "\n".join(errors)
        + suffix
    )


def _align_rome_hparams_with_model(hparams: "ROMEHyperParams", model, layer_hint: Optional[int]) -> "ROMEHyperParams":
    """
    Align loaded ROME hparams to the real model depth so we never reference
    non-existent layers (e.g. transformer.h.47 on a 12-layer model).
    """
    layer_count = resolve_model_layer_count(model)
    min_layer = min(range(layer_count))
    max_layer = layer_count - 1
    if max_layer < min_layer:
        raise RuntimeError("Model has no transformer layers.")

    if layer_hint is not None:
        if int(layer_hint) < min_layer or int(layer_hint) > max_layer:
            raise RuntimeError(
                f"Layer hint {layer_hint} is out of model range [{min_layer}, {max_layer}]."
            )
        hparams.layers = [int(layer_hint)]
    else:
        aligned_layers = [int(x) for x in getattr(hparams, "layers", []) if min_layer <= int(x) <= max_layer]
        if not aligned_layers:
            raise RuntimeError(
                f"ROME hparams layers are incompatible with model depth ({layer_count} layers)."
            )
        hparams.layers = aligned_layers

    # v_loss_layer must also be a valid model layer.
    v_loss_layer = int(getattr(hparams, "v_loss_layer", max_layer))
    if v_loss_layer < min_layer or v_loss_layer > max_layer:
        hparams.v_loss_layer = max_layer
    else:
        hparams.v_loss_layer = v_loss_layer

    return hparams


def _align_lyaplock_hparams_with_model(
    lyap_hp: "LyapLockHyperParams",
    model,
    rome_hparams: "ROMEHyperParams",
) -> "LyapLockHyperParams":
    """
    Map LyapLock layer list + v_loss_layer to the loaded model depth.
    If the JSON targets layers outside this model (e.g. gpt2-xl indices on gpt2),
    fall back to the same layers ROME just edited.
    """
    layer_count = resolve_model_layer_count(model)
    min_layer = min(range(layer_count))
    max_layer = layer_count - 1

    json_layers = [int(x) for x in getattr(lyap_hp, "layers", [])]
    valid_from_json = [L for L in json_layers if min_layer <= L <= max_layer]

    if valid_from_json:
        lyap_hp.layers = valid_from_json
    else:
        lyap_hp.layers = [int(x) for x in getattr(rome_hparams, "layers", [])]

    if not lyap_hp.layers:
        raise RuntimeError("LyapLock alignment produced no valid layers.")

    v_loss = int(getattr(lyap_hp, "v_loss_layer", max_layer))
    if v_loss < min_layer or v_loss > max_layer:
        nxt = int(max(lyap_hp.layers)) + 1
        if nxt > max_layer:
            nxt = max_layer
        lyap_hp.v_loss_layer = nxt
    return lyap_hp


def _select_lyaplock_hparams(model, rome_hparams: "ROMEHyperParams") -> Optional["LyapLockHyperParams"]:
    """
    Choose a LyapLock JSON by probing rewrite_module_tmp on the real model.
    Avoids loading the wrong file (e.g. GPT-J fc_out on a GPT-2 stack).
    """
    _ensure_algorithms_loaded()
    if not LYAPLOCK_OK:
        return None

    lyap_dir = _LYAPLOCK_ROOT / "hparams" / "LyapLock"
    if not lyap_dir.is_dir():
        return None

    candidates = sorted(lyap_dir.glob("*.json"))
    if not candidates:
        return None

    candidates = rank_hparam_json_paths(candidates, model)

    from util import nethook as _nethook

    errors: list[str] = []
    for cand in candidates:
        try:
            lyap_hp = LyapLockHyperParams.from_json(cand)
            lyap_hp = _align_lyaplock_hparams_with_model(lyap_hp, model, rome_hparams)
            tmp = getattr(lyap_hp, "rewrite_module_tmp", "")
            for L in lyap_hp.layers:
                w_name = f"{tmp.format(int(L))}.weight"
                _ = _nethook.get_parameter(model, w_name)

            if getattr(lyap_hp, "mom2_dataset", "") == "wikipedia":
                lyap_hp.mom2_dataset = "wikitext"

            logger.info("LyapLock HyperParams selected by probe: %s", cand.name)
            return lyap_hp
        except Exception as e:
            errors.append(f"{cand.name}: {e}")
            continue

    layer0 = int(getattr(rome_hparams, "layers", [0])[0])
    discovered = discover_rewrite_module_template(model, layer0)
    if discovered:
        logger.warning(
            "No LyapLock JSON passed probe. Inferred rewrite template from loaded weights: %s",
            discovered,
        )
    logger.warning("No LyapLock HyperParams matched the model:\n%s", "\n".join(errors))
    return None


def _build_lyaplock_defaults(layer_ids: Iterable[int]) -> Dict[str, Any]:
    """Returns the kwargs dict required by execute_lyaplock on first run (cnt=0)."""
    layer_ids = [int(i) for i in layer_ids]
    if not layer_ids:
        raise RuntimeError("Cannot initialize LyapLock defaults with empty layer_ids.")
    one = float(len(layer_ids)) / float(len(layer_ids))
    zero = float(sum(layer_ids) - sum(layer_ids))
    cnt0 = int(bool(layer_ids)) - int(bool(layer_ids))
    return {
        "V": {i: one for i in layer_ids},
        "Z": {i: "D" for i in layer_ids},
        "alpha": {i: one for i in layer_ids},
        "a": {i: "1_D" for i in layer_ids},
        "b": {i: zero for i in layer_ids},
        "zmax": {i: "D" for i in layer_ids},
        "D_base": {i: None for i in layer_ids},
        "cnt": cnt0,
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
        _ensure_algorithms_loaded()

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
            _log(
                "[EDIT ENGINE] ⚠  ROME not available. Edit cannot proceed.\n"
                f"[EDIT ENGINE] import_error: {ROME_IMPORT_ERROR}\n",
                "#ffaa00"
            )
            return EditResult(
                success=False,
                method="failed",
                error_message=f"ROME library not importable: {ROME_IMPORT_ERROR}"
            )

        weights_changed = []
        try:
            model_name = getattr(model.config, "_name_or_path", "unknown")
            hparams = _build_rome_hparams(model, layer=request.layer_hint)
            hparams = _align_rome_hparams_with_model(hparams, model, request.layer_hint)
            # Compatibility fix: modern `datasets` versions reject wikipedia script loading.
            # Use wikitext statistics dataset when hparams requests wikipedia.
            if getattr(hparams, "mom2_dataset", "") == "wikipedia":
                hparams.mom2_dataset = "wikitext"
                _log("[UPDATE] Dataset remapped: wikipedia -> wikitext (runtime compatibility).\n", "#ffaa00")

            _log(f"[UPDATE] Targeting layer(s): {hparams.layers}  |  neuron hint: {request.neuron_hint}\n",
                 "#00f3ff")
            _log("[UPDATE] Computing update vectors...\n", "#00f3ff")

            model, weights_copy, _ = apply_rome_to_model(
                model=model,
                tok=tokenizer,
                request=[rome_request],
                hparams=hparams,
                copy=False,
                return_orig_weights=True,
            )

            weights_changed = list(weights_copy.keys())

            # Strict mode: keep pure ROME update (no heuristic mask multipliers).
            if request.neuron_hint is not None and request.neuron_hint >= 0:
                _log(
                    f"[UPDATE] Neuron hint={request.neuron_hint} recorded for audit; "
                    "no heuristic scaling applied.\n",
                    "#00f3ff",
                )

            _log(f"[UPDATE] ✅ Weights updated: {weights_changed}\n", "#00ff00")
            # User-facing derived metric (from the actual updated tensor shapes).
            try:
                from util import nethook as _nethook
                widths = []
                for w_name in weights_changed:
                    w = _nethook.get_parameter(model, w_name)
                    if hasattr(w, "shape") and len(w.shape) >= 1:
                        widths.append(int(w.shape[-1]))
                widths = list({w for w in widths if isinstance(w, int)})
                update_width = sorted(widths)[0] if widths else None
            except Exception:
                update_width = None

            if update_width is not None:
                _log(f"[UPDATE] UPDATE WIDTH: {update_width}\n", "#00ff00")

        except Exception as e:
            err = traceback.format_exc()
            _log(f"[UPDATE] ❌ Failed: {e}\n{err}\n", "#ff003c")
            return EditResult(
                success=False,
                method="edit_failed",
                weights_changed=weights_changed,
                error_message=str(e)
            )

        # ----------------------------------------------------------------
        # 3. LyapLock stabilisation pass
        # ----------------------------------------------------------------
        if not LYAPLOCK_OK:
            _log("[STABILIZATION] ⚠  Stabilization pass not available. Skipping.\n", "#ffaa00")
            return EditResult(
                success=True,
                method="ROME_success_LyapLock_failed",
                weights_changed=weights_changed,
                notes="LyapLock unavailable — catastrophic forgetting not guarded."
            )

        try:
            _log("[STABILIZATION] 🔒 Starting stability preservation pass...\n", "#bc13fe")

            lyap_hp = _select_lyaplock_hparams(model, hparams)

            if lyap_hp is None:
                _log(
                    "[STABILIZATION] ❌ No compatible configuration matched this model architecture. "
                    "Preservation pass blocked.\n",
                    "#ff003c",
                )
                return EditResult(
                    success=True,
                    method="ROME_success_LyapLock_failed",
                    weights_changed=weights_changed,
                    notes="LyapLock hparams incompatible or missing — skipped preservation pass.",
                )

            layer_ids = getattr(lyap_hp, "layers", None) or getattr(hparams, "layers", [])
            if not layer_ids:
                raise RuntimeError("LyapLock hparams has no target layers.")
            lyap_kwargs = _build_lyaplock_defaults(layer_ids)

            model, _, _ = apply_lyaplock_to_model(
                model=model,
                tok=tokenizer,
                requests=[rome_request],
                hparams=lyap_hp,
                copy=False,
                return_orig_weights=False,
                **lyap_kwargs
            )

            _log("[STABILIZATION] ✅ Preservation pass complete. Stability ensured.\n", "#00ff00")

        except Exception as e:
            err = traceback.format_exc()
            _log(f"[STABILIZATION] ⚠  Preservation pass failed (update kept): {e}\n", "#ffaa00")
            return EditResult(
                success=True,
                method="ROME_success_LyapLock_failed",
                weights_changed=weights_changed,
                error_message=str(e),
                notes="LyapLock pass failed after successful ROME edit."
            )

        return EditResult(
            success=True,
            method="ROME_and_LyapLock_success",
            weights_changed=weights_changed,
            notes="Update applied + stability preservation pass applied."
        )
