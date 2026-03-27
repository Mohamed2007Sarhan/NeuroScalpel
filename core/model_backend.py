"""
model_backend.py
================
Core backend for NeuroScalpel.

ModelManager  — loads local or HuggingFace models and extracts real 3D
                geometric embeddings via PCA for the point-cloud visualizer.

apply_real_edit — top-level function called by the UI to execute a
                  ROME + LyapLock weight update on the active model.
"""

import json
import numpy as np
import logging
import traceback
from typing import Optional, Tuple

logger = logging.getLogger("NeuroScalpel.ModelBackend")

# Optional heavy imports
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.decomposition import PCA
    TORCH_OK = True
except Exception as _e:
    TORCH_OK = False
    logger.warning(f"PyTorch/Transformers unavailable: {_e}")


class ModelManager:
    """
    Manages the currently active HuggingFace model + tokenizer.
    Keeps them in memory so the edit engine can reuse them without
    reloading from disk on every edit.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name: str = ""
        self._device = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_local_model(self, model_path: str, log_callback=None) -> bool:
        """Loads a model from a local folder path."""
        self._log(log_callback, f"Loading local model from: {model_path}", "#00f3ff")
        return self._load(model_path, log_callback)

    def load_hf_model(self, model_id: str, log_callback=None) -> bool:
        """Downloads and loads a model from HuggingFace Hub."""
        self._log(log_callback, f"Connecting to HuggingFace: {model_id}", "#00f3ff")
        return self._load(model_id, log_callback)

    def _load(self, identifier: str, log_callback=None) -> bool:
        if not TORCH_OK:
            self._log(log_callback, "PyTorch unavailable — cannot load model.", "#ff003c")
            return False

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if self._device.type == "cuda" else torch.float32

            self._log(log_callback, f"Device: {self._device} | dtype: {dtype}", "#00f3ff")

            self.tokenizer = AutoTokenizer.from_pretrained(identifier)
            # Many decoder-only models (e.g., GPT-2) ship without a pad token.
            # ROME/LyapLock generation utilities may request padding=True.
            if getattr(self.tokenizer, "pad_token", None) is None:
                eos = getattr(self.tokenizer, "eos_token", None)
                if eos is None:
                    raise RuntimeError("Tokenizer has no pad_token and no eos_token to use as pad.")
                self.tokenizer.pad_token = eos
                self._log(log_callback, f"Tokenizer pad_token set to eos_token ({repr(eos)}).", "#00ff88")
            self.model = AutoModelForCausalLM.from_pretrained(
                identifier,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            self.model = self.model.to(self._device)
            self.model.eval()
            self.model_name = identifier

            self._log(log_callback, f"Model '{identifier}' loaded successfully ✅", "#00ff00")
            return True

        except Exception as e:
            self._log(log_callback,
                      f"Model load failed: {e}\n{traceback.format_exc()}", "#ff003c")
            return False

    # ------------------------------------------------------------------
    # 3D Geometry extraction
    # ------------------------------------------------------------------

    def get_real_weights(
        self,
        model_id: Optional[str] = None,
        num_points: Optional[int] = None,
        log_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts a 3D PCA projection of the model's embedding space.
        Organises points by layer so the visualizer can show them as
        Z-separated slabs (one slab per transformer layer).

        Returns
        -------
        points : (N, 3)   float32 array  —  PCA-projected coordinates
        ids    : (N,)     int array      —  token / neuron indices
        layer_map : dict  layer_idx → (start_row, end_row) in points array
        """
        target_id = model_id or self.model_name

        if not TORCH_OK:
            self._log(log_callback, "PyTorch not available — returning empty.", "#ff003c")
            return np.zeros((1, 3), dtype=np.float32), np.zeros(1, dtype=np.int64)

        try:
            self._log(log_callback,
                      f"Extracting real PCA embedding geometry for: {target_id}", "#00f3ff")

            # Use already-loaded model if available, else load a lightweight one
            if self.model is not None and self.model_name == target_id:
                model_ref = self.model
                do_cleanup = False
            else:
                self._log(log_callback,
                          "Loading model for geometry extraction (low_cpu_mem_usage)...", "#00f3ff")
                model_ref = AutoModelForCausalLM.from_pretrained(
                    target_id,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
                do_cleanup = True

            # Pull input embeddings
            embeddings = (
                model_ref.get_input_embeddings()
                .weight.detach().float().cpu().numpy()
            )

            vocab_size = embeddings.shape[0]
            sample_limit = num_points if (num_points is not None and num_points > 0) else vocab_size
            if vocab_size > sample_limit:
                indices = np.random.choice(vocab_size, sample_limit, replace=False)
                sampled = embeddings[indices]
                ids = indices
            else:
                sampled = embeddings
                ids = np.arange(vocab_size)

            pca = PCA(n_components=3)
            points = pca.fit_transform(sampled).astype(np.float32)

            if do_cleanup:
                del model_ref
                if TORCH_OK and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._log(log_callback,
                      f"Extracted {len(ids)} vectors — PCA variance: "
                      f"{pca.explained_variance_ratio_.sum():.2%}", "#00ff00")

            return points, ids

        except Exception as e:
            self._log(log_callback,
                      f"Geometry extraction failed: {e}\n{traceback.format_exc()}", "#ff003c")
            return np.zeros((1, 3), dtype=np.float32), np.zeros(1, dtype=np.int64)

    def get_layer_neuron_geometry(
        self,
        num_neurons_per_layer: Optional[int] = None,
        log_callback=None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Extracts a 3D geometry organised by transformer layer.
        Each layer occupies a fixed Z-plane. Neurons within the layer
        are scattered in the X-Y plane using PCA of the FFN weight rows.

        Returns
        -------
        points    : (N, 3) float32
        labels    : (N,)   int  (layer index for each point)
        layer_map : dict   layer_idx → list of row indices in `points`
        """
        if not TORCH_OK or self.model is None:
            self._log(log_callback, "Model not loaded — cannot extract layer geometry.", "#ff003c")
            return np.zeros((1, 3), np.float32), np.zeros(1, np.int32), {}

        try:
            num_layers = self._get_num_layers()
            if num_layers is None or num_layers <= 0:
                self._log(log_callback, "Could not resolve model layer count.", "#ff003c")
                return np.zeros((1, 3), np.float32), np.zeros(1, np.int32), {}

            all_points = []
            all_labels = []
            layer_map = {}
            layer_xy_spreads = []

            self._log(log_callback,
                      f"Building layered 3D geometry for {num_layers} layers...", "#00f3ff")

            for layer_i in range(num_layers):
                # Try to get the FFN weight for this layer
                weight = self._extract_ffn_weight(layer_i)
                if weight is None:
                    continue

                dynamic_n = int(np.sqrt(max(weight.shape[0], 1)))
                target_n = dynamic_n if (num_neurons_per_layer is None or num_neurons_per_layer <= 0) else num_neurons_per_layer
                n = min(target_n, weight.shape[0])
                indices = np.random.choice(weight.shape[0], n, replace=False)
                w_sample = weight[indices].astype(np.float32)

                if w_sample.shape[1] >= 2:
                    pca = PCA(n_components=2)
                    xy = pca.fit_transform(w_sample)
                else:
                    xy = w_sample[:, :2] if w_sample.shape[1] >= 2 else np.zeros((n, 2))

                xy_spread = float(np.linalg.norm(np.std(xy, axis=0))) if len(xy) > 1 else 0.0
                layer_xy_spreads.append(xy_spread)
                z = np.full((n, 1), layer_i, dtype=np.float32)
                pts = np.hstack([xy, z])

                start = len(all_points)
                all_points.extend(pts)
                all_labels.extend([layer_i] * n)
                layer_map[layer_i] = list(range(start, start + n))

            points_arr = np.array(all_points, dtype=np.float32)
            labels_arr = np.array(all_labels, dtype=np.int32)
            if len(points_arr) > 0 and layer_xy_spreads:
                spacing = float(np.median([s for s in layer_xy_spreads if s > 0.0])) if any(
                    s > 0.0 for s in layer_xy_spreads
                ) else 1.0
                points_arr[:, 2] = points_arr[:, 2] * spacing

            self._log(log_callback,
                      f"Layer geometry built: {len(points_arr)} neurons across {num_layers} layers ✅",
                      "#00ff00")

            return points_arr, labels_arr, layer_map

        except Exception as e:
            self._log(log_callback,
                      f"Layer geometry extraction failed: {e}\n{traceback.format_exc()}", "#ff003c")
            return np.zeros((1, 3), np.float32), np.zeros(1, np.int32), {}

    def _extract_ffn_weight(self, layer_i: int) -> Optional[np.ndarray]:
        """Returns the FFN c_proj weight matrix for a given layer (supports GPT-2 and LLaMA)."""
        try:
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                mlp = self.model.transformer.h[layer_i].mlp
                # GPT-2 style c_proj
                w = mlp.c_proj.weight.detach().float().cpu().numpy()
                return w
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                mlp = self.model.model.layers[layer_i].mlp
                w = mlp.down_proj.weight.detach().float().cpu().numpy()
                return w
        except Exception:
            pass
        return None

    def _get_num_layers(self) -> Optional[int]:
        """Resolve the true transformer block count from model structure."""
        try:
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                return len(self.model.transformer.h)
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return len(self.model.model.layers)
            if hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
                return len(self.model.model.decoder.layers)
            cfg_layers = getattr(self.model.config, "num_hidden_layers", None)
            return int(cfg_layers) if cfg_layers is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log(cb, msg: str, color: str = "#00f3ff"):
        logger.info(msg)
        if cb:
            cb(f"{msg}\n", color)


# ------------------------------------------------------------------
# Public top-level edit function (called from main_window)
# ------------------------------------------------------------------

def apply_real_edit(
    model_manager: "ModelManager",
    subject: str,
    prompt_template: str,
    target_new: str,
    target_old: str,
    layer_hint: Optional[int],
    neuron_hint: Optional[int] = None,
    log_callback=None
) -> dict:
    """
    Top-level function wiring ModelManager → EditEngine.
    Returns {"success": bool, "method": str, "weights": list, "error": str}
    """
    from core.edit_engine import ROMEEditEngine, EditRequest

    if model_manager.model is None or model_manager.tokenizer is None:
        return {
            "success": False,
            "method": "failed",
            "weights": [],
            "error": "No model loaded in ModelManager."
        }

    req = EditRequest(
        subject=subject,
        prompt_template=prompt_template,
        target_new=target_new,
        target_old=target_old,
        layer_hint=layer_hint,
        neuron_hint=neuron_hint
    )

    result = ROMEEditEngine.apply_edit(
        model=model_manager.model,
        tokenizer=model_manager.tokenizer,
        request=req,
        log_callback=log_callback
    )

    method = result.method
    success = bool(result.success)
    post_checks = {
        "has_weights": len(result.weights_changed) > 0,
        "model_present": model_manager.model is not None,
        "tokenizer_present": model_manager.tokenizer is not None,
    }
    if method not in {
        "ROME_and_LyapLock_success",
        "ROME_success_LyapLock_failed",
        "edit_failed",
    }:
        if success:
            method = "ROME_success_LyapLock_failed"
        else:
            method = "edit_failed"

    notes = result.notes or ""
    checks_blob = json.dumps(post_checks, ensure_ascii=True)
    notes = f"{notes}\npost_checks={checks_blob}".strip()

    return {
        "success": success,
        "method": method,
        "weights": result.weights_changed,
        "error": result.error_message,
        "notes": notes,
        "post_checks": post_checks,
    }
