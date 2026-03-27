"""
model_introspection.py
======================
Architecture-agnostic introspection for whatever causal LM is loaded.

Used to rank ROME/LyapLock JSON configs and to resolve depths/paths from the
actual nn.Module + HF config (not from assumed model family names in code).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def get_model_identity_hints(model) -> set[str]:
    """
    Tokens derived only from the loaded model's HF config and path string.
    Used to order hparam JSON files (filename closeness) before structural probe.
    """
    hints: set[str] = set()
    cfg = getattr(model, "config", None)
    if cfg is None:
        return hints

    for attr in ("model_type", "_name_or_path"):
        v = getattr(cfg, attr, None)
        if not v:
            continue
        s = str(v).lower().strip()
        if s:
            hints.add(s)
        for tok in s.replace("\\", "/").replace("-", "_").split("/"):
            t = tok.strip("._- ")
            if len(t) >= 2:
                hints.add(t.lower())

    arch = getattr(cfg, "architectures", None)
    if isinstance(arch, (list, tuple)):
        for a in arch:
            if a:
                hints.add(str(a).lower())

    return {h for h in hints if h}


def _hparam_path_relevance(path: Path, hints: set[str]) -> int:
    """Higher = filename looks more like this model (pure string overlap)."""
    stem = path.stem.lower().replace("-", "_")
    score = 0
    for h in hints:
        if not h:
            continue
        key = h.lower().replace("-", "_")
        if len(key) < 2:
            continue
        if key in stem:
            score += len(key)
    return score


def rank_hparam_json_paths(paths: Iterable[Path], model) -> list[Path]:
    """
    Order JSON hparam candidates: best filename match to model identity first,
    then alphabetical. Actual compatibility is still decided by weight probe.
    """
    paths = list(paths)
    hints = get_model_identity_hints(model)

    def sort_key(p: Path) -> tuple:
        rel = _hparam_path_relevance(p, hints)
        return (-rel, p.name.lower())

    return sorted(paths, key=sort_key)


def resolve_model_layer_count(model) -> int:
    """Transformer block count from module tree or config (any causal LM we support)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(
        model.model.decoder, "layers"
    ):
        return len(model.model.decoder.layers)
    cfg_n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if cfg_n is None:
        raise RuntimeError("Cannot resolve model layer count from model structure/config.")
    return int(cfg_n)


def discover_rewrite_module_template(model, layer_idx: int) -> Optional[str]:
    """
    Inspect named_parameters() on the loaded model and infer a
    rewrite_module_tmp pattern (``...{}``.mlp....'') for the given block index.

    This is a fallback signal / logging aid when adding new JSON profiles; the
    edit engine still validates with nethook.get_parameter.
    """
    tag = str(int(layer_idx))
    for name, _ in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if f".{tag}." not in name:
            continue
        low = name.lower()
        if any(x in low for x in ("lm_head", "embed", "wte", "wpe", "tok_embeddings")):
            continue
        if not any(x in low for x in ("mlp", "ffn", "feed_forward", "mlp_residual")):
            continue
        base = name[: -len(".weight")]
        needle = f".{tag}."
        if needle not in base:
            continue
        return base.replace(needle, ".{}.", 1)
    return None
