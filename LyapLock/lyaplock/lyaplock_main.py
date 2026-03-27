import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from rome.layer_stats import layer_stats
from rome.layer_stats_ab_t import layer_stats_vk_t, layer_stats_kk_t, layer_stats_vv_t
from util import nethook
from util.generate import generate_fast
from util.globals import *
from util.nethook import get_module

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .LyapLock_hparams import LyapLockHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
V0K0_T_CACHE = {}
K0K0_T_CACHE = {}
V0V0_T_CACHE = {}


def _weight_as_vk_matrix(weight: torch.Tensor, vk_t: torch.Tensor) -> torch.Tensor:
    """
    Return weight matrix oriented as [V, K] to match VK/KK moments.
    Chooses orientation by actual shape compatibility, not model-name heuristics.
    """
    if weight.dim() != 2:
        raise RuntimeError(f"Expected 2D weight tensor, got rank={weight.dim()}.")
    if weight.shape[0] == vk_t.shape[0]:
        return weight
    if weight.shape[1] == vk_t.shape[0]:
        return weight.T
    raise RuntimeError(
        f"Cannot align weight shape {tuple(weight.shape)} with VK shape {tuple(vk_t.shape)}."
    )

def apply_lyaplock_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: LyapLockHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    model_device = next(model.parameters()).device
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas, kwargs = execute_lyaplock(model, tok, requests, hparams, cache_template=cache_template, **kwargs)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(model_device), val_mat.to(model_device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy, kwargs

def compute_fro_by_trace(KK_T, W, VK_T, VV_T):
    return (KK_T @ W.T @ W).trace() - 2 * (VK_T @ W.T).trace() + (VV_T).trace()

def compute_fro_by_norm(W, K, V):
    return (W @ K - V).norm(p = 'fro') ** 2

def execute_lyaplock(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: LyapLockHyperParams,
    cache_template: Optional[str] = None,
    **kwargs,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the LyapLock update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"LyapLock request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(next(model.parameters()).device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False

        v0k0_T = get_v0k0_T(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        k0k0_T = get_k0k0_T(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        v0v0_T = get_v0v0_T(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )


        # Compute update in double precision
        layer_ks, targets, v0k0_T, k0k0_T, v0v0_T = (
            layer_ks.double(),
            targets.double(),
            v0k0_T.double(),
            k0k0_T.double(),
            v0v0_T.double(),
        )
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers

        V = kwargs['V'][layer]
        Z = kwargs['Z'][layer]
        alpha = kwargs['alpha'][layer]
        a = kwargs['a'][layer]
        b = kwargs['b'][layer]
        zmax = kwargs['zmax'][layer]
        D_base = kwargs['D_base'][layer]

        if kwargs['cnt'] == 0:
            with torch.no_grad():
                W = _weight_as_vk_matrix(
                    weights_copy[f"{hparams.rewrite_module_tmp.format(layer)}.weight"].double(),
                    v0k0_T,
                )
                left = (resid @ layer_ks.T) + hparams.mom2_update_weight * (v0k0_T - W.double() @ k0k0_T)
    
            A = layer_ks @ layer_ks.T + hparams.mom2_update_weight * k0k0_T
            I = torch.eye(A.size(0), dtype=A.dtype).to(A.device)

            adj_k = torch.linalg.solve(
                A, I,
            )
        else:
            with torch.no_grad():
                W = _weight_as_vk_matrix(
                    weights_copy[f"{hparams.rewrite_module_tmp.format(layer)}.weight"].double(),
                    v0k0_T,
                )
                left = V * (resid @ layer_ks.T) + a * Z * hparams.mom2_update_weight * (v0k0_T - W.double() @ k0k0_T)

                if 'Pre_Cache' in kwargs and kwargs['Pre_Cache'][layer] != []:
                    left += V * (kwargs['Pre_Cache'][layer][0].to(W.device) - W.double() @ kwargs['Pre_Cache'][layer][1].to(W.device))

            A = V * layer_ks @ layer_ks.T + a * Z * hparams.mom2_update_weight * k0k0_T
            
            if 'Pre_Cache' in kwargs and kwargs['Pre_Cache'][layer] != []:
                A += V * (kwargs['Pre_Cache'][layer][1].to(W.device))

            I = torch.eye(A.size(0), dtype=A.dtype).to(A.device)

            adj_k = torch.linalg.solve(
                A, I,
            )

        upd_matrix = left @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)


        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                left.detach().cpu(),
            )

            cur_zs_new = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets_new = zs - cur_zs_new
            print("z_new error", torch.linalg.norm(targets_new, dim=0).mean())

            repeat_factor = (layer_ks.size(1) // targets_new.size(1))
            targets_new = targets_new.repeat_interleave(repeat_factor, dim=1)

            targets_new = targets_new.double()

            resid_new = targets_new / (len(hparams.layers) - i)
            
            if D_base == None:
                W_cur = _weight_as_vk_matrix(weights[weight_name].double(), v0k0_T)
                D_base = compute_fro_by_trace(k0k0_T, W_cur, v0k0_T, v0v0_T)
                print(f'Layer {layer} D_base = {D_base}')
            kwargs['D_base'][layer] = D_base
            if a == '1_D':
                kwargs['a'][layer] = 1 / (alpha * D_base)
                a = kwargs['a'][layer]
            elif isinstance(a, str) and '_sqrtD' in a:
                kwargs['a'][layer] = float(a.split('_')[0]) / (alpha * D_base).sqrt()
                a = kwargs['a'][layer]
            if Z == 'D':
                kwargs['Z'][layer] = alpha * D_base
                Z = kwargs['Z'][layer]
            elif isinstance(Z, str) and 'sqrtD' in Z:
                kwargs['Z'][layer] = float(Z.replace('sqrtD', '')) * (alpha * D_base).sqrt()
                Z = kwargs['Z'][layer]
            if zmax == 'D':
                kwargs['zmax'][layer] = alpha * D_base
                zmax = kwargs['zmax'][layer]
            elif isinstance(zmax, str) and 'sqrtD' in zmax:
                kwargs['zmax'][layer] = float(zmax.replace('sqrtD', '')) * (alpha * D_base).sqrt()
                zmax = kwargs['zmax'][layer]
            
            W_cur = _weight_as_vk_matrix(weights[weight_name].double(), v0k0_T)
            kwargs['Z'][layer] = max(
                Z + a * (compute_fro_by_trace(k0k0_T, W_cur, v0k0_T, v0v0_T) - alpha * D_base) + b,
                zmax,
            )


            if 'Pre_Cache' in kwargs:
                W_cur = _weight_as_vk_matrix(weights[weight_name].double(), v0k0_T)
                layer_v = W_cur @ layer_ks
                if kwargs['Pre_Cache'][layer] == []:
                    kwargs['Pre_Cache'][layer] = [(layer_v @ layer_ks.T).cpu(), (layer_ks @ layer_ks.T).cpu(), (layer_v @ layer_v.T).cpu()]
                else:
                    kwargs['Pre_Cache'][layer][0] += (layer_v @ layer_ks.T).cpu()
                    kwargs['Pre_Cache'][layer][1] += (layer_ks @ layer_ks.T).cpu()
                    kwargs['Pre_Cache'][layer][2] += (layer_v @ layer_v.T).cpu()

        # Clear GPU memory
        k0k0_T.cpu()
        v0k0_T.cpu()
        v0v0_T.cpu()
        for x in [layer_ks, cur_zs, targets, cur_zs_new, targets_new]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas, kwargs


def get_v0k0_T(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in V0K0_T_CACHE or force_recompute:
        stat = layer_stats_vk_t(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["VK_T"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        V0K0_T_CACHE[key] = stat.VK_T.moment().float().to("cpu")

    device = next(model.parameters()).device
    return torch.inverse(V0K0_T_CACHE[key].to(device)) if inv else V0K0_T_CACHE[key].to(device)

def get_k0k0_T(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in K0K0_T_CACHE or force_recompute:
        stat = layer_stats_kk_t(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["KK_T"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        K0K0_T_CACHE[key] = stat.KK_T.moment().float().to("cpu")

    device = next(model.parameters()).device
    return torch.inverse(K0K0_T_CACHE[key].to(device)) if inv else K0K0_T_CACHE[key].to(device)

def get_v0v0_T(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in V0V0_T_CACHE or force_recompute:
        stat = layer_stats_vv_t(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["VV_T"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        V0V0_T_CACHE[key] = stat.VV_T.moment().float().to("cpu")

    device = next(model.parameters()).device
    return torch.inverse(V0V0_T_CACHE[key].to(device)) if inv else V0V0_T_CACHE[key].to(device)

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    device = next(model.parameters()).device
    return torch.inverse(COV_CACHE[key].to(device)) if inv else COV_CACHE[key].to(device)


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by LyapLock does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
