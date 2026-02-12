from itertools import combinations
from typing import Dict, Union, List

import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from collections import OrderedDict
import logging

import pyarrow as pa


def gather_index(x: Union[Dict, Tensor, List, None], index: Tensor):
    if x is None:
        return None
    if isinstance(x, List):
        return [element[index] for element in x]
    if isinstance(x, Tensor):
        return x[index]
    out = OrderedDict()
    for k, v in x.items():
        out[k] = gather_index(v, index)
    return out


def get_transition(global_step, start_step, duration_steps, device):
    step_tensor = torch.tensor(global_step, dtype=torch.float32, device=device)
    progress = torch.clamp((step_tensor - start_step) / duration_steps, 0.0, 1.0)
    t = 0.5 * (1 - torch.cos(np.pi * progress))
    return t


# For Torch JD
def check_param_overlap(task_param_sets, task_names, model, current_step=None, check_every=1, verbose=False):
    """
    Check and optionally list overlapping parameters between tasks.

    Args:
        task_param_sets (List[List[torch.nn.Parameter]]): List of param lists per task.
        task_names (List[str]): Task names corresponding to each param set.
        model (torch.nn.Module): Model to extract parameter names.
        current_step (int, optional): If set, only check every `check_every` steps.
        check_every (int): Frequency of checking.
        verbose (bool): If True, prints names of overlapping parameters.

    Returns:
        Dict[Tuple[str, str], Set[str]]: Overlap mapping from task pairs to parameter names.
    """
    if current_step is not None and current_step % check_every != 0:
        return {}

    # Build reverse lookup of parameter id -> name
    param_id_to_name = {id(p): n for n, p in model.named_parameters()}

    id_sets = {name: set(map(id, params)) for name, params in zip(task_names, task_param_sets)}
    overlaps = {}

    for name1, name2 in combinations(task_names, 2):
        shared_ids = id_sets[name1] & id_sets[name2]
        shared_names = {param_id_to_name[pid] for pid in shared_ids if pid in param_id_to_name}
        overlaps[(name1, name2)] = shared_names
        print(f"[TorchJD Overlap] {name1} ↔ {name2}: {len(shared_names)} shared parameters")
        if verbose and shared_names:
            for pname in sorted(shared_names):
                print(f"    ↳ {pname}")

    return overlaps


def print_params_used_by_loss(loss, model, include_shapes=True, verbose=True):
    """
    Print parameter names from the model that are used in computing the given loss.

    Args:
        loss (torch.Tensor): Scalar loss tensor.
        model (torch.nn.Module): The model whose parameters will be checked.
        include_shapes (bool): Whether to print parameter shapes.
        verbose (bool): Whether to print output (vs return lists).

    Returns:
        used_names (List[str]): List of used parameter names.
    """
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    names, params = zip(*named_params)

    # Get gradients w.r.t. all parameters
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False
    )

    used_names = []
    for name, param, grad in zip(names, params, grads):
        if grad is not None:
            used_names.append(name)
            if verbose:
                msg = f"✅ USED: {name}"
                if include_shapes:
                    msg += f" — shape: {tuple(param.shape)}"
                print(msg)

    return used_names


def safe_load_state(model: nn.Module, state_dict: dict, prefix_to_strip: str = "model.", verbose=True) -> None:
    logger = logging.getLogger(f"{__name__}.safe_load_state")

    # Strip prefix (e.g., "model.")
    clean_sd = {k.replace(prefix_to_strip, ""): v for k, v in state_dict.items()}

    for k, v in clean_sd.items():
        if "_normalizer" in k:
            if verbose:
                logger.warning(f"[safe_load_state] ⚠️ Ignored normalizer parameter: {k}")
    clean_sd = {k: v for k, v in clean_sd.items() if "_normalizer" not in k}

    model_sd = model.state_dict()
    filtered_sd = {}
    for k, v in clean_sd.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                filtered_sd[k] = v
            elif verbose:
                logger.warning(f"[safe_load_state] ⚠️ Shape mismatch: {k} (ckpt: {v.shape}, model: {model_sd[k].shape})")
        elif verbose:
            logger.warning(f"[safe_load_state] ⚠️ Unmatched key (ignored): {k}")

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if verbose:
        logger.info(f"[safe_load_state] ✅ Loaded with {len(filtered_sd)} keys.")
        logger.info(f"[safe_load_state]   🔸 Missing keys: {missing}")
        logger.info(f"[safe_load_state]   🔸 Unexpected keys: {unexpected}")
