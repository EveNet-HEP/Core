import torch.nn as nn
from collections import defaultdict
import torch
from torch import Tensor
from typing import Dict, Optional, Iterable


class DebugHookManager:
    def __init__(self, track_forward=True, track_backward=True, save_values=False):
        self.forward_hooks = []
        self.backward_hooks = []
        self.grad_hooks = []
        self.save_values = save_values

        self.forward_outputs = defaultdict(list)
        self.backward_grads = defaultdict(list)

        self.track_forward = track_forward
        self.track_backward = track_backward

    def check_forward(self, name):
        def hook(module, input, output):
            # Check output
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"[NaN Detected] Forward output of {name}")
                if torch.isinf(output).any():
                    print(f"[Inf Detected] Forward output of {name}")
                if self.save_values:
                    self.forward_outputs[name].append(output.detach().cpu())
            # Check inputs
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    if torch.isnan(inp).any():
                        print(f"[NaN Detected] Forward input {i} of {name}")
                    if torch.isinf(inp).any():
                        print(f"[Inf Detected] Forward input {i} of {name}")

        return hook

    def check_backward(self, name):
        def hook(module, grad_input, grad_output):
            for i, g in enumerate(grad_input):
                if isinstance(g, torch.Tensor):
                    if torch.isnan(g).any():
                        print(f"[NaN Detected] Grad input {i} of {name}")
                    if torch.isinf(g).any():
                        print(f"[Inf Detected] Grad input {i} of {name}")
                    if self.save_values:
                        self.backward_grads[name].append(g.detach().cpu())

        return hook

    def check_param_grad(self, name, param):
        def hook(grad):
            if grad is not None:
                if torch.isnan(grad).any():
                    print(f"[NaN Detected] Grad of param {name}")
                if torch.isinf(grad).any():
                    print(f"[Inf Detected] Grad of param {name}")
                if self.save_values:
                    self.backward_grads[f"param::{name}"].append(grad.detach().cpu())

        return hook

    def attach_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, nn.Module) and len(list(module.children())) == 0:  # only leaf modules
                if self.track_forward:
                    fh = module.register_forward_hook(self.check_forward(name))
                    self.forward_hooks.append(fh)
                if self.track_backward:
                    bh = module.register_full_backward_hook(self.check_backward(name))
                    self.backward_hooks.append(bh)

        for name, param in model.named_parameters():
            if param.requires_grad:
                gh = param.register_hook(self.check_param_grad(name, param))
                self.grad_hooks.append(gh)

    def remove_hooks(self):
        for h in self.forward_hooks + self.backward_hooks:
            h.remove()
        self.forward_hooks.clear()
        self.backward_hooks.clear()
        self.grad_hooks.clear()
        print("✅ All hooks removed.")

    def dump_debug_data(self):
        # Optional utility: Save collected outputs/grads to disk or analyze
        print("🔍 Dumped forward activations:")
        for k, v in self.forward_outputs.items():
            print(f"{k}: {len(v)} tensors")

        print("🔍 Dumped backward gradients:")
        for k, v in self.backward_grads.items():
            print(f"{k}: {len(v)} tensors")


import time
from functools import wraps
from collections import defaultdict
from lightning.pytorch.loggers import WandbLogger

# Global dictionary to store function stats
function_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})


def time_decorator(name=None):
    def wrapper(func):
        key = name or func.__qualname__

        @wraps(func)
        def timed_fn(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            function_stats[key]["count"] += 1
            function_stats[key]["total_time"] += elapsed

            return result

        return timed_fn

    return wrapper


def print_stats():
    print("Function timing stats:")
    for name, stat in function_stats.items():
        print(f"{name}: {stat['count']} calls, {stat['total_time']:.4f} seconds")


def log_function_stats(logger: WandbLogger, table_name: str = "function_timing"):
    columns = ["function", "call_count", "total_time [sec]", "average_time [sec]"]
    data = []

    for func_name, stat in function_stats.items():
        count = stat["count"]
        total_time = stat["total_time"]
        avg_time = total_time / count if count > 0 else 0.0
        data.append([func_name, count, total_time, avg_time])

    logger.log_table(key=table_name, columns=columns, data=data)


@torch.no_grad()
def debug_nonfinite_batch(
        tensors: Dict[str, Optional[Tensor]],
        *,
        batch_dim: int = 0,
        name: str = "loss_debug",
        logger=None,
        ids: Optional[Iterable] = None,  # e.g. event IDs parallel to batch
        max_items: int = 10,  # max bad items to print
        max_values: int = 256,  # truncate very large per-item dumps
        save_path: Optional[str] = None,  # save bad slices to .pt for offline
):
    """
    Find batch entries that contain NaN or Inf in ANY provided tensor, then dump
    all tensors for those indices.

    Args
    - tensors: dict like {
          "pred": Tensor[..., F?], "target": Tensor[..., F?],
          "mask": Tensor[...] or None, "weight": Tensor[...] or None, ...
      }
      All tensors are assumed to share the same batch size on `batch_dim`.
      Scalars or None are allowed.
    - batch_dim: which dimension is the batch axis in these tensors
    - name: a tag included in log lines
    - logger: Python logger. If None, prints to stdout
    - ids: optional iterable of identifiers for each batch item
    - max_items: truncate number of bad items reported
    - max_values: truncate long per-item value dumps
    - save_path: if set, saves a dict with all bad slices to this path via torch.save
    """

    def log(msg: str):
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    # helper: reduce a tensor to boolean per-batch mask of non-finite presence
    def per_batch_nonfinite(x: Tensor) -> Tensor:
        if x.ndim == 0:
            # scalar: mark all items bad if non-finite
            return torch.full((batch_size,), ~torch.isfinite(x), dtype=torch.bool, device=x.device)
        # move batch_dim to 0, then reduce over all remaining dims
        x_moved = x.transpose(0, batch_dim)
        bad = ~torch.isfinite(x_moved)
        # bad = x_moved > 0
        if bad.ndim == 1:
            return bad
        reduce_dims = tuple(range(1, bad.ndim))
        return bad.any(dim=reduce_dims)

    # figure out batch_size from the first present tensor with a batch axis
    batch_size = None
    for t in tensors.values():
        if isinstance(t, Tensor) and t.ndim > 0:
            batch_size = t.size(batch_dim)
            break
    if batch_size is None:
        log(f"[{name}] No tensors with a batch dimension were provided")
        return

    # accumulate bad mask across all tensors
    any_bad = torch.zeros(batch_size, dtype=torch.bool, device=next(
        (t.device for t in tensors.values() if isinstance(t, Tensor)), torch.device("cpu")
    ))
    per_tensor_bad = {}

    for key, t in tensors.items():
        if t is None:
            continue
        if not isinstance(t, Tensor):
            continue
        bad_mask = per_batch_nonfinite(t)
        any_bad |= bad_mask
        per_tensor_bad[key] = bad_mask

    bad_indices = torch.nonzero(any_bad, as_tuple=False).flatten()
    n_bad = bad_indices.numel()
    if n_bad == 0:
        # log(f"[{name}] No NaN/Inf found across {list(tensors.keys())}")
        return

    log(f"[{name}] Found {n_bad} bad batch item(s) across {list(tensors.keys())}")

    # save container for optional torch.save
    save_dump = {"indices": bad_indices.detach().cpu()}

    # small utility to slice along arbitrary batch_dim
    def slice_item(x: Tensor, idx: int) -> Tensor:
        if x.ndim == 0:
            return x
        return x.select(batch_dim, idx)

    # pretty dump with truncation
    def fmt_tensor_values(x: Tensor) -> str:
        flat = x.detach().cpu().reshape(-1)
        n = flat.numel()
        if n <= max_values:
            return flat.tolist().__repr__()
        head = flat[:max_values].tolist()
        return f"{head} ... (total {n} values)"

    # iterate over bad items
    for rank, idx in enumerate(bad_indices.tolist()[:max_items], start=1):
        hdr = f"[{name}] Bad item {rank}/{n_bad} at batch index {idx}"
        if ids is not None:
            try:
                ident = list(ids)[idx]
                hdr += f" (id={ident})"
            except Exception:
                pass
        log(hdr)

        for key, t in tensors.items():
            if t is None:
                log(f"  - {key}: None")
                continue
            if not isinstance(t, Tensor):
                log(f"  - {key}: non-tensor {type(t)}")
                continue
            item = slice_item(t, idx)
            shape = tuple(item.shape)
            n_nans = torch.isnan(item).sum().item() if torch.is_floating_point(item) else 0
            n_infs = torch.isinf(item).sum().item() if torch.is_floating_point(item) else 0
            log(f"  - {key}: shape={shape}, dtype={item.dtype}, nans={n_nans}, infs={n_infs}")
            log(f"    values: {fmt_tensor_values(item)}")
            save_dump.setdefault(key, []).append(item.detach().cpu())

    if n_bad > max_items:
        log(f"[{name}] ... truncated. Shown {max_items} of {n_bad} bad items")
