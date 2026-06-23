import copy
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.current_epoch = 0

        self.shadow = {}
        self.model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._bind_tensors()

    def _bind_tensors(self):
        """Cache aligned tensor lists required by fused foreach operations."""
        parameters = dict(self.model.named_parameters())
        self._names = tuple(
            name for name in self.shadow if name in parameters
        )
        self._shadow_tensors = [self.shadow[name] for name in self._names]
        self._model_tensors = [parameters[name].detach() for name in self._names]

    @torch.no_grad()
    def update(self, model: nn.Module, decay_: float = None):
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if model is not self.model:
            self.model = model
            self._bind_tensors()
        decay = float(self.decay if decay_ is None else decay_)

        # One foreach launch per operation replaces a Python loop and one or
        # more CUDA launches per parameter. The element-wise EMA equation and
        # update ordering are unchanged.
        if self._shadow_tensors:
            torch._foreach_mul_(self._shadow_tensors, decay)
            torch._foreach_add_(
                self._shadow_tensors,
                self._model_tensors,
                alpha=1.0 - decay,
            )

    def copy_to(self, model: nn.Module):
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict, device=None):
        self.shadow = {k: v.clone().to(device) for k, v in state_dict.items()}
        self._bind_tensors()
