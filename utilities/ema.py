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

    @torch.no_grad()
    def update(self, model: nn.Module, decay_: float = None):
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        decay = self.decay if decay_ is None else decay_

        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def copy_to(self, model: nn.Module):
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict, device=None):
        self.shadow = {k: v.clone().to(device) for k, v in state_dict.items()}
