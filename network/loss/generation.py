import torch
from torch import Tensor
from typing import Optional


def loss(
        predict: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        feature_dim: Optional[int] = None,
        event_weight: Optional[Tensor] = None
):
    if event_weight is not None:
        weight = event_weight.view(-1, *([1] * (predict.ndim - 1))).float()
        if mask is not None:
            den = torch.sum(mask.float() * weight) * feature_dim
            if den == 0:
                return torch.tensor(0.0, device=predict.device, dtype=predict.dtype)
            return torch.sum(((predict - target) ** 2) * mask.float() * weight) / den
        else:
            return torch.sum(((predict - target) ** 2) * weight) / torch.sum(weight)
    else:
        if mask is not None:
            den = torch.sum(mask.float()) * feature_dim
            if den == 0:
                return torch.tensor(0.0, device=predict.device, dtype=predict.dtype)
            return torch.sum(((predict - target) ** 2) * mask.float()) / den
        else:
            return torch.mean((predict - target) ** 2)
