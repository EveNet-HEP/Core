import torch
from torch.nn import functional as F


def masked_log_softmax(
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int = -1
) -> torch.Tensor:
    """
    Gradient-safe masked log-softmax.
    Sets masked values to -inf and ensures no row is all masked to avoid NaNs.
    """

    if mask is not None:
        fill_value = float("-inf")

        # Mask invalid entries
        vector = vector.masked_fill(~mask, fill_value)

        # Prevent rows from becoming all -inf (causes NaN in log_softmax)
        all_masked = (mask.sum(dim=dim, keepdim=True) == 0)
        vector = vector.masked_fill(all_masked, 0.0)

    return F.log_softmax(vector, dim=dim)


def masked_softmax(
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int = -1,
        memory_efficient: bool = False,
) -> torch.Tensor:
    return torch.exp(masked_log_softmax(vector, mask, dim))
