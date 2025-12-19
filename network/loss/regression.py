import torch
import torch.nn.functional as F


def loss(predict: torch.Tensor,
         target: torch.Tensor,
         mask: torch.Tensor = None,
         event_weight: torch.Tensor=None,
         beta: float = 1.0) -> torch.Tensor:
    """
    Compute Smooth L1 (Huber) regression loss per event.

    Args:
        predict (torch.Tensor): Predicted values, shape (N, D)
        target (torch.Tensor): Ground truth values, shape (N, D)
        mask (torch.Tensor, optional): Optional mask to ignore certain dimensions, shape (N, D). If None, all dimensions are considered.
        beta (float): Transition point from L1 to L2 loss.
        event_weight(torch.Tensor, optional): Weights for each event, shape (N,). If None, all events are equally weighted.

    Returns:
        torch.Tensor: Per-event loss, shape (N,)
    """
    # Compute element-wise Smooth L1 loss
    loss_per_dim = F.smooth_l1_loss(predict, target, reduction='none', beta=beta)  # shape: (N, D)
    if event_weight is not None:
        # Apply event weights if provided
        loss_per_dim = loss_per_dim * event_weight.unsqueeze(1)

    if mask is not None:
        # Apply the mask
        masked_loss = loss_per_dim * mask

        # Avoid division by zero in case some events have no valid entries
        valid_counts = (mask*event_weight).sum(dim=1).clamp(min=1e-6)
        loss_per_event = masked_loss.sum(dim=1) / valid_counts
    else:
        # No mask, just mean across features
        if event_weight is not None:
            # If event weights are provided, apply them
            loss_per_dim = loss_per_dim.sum(dim=1)/event_weight.sum(dim=0).clamp(min=1e-6)
        else:
            loss_per_event = loss_per_dim.mean(dim=1)

    return loss_per_event  # shape: (N,)
