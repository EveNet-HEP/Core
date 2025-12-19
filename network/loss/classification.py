import torch
import torch.nn.functional as F

def loss(predict, target, class_weight=None, event_weight=None):
    """
    Cross-entropy loss with optional class weights.
    Returns per-sample loss for external weighting.

    Args:
        predict (Tensor): (N, C) logits
        target (Tensor): (N,) class indices
        class_weight (Tensor, optional): (C,) class weights

    Returns:
        Tensor: (N,) if reduction='none', else scalar
    """
    class_loss = F.cross_entropy(
        input=predict,
        target=target,
        weight=class_weight,
        reduction='none',
        ignore_index=-1,
    )

    if event_weight is not None:
        class_loss = class_loss * event_weight

        # Mask out ignored targets
    mask_cls = target != -1
    valid_targets = target[mask_cls]

    # The denominator: sum of weights of valid targets
    if valid_targets.numel() > 0:
        overall_class_weight_sum = (class_weight[valid_targets]).sum() if event_weight is None else (
                    class_weight[valid_targets] * event_weight[mask_cls]).sum()

    return class_loss.sum()/overall_class_weight_sum if overall_class_weight_sum > 0 else class_loss.mean()