from collections import OrderedDict
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices


def DICE_loss(inputs, targets, mask = None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid() # (B, *, P)
    numerator = 2 * (inputs * targets).sum(-1) # (B, *)
    denominator = inputs.sum(-1) + targets.sum(-1) # (B, *)
    loss = 1 - (numerator + 1) / (denominator + 1) # (B, *)
    if mask is not None:
        loss = loss * mask.float() # (B, *)
    return loss

def sigmoid_focal_loss(inputs, targets, mask = None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. (B, N, P)
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
                shape: (B, N, P)
        mask: (optional) A boolean tensor of the shape: (B,N)
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor (B, N) where B is the batch size and N is the number of elements in each example.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # (B, *, P)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss # (B, *, P)

    if mask is not None:
        loss = loss * (mask.float().unsqueeze(-1))  # (B, *, P)

    return loss

def hungarian_matching(
    predict_cls: Tensor,
    predict_mask: Tensor,
    target_cls: Tensor,
    target_mask: Tensor,
    class_weight: Tensor = None,
    dice_loss_weight: float = 1.0,
    cls_loss_weight: float = 1.0,
    mask_ce_loss_weight: float = 1.0,
    include_cls_cost: bool = True,
):
    """
    Perform Hungarian matching between predicted and target masks.

    Args:
        predict_cls (Tensor): (B, N, C) logits for class predictions
        predict_mask (Tensor): (B, N, P) logits for mask predictions
        target_cls (Tensor): (B, N, C) class labels
        target_mask (Tensor): (B, N, P) mask labels
        class_weight (Tensor, optional): (C,) class weights
        segmentation_mask (Tensor, optional): (B, N) mask to ignore certain pixels

    Returns:
        Tuple[Tensor, Tensor]: matched indices and matched masks
    """
    # Placeholder for the actual implementation of Hungarian matching
    # This function should return the matched indices and masks based on the inputs.

    num_queries = predict_cls.shape[1]
    predict_mask_expanded = predict_mask.unsqueeze(2).repeat(1, 1, num_queries, 1)  # (B, N_pred, N_tgt, P)
    target_mask_expanded = target_mask.unsqueeze(1).repeat(1, num_queries, 1, 1)  # (B, N_pred, N_tgt, P)

    predict_cls_expanded = predict_cls.softmax(-1).unsqueeze(2).repeat(1, 1, num_queries, 1)  # (B, N_pred, N_tgt, C)
    target_cls_expanded = target_cls.unsqueeze(1).repeat(1, num_queries, 1, 1) # (B, N_pred, N_tgt, C)
    non_null_cls_masking = (~((target_cls.argmax(-1) == 0).unsqueeze(1).repeat(1, num_queries, 1))).float() # (B, N_pred, N_tgt)

    # mask_cost = sum{c!=null}L_{mask}(pred, tgt)
    mask_cost = sigmoid_focal_loss(
        inputs = predict_mask_expanded,
        targets = target_mask_expanded,
    ) # (B, N_pred, N_tgt, P)
    mask_cost = mask_cost.sum(dim=-1)   # Sum over the last dimension (P) to get (B, N, N)
    mask_cost = mask_cost * non_null_cls_masking # (B, N_pred, N_tgt)

    dice_cost = DICE_loss(
        inputs = predict_mask_expanded,
        targets = target_mask_expanded,
    ) # (B,  N_pred, N_tgt)
    dice_cost = dice_cost * non_null_cls_masking # (B, N_pred, N_tgt)

    total_cost = mask_cost * mask_ce_loss_weight + dice_cost * dice_loss_weight # (B, N_pred, N_tgt)

    if include_cls_cost:
        class_cost = -(predict_cls_expanded * target_cls_expanded)
        total_cost = total_cost + class_cost.sum(dim=-1) * cls_loss_weight # (B, N_pred, N_tgt)

    src_indices, tgt_indices = assignment_to_indices(batch_linear_assignment(total_cost.detach())) # (B, N)

    return src_indices, tgt_indices

def loss(
        predict_cls,
        predict_mask,
        target_cls,
        target_mask,
        class_weight=None,
        point_cloud_mask = None,
        dice_loss_weight=1.0,
        cls_loss_weight=1.0,
        mask_loss_weight=1.0,
        event_weight = None,
        aux_outputs=None
    ):
    """

    Args:
        predict_cls (Tensor): (B, N, C) logits for class predictions
        predict_mask (Tensor): (B, N, P) logits for mask predictions
        target_cls (Tensor): (B, N, C) class labels
        target_mask (Tensor): (B, N, P) mask labels
        class_weight (Tensor, optional): (C,) class weights
        segmentation_mask (Tensor, optional): (B, N) mask to ignore certain pixels
        point_cloud_mask (Tensor, optional): (B, P, 1) mask to ignore certain pixels
        reduction (str): 'none', 'mean', or 'sum'

    Returns:
        Tensor: (B,) if reduction is 'none' (i.e. no reduction applied),)
    """
    bs, num_queries, num_patch = predict_mask.shape

    target_mask = target_mask.float()  # Ensure target_mask is float for loss calculations

    mask_loss, dice_loss, class_loss = calculate_loss(
        predict_cls=predict_cls,
        predict_mask=predict_mask,
        target_cls=target_cls,
        target_mask=target_mask,
        class_weight=class_weight,
        dice_loss_weight=dice_loss_weight,
        cls_loss_weight=cls_loss_weight,
        mask_ce_loss_weight=mask_loss_weight,
        point_cloud_mask=point_cloud_mask,
        event_weight = event_weight
    ) # (B,)

    mask_loss_aux = 0
    dice_loss_aux = 0
    cls_loss_aux = 0

    if aux_outputs is not None:
        for aux_predict in aux_outputs:
            mask_loss_, dice_loss_, class_loss_ = calculate_loss(
                predict_cls=aux_predict['pred_logits'],
                predict_mask=aux_predict['pred_masks'],
                target_cls=target_cls,
                target_mask=target_mask,
                dice_loss_weight=dice_loss_weight,
                cls_loss_weight=cls_loss_weight,
                mask_ce_loss_weight=mask_loss_weight,
                class_weight=class_weight,
                point_cloud_mask=point_cloud_mask,
                event_weight = event_weight,
            )
            mask_loss_aux += mask_loss_
            dice_loss_aux += dice_loss_
            cls_loss_aux += class_loss_
        # Average the losses from all auxiliary outputs
        num_aux = max(len(aux_outputs), 1)
        mask_loss_aux /= num_aux
        dice_loss_aux /= num_aux
        cls_loss_aux /= num_aux

    return mask_loss, dice_loss, class_loss, mask_loss_aux, dice_loss_aux, cls_loss_aux

def calculate_loss(
        predict_cls,
        predict_mask,
        target_cls,
        target_mask,
        class_weight=None,
        dice_loss_weight=1.0,
        cls_loss_weight=1.0,
        mask_ce_loss_weight=1.0,
        point_cloud_mask = None,
        event_weight = None
    ):
    """
    Calculate the loss given matched indices.
    """

    with torch.no_grad():
        # print("predict_cls", predict_cls.shape, "predict-mask", predict_mask.shape, target_cls.shape, target_mask.shape, class_weight.shape if class_weight is not None else None, segmentation_mask.shape if segmentation_mask is not None else None)
        pred_indices, tgt_indices = hungarian_matching(
            predict_cls=predict_cls,
            predict_mask=predict_mask,
            target_cls=target_cls,
            target_mask=target_mask,
            class_weight=class_weight,
            dice_loss_weight=dice_loss_weight,
            cls_loss_weight=cls_loss_weight,
            mask_ce_loss_weight=mask_ce_loss_weight,
        ) # (B, N) indices of matched predictions and targets

    # print("pred_indices", pred_indices.shape, "tgt_indices", tgt_indices.shape)
    B, N_match = pred_indices.shape
    batch_idx = torch.arange(B, device=pred_indices.device).unsqueeze(-1)  # (B, 1)

    # Gather the predicted and target masks based on the matched indices
    predict_mask_best = predict_mask[batch_idx, pred_indices]  # (B, N, P)
    target_mask_best = target_mask[batch_idx, tgt_indices]  # (B, N, P)
    predict_class_best = predict_cls[batch_idx, pred_indices]  # (B, N, C)
    target_class_best = target_cls[batch_idx, tgt_indices]  # (B, N, C)
    point_cloud_mask = point_cloud_mask.squeeze(-1) if point_cloud_mask is not None else None


    non_null_cls_masking = (~((target_class_best.argmax(-1) == 0))).float() # (B, N)

    # print("predict_mask_best", predict_mask_best.shape, "target_mask_best", target_mask_best.shape, "predict_class_best", predict_class_best.shape, "target_class_best", target_class_best.shape, "segmentation_mask_best", segmentation_mask_best.shape)

    mask_loss = sigmoid_focal_loss(
        inputs = predict_mask_best,
        targets = target_mask_best,
    ) # (B, N, P)

    if point_cloud_mask is not None:
        # print("mask_loss", mask_loss.shape, point_cloud_mask.shape)
        mask_loss = (mask_loss * point_cloud_mask.unsqueeze(1).float()).sum(-1) / (point_cloud_mask.float().sum(-1).unsqueeze(1) + 1e-6) # (B, N)
    else:
        mask_loss = mask_loss.mean(-1) # (B, N)

    # if segmentation_mask is not None:
    #     mask_loss = mask_loss.sum(-1) / (segmentation_mask.sum(-1) + 1e-6) # (B, )
    # else:
    #     mask_loss = mask_loss.mean(-1)

    dice_loss = DICE_loss(
        inputs = predict_mask_best,
        targets = target_mask_best,
    ) # (B, N)

    # print("predict_mask", predict_mask_best[0,1], "target_mask", target_mask_best[0,1])


    # if segmentation_mask is not None:
    #     dice_loss = dice_loss.sum(-1) / (segmentation_mask.sum(-1) + 1e-6)
    # else:
    #     dice_loss = dice_loss.mean(-1) # (B,)

    class_loss = F.cross_entropy(
        input=predict_class_best.permute(0,2,1), # (B, C, N)
        target= target_class_best.argmax(dim=-1), # TODO: check
        weight=class_weight,
        reduction="none",
        ignore_index=-1,
    ) # (B,)

    target_class_label_best = target_class_best.argmax(dim=-1)  # (B, N)
    final_event_weight = event_weight if event_weight is not None else torch.ones(B, device=predict_cls.device)
    final_event_weight_w_cls_weight = final_event_weight.unsqueeze(-1) * class_weight[target_class_label_best] # (B, N)

    # focal/dice loss should handle class imbalance, so we do not weight them by class weight
    mask_loss = (mask_loss * non_null_cls_masking).sum() / (non_null_cls_masking.sum().clamp(1e-6))
    dice_loss = (dice_loss * non_null_cls_masking).sum() / (non_null_cls_masking.sum().clamp(1e-6))

    # only class related loss is weighted by event weight
    class_loss = (class_loss * final_event_weight.unsqueeze(-1)).sum() / final_event_weight_w_cls_weight.sum().clamp(1e-6)

    return mask_loss, dice_loss, class_loss
