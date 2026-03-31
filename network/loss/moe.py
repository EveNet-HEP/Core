"""MoE load-balancing losses.

The auxiliary load-balancing loss (``L_aux``) and router z-loss (``cz_Lz``) are
pre-computed inside the model's forward pass (see ``MoE.forward`` in
``network/layers/transformer.py``) and aggregated by ``EveNetModel.shared_step``.

This module provides the canonical helper for training engines to collect those
losses from the model output dict and add them to the overall loss with weight 1.0.
"""

from typing import Optional, Tuple

from torch import Tensor


def loss(
    l_aux: Optional[Tensor],
    cz_lz: Optional[Tensor],
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Return the pre-computed MoE load-balancing losses unchanged.

    Both values are already scalar tensors produced during the model forward
    pass:

    * ``l_aux``  – auxiliary load-balancing loss (Switch-Transformer style):
                   ``alpha * num_experts * Σ(fi * pi)``
    * ``cz_lz``  – router z-loss that penalises large logit magnitudes:
                   ``c_z * mean(logsumexp(router_logits, dim=-1)²)``

    They should be summed into the total loss with weight 1.0 (no task-weight
    scaling) by the calling engine.

    Args:
        l_aux:  Scalar auxiliary loss tensor, or ``None`` when MoE is disabled.
        cz_lz:  Scalar z-loss tensor, or ``None`` when MoE is disabled.

    Returns:
        ``(l_aux, cz_lz)`` – the same tensors passed in, unchanged.
    """
    return l_aux, cz_lz
