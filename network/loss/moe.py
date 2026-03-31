from torch import Tensor
from typing import Optional, Tuple, Dict, Any


def loss(model_output: Dict[str, Any]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Extract MoE auxiliary losses from model output.

    The auxiliary load-balancing loss (l_aux) and z-loss (cz_lz) are computed
    during the model forward pass inside each MoE transformer block and
    accumulated onto the model output dict under the keys "L_aux" and "cz_Lz".
    Callers should add both directly to their total loss without additional
    scaling; the weights are already baked in via the alpha and c_z hyperparameters
    configured on each MoE layer.

    Args:
        model_output: dict returned by EveNetModel.forward()

    Returns:
        l_aux:  auxiliary load-balancing loss, or None when MoE is disabled
        cz_lz:  z-loss regularisation term, or None when MoE is disabled
    """
    l_aux = model_output.get("L_aux", None)
    cz_lz = model_output.get("cz_Lz", None)
    return l_aux, cz_lz
