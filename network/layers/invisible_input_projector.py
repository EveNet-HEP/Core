from torch import Tensor, nn

from evenet.network.layers.utils import LayerScale


class InvisibleInputProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.pre_norm = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(approximate="none"),
            nn.Linear(output_dim, output_dim),
        )
        # Keep the residual branch active from the start without dropout.
        self.layer_scale = LayerScale(1.0, output_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.input_projection(x)
        residual = self.layer_scale(self.mlp(self.pre_norm(x)), mask=mask)
        x = x + residual
        if mask is not None:
            x = x * mask
        return x
