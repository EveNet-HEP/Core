import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 16, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up   = nn.Linear(bottleneck, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # 讓一開始 adapter ≈ identity（更穩）
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.drop(self.up(self.act(self.down(x))))
