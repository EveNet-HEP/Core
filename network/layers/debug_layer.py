import torch
import torch.nn as nn

class PointCloudTransformer(nn.Module):
    def __init__(self, point_dim=3, embed_dim=64, num_heads=4, ff_dim=128, num_layers=1, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token

        self.input_proj = nn.Linear(point_dim, embed_dim)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: Tensor of shape [B, N, point_dim]
           B = batch size, N = number of points, point_dim = point feature dimension
        """
        x = self.input_proj(x)  # [B, N, embed_dim]

        if self.use_cls_token:
            B = x.size(0)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)          # [B, N+1, embed_dim]

        x = self.encoder(x)  # [B, N+1, embed_dim] or [B, N, embed_dim]

        if self.use_cls_token:
            return x[:, 0]  # return CLS token
        else:
            return x.mean(dim=1)  # mean pooling
