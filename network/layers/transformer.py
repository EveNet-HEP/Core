import torch.nn as nn
from torch import Tensor
import torch

from evenet.network.layers.utils import TalkingHeadAttention, StochasticDepth, LayerScale
from evenet.network.layers.linear_block import GRUGate, GRUBlock
from evenet.network.layers.activation import create_residual_connection

from typing import Optional

class TransformerBlockModule(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init,
                 drop_probability):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.talking_head = talking_head
        self.layer_scale_flag = layer_scale
        self.drop_probability = drop_probability

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)

        if talking_head:
            self.attn = TalkingHeadAttention(projection_dim, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Dropout(dropout),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        self.drop_path = StochasticDepth(drop_probability)

        if layer_scale:
            self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
            self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

    def forward(self, x, mask, attn_mask=None):
        # TransformerBlock input shapes: x: torch.Size([B, P, 128]), mask: torch.Size([B, P, 1])
        padding_mask = ~(mask.squeeze(2).bool()) if mask is not None else None  # [batch_size, num_objects]
        if self.talking_head:

            if attn_mask is None:
                int_matrix = None
            else:
                # Step 1: Create additive attention bias (float) with -inf where masked
                int_matrix = torch.zeros_like(attn_mask, dtype=torch.float32)  # (N, N)
                int_matrix[attn_mask] = float('-inf')  # or -1e9 if you prefer finite

                # Step 2: Broadcast to (B, num_heads, N, N)
                int_matrix = int_matrix.unsqueeze(0).unsqueeze(0).expand(x.shape[0], self.num_heads, attn_mask.shape[0], attn_mask.shape[1])
            updates, _ = self.attn(self.norm1(x), int_matrix=int_matrix, mask=mask) # TODO: check if attn_mask is correct
        else:
            if (attn_mask is not None) and (attn_mask.dim() == 3):
                batch_size, tgt_len, src_len = attn_mask.size()
                attn_mask = attn_mask.view(batch_size, 1, tgt_len, src_len)
                attn_mask = attn_mask.expand(batch_size, self.num_heads, tgt_len, src_len)
                attn_mask = attn_mask.reshape(batch_size * self.num_heads, tgt_len, src_len)

            updates, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                   key_padding_mask=padding_mask,
                                   attn_mask=attn_mask)

        if self.layer_scale_flag:
            # Input updates: torch.Size([B, P, 128]), mask: torch.Size([B, P])
            x2 = x + self.drop_path(self.layer_scale1(updates, mask))
            x3 = self.norm2(x2)
            x = x2 + self.drop_path(self.layer_scale2(self.mlp(x3), mask))
        else:
            x2 = x + self.drop_path(updates)
            x3 = self.norm2(x2)
            x = x2 + self.drop_path(self.mlp(x3))

        if mask is not None:
            x = x * mask

        return x


class GTrXL(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 hidden_dim_scale: float,
                 num_heads: int,
                 dropout: float):
        super(GTrXL, self).__init__()

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_gate = GRUGate(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = GRUBlock(input_dim=hidden_dim,
                                     hidden_dim_scale=hidden_dim_scale,
                                     output_dim=hidden_dim,
                                     normalization_type="LayerNorm",
                                     activation_type="gelu",
                                     dropout=dropout,
                                     skip_connection=True)

    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        """

        :param x: (batch_size, num_objects, hidden_dim)
        :param padding_mask: (batch_size, num_objects)
        :param sequence_mask: (batch_size, num_objects, hidden_dim)
        :return:
        """
        output = self.attention_norm(x)
        output, _ = self.attention(
            output, output, output,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        output = self.attention_gate(output, x)

        return self.feed_forward(x=output, sequence_mask=sequence_mask)


class GatedTransformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 num_heads: int,
                 transformer_activation: str,
                 transformer_dim_scale: float,
                 dropout: float,
                 drop_probability: float = 0.0,
                 skip_connection: bool = False):
        super(GatedTransformer, self).__init__()
        self.num_layers = num_layers

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.transformer_activation = transformer_activation
        self.transformer_dim_scale = transformer_dim_scale
        self.skip_connection = skip_connection
        self.drop_probability = drop_probability

        self.layers = nn.ModuleList([
            GTrXL(hidden_dim=self.hidden_dim,
                  hidden_dim_scale=self.transformer_dim_scale,
                  num_heads=self.num_heads,
                  dropout=self.dropout)
            for _ in range(num_layers)
        ])

        if self.skip_connection:
            self.norm = nn.LayerNorm(self.hidden_dim)
            self.drop_path = StochasticDepth(drop_probability)
            self.mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                nn.GELU(approximate='none'),
                nn.Dropout(dropout),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
    def forward(self, x: Tensor, padding_mask: Tensor, sequence_mask: Tensor) -> Tensor:
        """
        :param x: (batch_size, num_objects, hidden_dim)
        :param padding_mask: (batch_size, num_objects)
        :param sequence_mask: (batch_size, num_objects, hidden_dim)
        :return:
        """

        output = x
        for layer in self.layers:
            updates = layer(
                x=output,
                padding_mask=padding_mask,
                sequence_mask=sequence_mask
            )

            if self.skip_connection:
                x2 = output + self.drop_path(updates)
                x3 = self.norm(x2) * sequence_mask
                output = x2 + self.drop_path(self.mlp(x3))
                output = output * sequence_mask
            else:
                output = updates * sequence_mask

        return output


def create_transformer(
        transformer_type: str,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        transformer_activation: str,
        transformer_dim_scale: float,
        dropout: float,
        skip_connection: bool) -> nn.Module:
    """
    Create a transformer model with the specified options.

    :param options: Options for the transformer model.
    :param num_layers: Number of layers in the transformer.
    :return: Transformer model.
    """
    if transformer_type == "GatedTransformer":
        return GatedTransformer(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transformer_activation=transformer_activation,
            transformer_dim_scale=transformer_dim_scale,
            dropout=dropout,
            skip_connection=skip_connection
        )


class ClassifierTransformerBlockModule(nn.Module):
    def __init__(self,
                 input_dim: int,
                 projection_dim: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.bridge_class_token = create_residual_connection(
            skip_connection=True,
            input_dim=input_dim,
            output_dim=projection_dim
        )

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)
        self.norm3 = nn.LayerNorm(projection_dim)

        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Dropout(dropout),
            nn.Linear(2 * projection_dim, input_dim),
        )

    def forward(self, x, class_token, mask=None):
        """

        :param x: point_cloud (batch_size, num_objects, projection_dim)
        :param class_token: (batch_size, input_dim)
        :param mask: (batch_size, num_objects, 1)
        :return:
        """
        class_token = self.bridge_class_token(class_token)
        x1 = self.norm1(x)
        query = class_token.unsqueeze(1)  # Only use the class token as query

        padding_mask = ~(mask.squeeze(2).bool()) if mask is not None else None
        updates, _ = self.attn(query, x1, x1, key_padding_mask=padding_mask)  # [batch_size, 1, projection_dim]
        updates = self.norm2(updates)

        x2 = updates + query
        x3 = self.norm3(x2)
        cls_token = self.mlp(x3)

        return cls_token.squeeze(1)


class GeneratorTransformerBlockModule(nn.Module):
    def __init__(self,
                 projection_dim: int,
                 num_heads: int,
                 dropout: float,
                 layer_scale: bool,
                 layer_scale_init: float,
                 drop_probability: float):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_scale_flag = layer_scale
        self.drop_probability = drop_probability

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm3 = nn.LayerNorm(projection_dim)

        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        if layer_scale:
            self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
            self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

    def forward(self, x, cond_token, mask=None, attn_mask=None):
        """
        :param x: point_cloud (batch_size, num_objects, projection_dim)
        :param cond_token: (batch_size, 1, projection_dim)
        :param mask: (batch_size, num_objects, 1)
        """
        x1 = self.norm1(x)
        padding_mask = ~(mask.squeeze(2).bool()) if mask is not None else None

        if (attn_mask is not None) and (attn_mask.dim() == 3):
            batch_size, tgt_len, src_len = attn_mask.size()
            attn_mask = attn_mask.view(batch_size, 1, tgt_len, src_len)
            attn_mask = attn_mask.expand(batch_size, self.num_heads, tgt_len, src_len)
            attn_mask = attn_mask.reshape(batch_size * self.num_heads, tgt_len, src_len)

        updates, _ = self.attn(x1, x1, x1, key_padding_mask=padding_mask, attn_mask=attn_mask)

        if self.layer_scale_flag:
            updates = self.layer_scale1(updates, mask)
        x2 = updates + cond_token
        x3 = self.norm3(x2)
        x3 = self.mlp(x3)

        if self.layer_scale_flag:
            x3 = self.layer_scale2(x3, mask)
        cond_token = x2 + x3

        return x, cond_token

class SegmentationTransformerBlockModule(nn.Module):
    def __init__(self,
        projection_dim: int,
        num_heads: int,
        dropout: float,
    ):

        """
        Transformer block for segmentation tasks. Adopt from DETR architecture. https://github.com/facebookresearch/detr/blob/main/models/transformer.py#L127
        """
        super().__init__()

        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)
        self.norm3 = nn.LayerNorm(projection_dim)

        self.self_attn = nn.MultiheadAttention(self.projection_dim, self.num_heads, self.dropout, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout)

        self.multihead_attn = nn.MultiheadAttention(self.projection_dim, self.num_heads, self.dropout, batch_first=True)
        self.dropout2 = nn.Dropout(self.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(approximate='none'),
            nn.Dropout(self.dropout),
            nn.Linear(2 * projection_dim, projection_dim),
        )
        self.dropout3 = nn.Dropout(self.dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        Forward pass of the transformer block.
        """

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.mlp(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

