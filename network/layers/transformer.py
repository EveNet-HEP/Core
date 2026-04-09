import logging

import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F

from evenet.network.layers.utils import TalkingHeadAttention, StochasticDepth, LayerScale
from evenet.network.layers.linear_block import GRUGate, GRUBlock
from evenet.network.layers.activation import create_residual_connection

from typing import Optional

_moe_logger = logging.getLogger(__name__)


class Gate(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        select_top_k: int,
        use_router_noise: bool
    ) -> None:
        super().__init__()
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        self.noise_router = nn.Linear(embed_dim, num_experts, bias=False) if use_router_noise else None
        self.select_top_k = select_top_k

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        router_logits = self.router(x)
        noisy_router_logits = router_logits

        # noisy top-k routing during training to keep exploration healthy
        if self.training and self.noise_router is not None:
            noise_std = F.softplus(self.noise_router(x))
            noisy_router_logits = noisy_router_logits + torch.randn_like(noisy_router_logits) * noise_std

        # get top-k expert scores per object
        topk_logits, topk_indices = torch.topk(noisy_router_logits, self.select_top_k, dim=-1)
        # probability distribution over selected experts only
        topk_weights = torch.softmax(topk_logits, dim=-1)

        # dense gate weights over all experts from clean logits; used for losses/stats
        dense_gate_weights = torch.softmax(router_logits, dim=-1)

        return router_logits, dense_gate_weights, topk_weights, topk_indices


class Expert(nn.Module):
    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            # GELU activation maintained from original PET FFN
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class MoE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
        base_num_experts: int,
        base_select_top_k: int,
        num_shared_experts: int,
        expert_segmentation_factor: int,
        scale_expert_dim: bool,
        alpha: float,
        c_z: float,
        use_router_noise: bool,
        dropout: float
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.base_num_experts = base_num_experts
        self.base_select_top_k = base_select_top_k
        self.expert_segmentation_factor = expert_segmentation_factor
        self.num_shared_experts = num_shared_experts
        self.alpha = alpha
        self.c_z = c_z

        total_experts = self.base_num_experts * self.expert_segmentation_factor
        # num_experts is the total budget - routed experts fill the remainder after reserving shared slots
        self.num_experts = total_experts - num_shared_experts
        self.select_top_k = self.base_select_top_k * self.expert_segmentation_factor
        # when scale_expert_dim is True divide each expert's hidden dim by select_top_k to keep per-token compute constant vs. a vanilla FFN
        # note: even shared experts are being impacted by segmentation scaling of k
        self.expert_hidden_dim = int(feedforward_dim / (self.select_top_k + self.num_shared_experts)) if scale_expert_dim else feedforward_dim

        self.gate = Gate(embed_dim, self.num_experts, self.select_top_k, use_router_noise=use_router_noise)
        self.routed_experts = nn.ModuleList([
                Expert(embed_dim, self.expert_hidden_dim, dropout)
                for _ in range(self.num_experts)
        ])
        self.shared_experts = nn.ModuleList([
                Expert(embed_dim, self.expert_hidden_dim, dropout)
                for _ in range(self.num_shared_experts)
        ])
        self._forward_logged = False
        self.register_buffer('expert_dispatch_counts', torch.zeros(self.num_experts, dtype=torch.long))

    def reset_expert_dispatch_counts(self) -> None:
        """Reset the accumulated eval-time dispatch counts to zero."""
        self.expert_dispatch_counts.zero_()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = x.shape
        # collapse batch/object axes to a 2D tensor so that each row corresponds to a single object to route to experts
        x = x.reshape(-1, x.shape[-1])
        num_objects = x.shape[0]

        router_logits, dense_gate_weights, topk_weights, topk_indices = self.gate(x)

        if not self._forward_logged:
            expert_w_in  = self.routed_experts[0].ffn[0].weight.shape  # (hidden, embed)
            expert_w_out = self.routed_experts[0].ffn[3].weight.shape  # (embed, hidden)
            gate_w       = self.gate.router.weight.shape                # (num_experts, embed)

            shape_ok = (
                x.shape[-1]             == self.embed_dim
                and router_logits.shape == (num_objects, self.num_experts)
                and topk_indices.shape  == (num_objects, self.select_top_k)
                and expert_w_in[0]      == self.expert_hidden_dim
                and expert_w_in[1]      == self.embed_dim
                and gate_w[0]           == self.num_experts
            )
            level = _moe_logger.info if shape_ok else _moe_logger.error

            level(
                f"[MoE] First forward — input: {tuple(original_shape)}  "
                f"(flattened tokens: {num_objects})"
            )
            level(
                f"[MoE]   gate    weight : {tuple(gate_w)}  "
                f"→ router_logits: {tuple(router_logits.shape)}  "
                f"topk_indices: {tuple(topk_indices.shape)}"
            )
            level(
                f"[MoE]   expert  ffn[0] : {tuple(expert_w_in)}  "
                f"ffn[3]: {tuple(expert_w_out)}  "
                f"(expected hidden={self.expert_hidden_dim}, embed={self.embed_dim})"
            )
            if self.num_shared_experts > 0:
                sh_w_in  = self.shared_experts[0].ffn[0].weight.shape
                sh_w_out = self.shared_experts[0].ffn[3].weight.shape
                level(
                    f"[MoE]   shared  ffn[0] : {tuple(sh_w_in)}  "
                    f"ffn[3]: {tuple(sh_w_out)}"
                )
            if not shape_ok:
                _moe_logger.error("[MoE]   ❌ Shape mismatch detected — check config vs checkpoint.")
            else:
                _moe_logger.info("[MoE]   ✅ All shapes consistent.")
            self._forward_logged = True

        routed_output = torch.zeros((num_objects, self.embed_dim), dtype=x.dtype, device=x.device)
        objects_per_expert = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)

        if num_objects > 0:
            # get flat list of each object id repeated for each of its top-k experts - e.g., [0, 0, 1, 1, 2, 2, ...]
            object_indices = torch.arange(num_objects, device=x.device).unsqueeze(1).expand(-1, self.select_top_k).reshape(-1)
            # get flat list of which expert each object is assigned to - of size [num_objects * top_k]
            expert_indices = topk_indices.reshape(-1)
            # get flat list of corresponding expert weights for each object - of size [num_objects * top_k]
            expert_weights = topk_weights.reshape(-1)

            # sort by expert index so that all objects for each expert are grouped together
            order = torch.argsort(expert_indices)
            # update to be in sorted order by expert index
            object_indices = object_indices[order]
            expert_indices = expert_indices[order]
            expert_weights = expert_weights[order]
            # count how many objects are assigned to each expert to know how to split the input tensor for each expert's forward pass
            objects_per_expert = torch.bincount(expert_indices, minlength=self.num_experts)

            if not self.training:
                self.expert_dispatch_counts.add_(objects_per_expert.to(self.expert_dispatch_counts.device))

            cursor = 0
            # iterate through each expert's assigned objects in order of expert index
            for expert_id, count in enumerate(objects_per_expert.tolist()):
                if count == 0:
                    continue
                end = cursor + count

                # create minibatch if all objects assigned to the current expert
                current_object_indices = object_indices[cursor:end]
                current_inputs = x.index_select(0, current_object_indices)

                # forward pass through the current expert with created minibatch
                current_outputs = self.routed_experts[expert_id](current_inputs)
                # get the corresponding expert weights for the current expert's assigned objects
                current_weights = expert_weights[cursor:end].unsqueeze(-1)

                # weight the expert outputs by the corresponding expert weights for each object,
                # then add to the correct rows of the final output tensor using the object indices
                routed_output.index_add_(0, current_object_indices, current_outputs * current_weights)

                cursor = end

        # as shared experts are not part of the routing decisions,
        # run all objects through all shared experts and add to the final output
        if self.num_shared_experts > 0:
            shared_output = torch.zeros_like(routed_output)
            for shared_expert in self.shared_experts:
                shared_output = shared_output + shared_expert(x)
            final_output = routed_output + shared_output
        else:
            final_output = routed_output

        # fi - proportion of objects assigned to each expert
        denom = max(num_objects * self.select_top_k, 1)
        dispatch_fraction = objects_per_expert.to(dtype=dense_gate_weights.dtype) / denom
        # pi - average probability of each expert being selected across all objects
        mean_router_prob = dense_gate_weights.mean(dim=0) if num_objects > 0 else torch.zeros_like(dispatch_fraction)
        # l_aux is the sum across experts of fi * pi, scaled by alpha and num_experts
        l_aux = self.alpha * self.num_experts * torch.sum(dispatch_fraction * mean_router_prob)

        if num_objects > 0:
            # use clean router logits without noise (pre-softmax)
            cz_lz = self.c_z * torch.mean(torch.logsumexp(router_logits, dim=-1).pow(2))
        else:
            cz_lz = torch.zeros((), dtype=x.dtype, device=x.device)

        # convert back to original batch/object shape, with the MoE output in the last dimension
        final_output = final_output.view(original_shape[0], original_shape[1], self.embed_dim)

        return final_output, l_aux, cz_lz


class TransformerBlockModule(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout, talking_head, layer_scale, layer_scale_init,
                 drop_probability, use_moe: bool = False, moe_base_num_experts: int = 4,
                 moe_base_select_top_k: int = 2, moe_num_shared_experts: int = 0,
                 moe_expert_segmentation_factor: int = 1, moe_scale_expert_dim: bool = False,
                 moe_alpha: float = 0.01, moe_cz: float = 0.0, moe_use_router_noise: bool = False):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.talking_head = talking_head
        self.layer_scale_flag = layer_scale
        self.drop_probability = drop_probability
        self.use_moe = use_moe

        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)

        if talking_head:
            self.attn = TalkingHeadAttention(projection_dim, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout, batch_first=True)

        if not self.use_moe:
            self.mlp = nn.Sequential(
                nn.Linear(projection_dim, 2 * projection_dim),
                nn.GELU(approximate="none"),
                nn.Dropout(dropout),
                nn.Linear(2 * projection_dim, projection_dim),
            )
        else:
            self.mlp = MoE(
                embed_dim=projection_dim,
                feedforward_dim=2 * projection_dim,
                base_num_experts=moe_base_num_experts,
                base_select_top_k=moe_base_select_top_k,
                num_shared_experts=moe_num_shared_experts,
                expert_segmentation_factor=moe_expert_segmentation_factor,
                scale_expert_dim=moe_scale_expert_dim,
                alpha=moe_alpha,
                c_z=moe_cz,
                use_router_noise=moe_use_router_noise,
                dropout=dropout,
            )

        self.moe_l_aux = torch.tensor(0.0)
        self.moe_cz_lz = torch.tensor(0.0)
        self.drop_path = StochasticDepth(drop_probability)

        if layer_scale:
            self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
            self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

    def forward(self, x, mask, attn_mask=None):
        # TransformerBlock input shapes: x: torch.Size([B, P, 128]), mask: torch.Size([B, P, 1])
        self.moe_l_aux = x.new_zeros(())
        self.moe_cz_lz = x.new_zeros(())

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
            if self.use_moe:
                x4, self.moe_l_aux, self.moe_cz_lz = self.mlp(x3)
            else:
                x4 = self.mlp(x3)
            x = x2 + self.drop_path(self.layer_scale2(x4, mask))
        else:
            x2 = x + self.drop_path(updates)
            x3 = self.norm2(x2)
            if self.use_moe:
                x4, self.moe_l_aux, self.moe_cz_lz = self.mlp(x3)
            else:
                x4 = self.mlp(x3)
            x = x2 + self.drop_path(x4)

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


def log_moe_expert_distribution(
    model: nn.Module,
    reset: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log the expert dispatch distribution for every MoE layer in *model*.

    Call this after running eval batches to see how evenly inputs were routed.
    Counts are accumulated across all eval forward passes since the last reset.

    Args:
        model:  Any nn.Module that may contain MoE sub-modules.
        reset:  If True (default), zero the counts after logging so the next
                eval epoch starts fresh.
        logger: Logger to write to.  Defaults to this module's logger.

    Example usage in an eval loop::

        model.eval()
        for batch in eval_loader:
            with torch.no_grad():
                model(batch)
        log_moe_expert_distribution(model)   # prints distribution, resets counts
    """
    log = (logger or _moe_logger).info

    moe_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, MoE)]
    if not moe_layers:
        _moe_logger.warning("log_moe_expert_distribution: no MoE layers found in model.")
        return

    for name, moe in moe_layers:
        counts = moe.expert_dispatch_counts.cpu()
        total = counts.sum().item()

        if total == 0:
            log(f"[MoE dist] {name}: no eval data recorded (counts are all zero).")
            if reset:
                moe.reset_expert_dispatch_counts()
            continue

        uniform = total / moe.num_experts
        lines = [
            f"[MoE dist] {name}  "
            f"(total dispatches={total:,}, ideal per expert={uniform:,.1f})"
        ]
        for i, c in enumerate(counts.tolist()):
            pct = 100.0 * c / total
            bar = "█" * int(pct / 2)  # each block ≈ 2 %
            lines.append(f"  expert {i:3d}: {c:9,d}  ({pct:5.1f}%)  {bar}")
        log("\n".join(lines))

        if reset:
            moe.reset_expert_dispatch_counts()

