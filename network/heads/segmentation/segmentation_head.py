import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn.modules.module import T

from evenet.network.layers.utils import get_activation_fn
from evenet.network.layers.transformer import ClassifierTransformerBlockModule

from typing import Optional

from evenet.network.layers.transformer import SegmentationTransformerBlockModule

"""
Segmentation Transformer Head modified from Mask2Former
https://github.com/facebookresearch/Mask2Former/
"""


class SelfAttentionLayer(nn.Module):
    def __init__(
            self,
            projection_dim,
            num_heads,
            dropout=0.0,
            activation="relu",
            normalize_before=False):
        super(SelfAttentionLayer, self).__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(selfself, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self, tgt,
            tgt_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, pos)


class CrossAttentionLayer(nn.Module):
    def __init__(
            self, projection_dim,
            num_heads, dropout=0.0,
            activation="relu", normalize_before=False):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout, batch_first=True)
        self.normalize_before = normalize_before

        self.norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self, tgt, memory,
            memory_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None):

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
            self, tgt, memory,
            memory_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)



class FFNLayer(nn.Module):
    def __init__(self, projection_dim, ffn_dim, dropout=0.0, activation="relu", normalize_before=False):
        super(FFNLayer, self).__init__()
        self.projection_dim = projection_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.normalize_before = normalize_before

        self.linear1 = nn.Linear(projection_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, projection_dim)
        self.norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




class SegmentationHead(nn.Module):
    def __init__(
            self,
            projection_dim: int,
            mask_dim: int,
            num_heads: int,
            dropout: float,
            num_layers: int,
            num_mask_mlp_layers: int = 1,
            num_class: int = 1,  # Binary classification for mask prediction
            num_queries: int = 5,
            return_intermediate: bool = False,
            norm_before: bool = True,
            encode_event_token: bool = False,
    ):
        super(SegmentationHead, self).__init__()

        self.return_intermediate = return_intermediate
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_class = num_class
        self.num_layers = num_layers
        self.norm_before = norm_before

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    projection_dim=projection_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    normalize_before=norm_before,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    projection_dim=projection_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    normalize_before=norm_before,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    projection_dim=projection_dim,
                    ffn_dim=projection_dim * 2,
                    dropout=dropout,
                    normalize_before=norm_before,
                )
            )

        self.decoder_norm = nn.LayerNorm(projection_dim)

        # learnable query features
        self.query_feat = nn.Embedding(num_queries, projection_dim)
        self.query_embed = nn.Embedding(num_queries, projection_dim)

        self.class_embed = nn.Linear(projection_dim, num_class + 1)  # Binary classification for mask prediction
        self.mask_embed = MLP(projection_dim, projection_dim, mask_dim, num_mask_mlp_layers)

        # Encode event token
        self.encode_event_token = encode_event_token
        if encode_event_token:
            self.class_transformer = ClassifierTransformerBlockModule(
                input_dim=projection_dim,
                projection_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout
            )

    def forward(self, memory, memory_mask,
                pos_embed: Optional[Tensor] = None,
                event_token: Optional[Tensor] = None,

                **kwargs
                ):
        """
        Args:
            memory: (batch_size, num_patches, projection_dim)
            memory_mask: (batch_size, num_patches, 1)
            **kwargs: additional arguments for the decoder
        """

        batch_size, num_patches, projection_dim = memory.shape
        padding_mask = ~(
            memory_mask.squeeze(-1).bool()) if memory_mask is not None else None  # (batch_size, num_patches)

        predictions_class = []
        predictions_mask = []

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, num_queries, projection_dim)
        output = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, num_queries, projection_dim)

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output=output,
            mask_features=memory,
            padding_mask=padding_mask
        )

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # if all elements are masked, then do not apply the mask
            output = self.transformer_cross_attention_layers[i](
                tgt=output,
                memory=memory,
                memory_key_padding_mask=None,
                pos=pos_embed,
                query_pos=query_embed,
                memory_mask=attn_mask
            )
            output = self.transformer_self_attention_layers[i](
                tgt=output,
                tgt_key_padding_mask=None,
                pos=query_embed
            )

            output = self.transformer_ffn_layers[i](output)
            output_class, output_mask, attn_mask = self.forward_prediction_heads(
                output=output,
                mask_features=memory,
                padding_mask=padding_mask
            )
            predictions_class.append(output_class)
            predictions_mask.append(output_mask)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask
            ),
        }
        if self.encode_event_token and event_token is not None:
            # event_token: (batch_size, token_dim)
            out['event-token'] = self.class_transformer(output, event_token)

        return out

    def forward_prediction_heads(self, output, mask_features, padding_mask=None):
        """
        Args:
            output: (batch_size, num_queries, projection_dim)
            mask_features: (batch_size, num_jets, mask_dim)
        """
        decoder_output = self.decoder_norm(output) #(batch_size, num_queries, projection_dim)
        outputs_class = self.class_embed(decoder_output) # (batch_size, num_queries, num_class + 1)
        mask_embed = self.mask_embed(decoder_output) # (batch_size, num_queries, mask_dim)
        outputs_mask = torch.einsum("bqh, bnh -> bqn", mask_embed, mask_features) # (batch_size, num_queries, num_jets)

        outputs_mask = outputs_mask + padding_mask.unsqueeze(1).repeat(1, self.num_queries, 1) * -1e9 if padding_mask is not None else outputs_mask

        attn_mask = (outputs_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,1) < 0.5).bool() # (batch_size*num_heads, num_queries, num_jets)
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    def _set_aux_loss(self, outputs_class, outputs_mask):
        # this is a workaround to make it work with torchscript

        if self.return_intermediate:
            return [{'pred_logits': a, 'pred_masks': b}
                    for a, b in zip(outputs_class[:-1], outputs_mask[:-1])]
        else:
            return None

