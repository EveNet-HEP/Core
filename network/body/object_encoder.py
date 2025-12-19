from typing import Tuple

import torch
from torch import Tensor, nn

from evenet.network.layers.linear_block import create_linear_block
from evenet.network.layers.transformer import create_transformer
from evenet.network.body.embedding import CombinedEmbedding

from evenet.network.layers.activation import create_residual_connection

class ObjectEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            position_embedding_dim: int,
            num_heads: int,
            transformer_dim_scale: float,
            num_linear_layers: int,
            num_encoder_layers: int,
            dropout: float,
            skip_connection: bool = False,
            encoder_skip_connection: bool = False,
            conditioned: bool = False
    ):
        super(ObjectEncoder, self).__init__()

        self.point_cloud_bridge = create_residual_connection(
            skip_connection=True,
            input_dim=input_dim,
            output_dim=hidden_dim,
        )
        self.global_cond_bridge = create_residual_connection(
            skip_connection=True,
            input_dim=input_dim,
            output_dim=hidden_dim,
        )

        self.particle_vector = nn.Parameter(torch.randn(1, 1, hidden_dim))
        if conditioned:
            self.dense_particle = nn.Linear(2 * hidden_dim, hidden_dim)

        self.combined_embedding = CombinedEmbedding(
            hidden_dim=hidden_dim,
            position_embedding_dim=position_embedding_dim
        )

        self.encoder = create_transformer(
            "GatedTransformer",
            num_layers=num_encoder_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transformer_activation="gelu",
            transformer_dim_scale=transformer_dim_scale,
            dropout=dropout,
            skip_connection=encoder_skip_connection,
        )
        self.embedding = nn.ModuleList([
            create_linear_block(
                linear_block_type="GRU",
                input_dim=hidden_dim,
                hidden_dim_scale=2.0,
                output_dim=hidden_dim,
                normalization_type="LayerNorm",
                activation_type="gelu",
                dropout=dropout,
                skip_connection=skip_connection,
            ) for _ in range(num_linear_layers)])

        self.output_bridge = create_residual_connection(
            skip_connection=True,
            input_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self,
                encoded_vectors: Tensor,
                mask: Tensor,
                condition_vectors: Tensor,
                condition_mask: Tensor,
                cond_vector: Tensor = None) -> Tuple[Tensor, Tensor]:
        """

        :param encoded_vectors: (batch_size, num_vectors, hidden_dim)
        :param condition_vectors: (batch_size, num_conditions, hidden_dim)
        :param mask: (batch_size, num_vectors, 1)
        :param condition_mask: (batch_size, num_conditions, 1)
        :param cond_vector: (batch_size, hidden_dim)
        :return:
        """

        encoded_vectors = self.point_cloud_bridge(encoded_vectors) * mask
        condition_vectors = self.global_cond_bridge(condition_vectors) * condition_mask

        batch_size, num_vectors, hidden_dim = encoded_vectors.shape
        batch_size, num_conditions, hidden_dim = condition_vectors.shape

        encoded_vectors, mask = self.combined_embedding(
            x = encoded_vectors,
            y = condition_vectors,
            x_mask = mask,
            y_mask = condition_mask)
        padding_mask = ~(mask.squeeze(2).bool())

        # -----------------------------------------------------------------------------
        # Embed vectors again into particle space
        # vectors: (batch_size, num_vectors, hidden_dim)
        # -----------------------------------------------------------------------------
        for embedding_layer in self.embedding:
            encoded_vectors = embedding_layer(encoded_vectors, mask)

        # -----------------------------------------------------------------------------
        # Add a "particle vector" which will store particle level data.
        # particle_vector: (batch_size, 1, hidden_dim)
        # combined_vectors: (batch_size, num_vectors + num_conditions + 1, hidden_dim)
        # -----------------------------------------------------------------------------
        particle_vector = self.particle_vector.expand(batch_size, 1, hidden_dim)

        if cond_vector is not None:
            cond_vector_expand = cond_vector.expand(batch_size, 1, hidden_dim)
            particle_vector = torch.cat((particle_vector, cond_vector_expand), dim=-1)
            particle_vector = self.dense_particle(particle_vector)

        combined_vectors = torch.cat((particle_vector, encoded_vectors), dim=1)

        # -----------------------------------------------------------------------------
        # Also modify the padding mask to indicate that the particle vector is real.
        # particle_padding_mask: (batch_size, 1)
        # combined_padding_mask: (batch_size, num_vectors + num_conditions + 1)
        # -----------------------------------------------------------------------------
        particle_padding_mask = padding_mask.new_zeros(batch_size, 1)
        combined_padding_mask = torch.cat((particle_padding_mask, padding_mask), dim=1)

        # -----------------------------------------------------------------------------
        # Also modify the sequence mask to indicate that the particle vector is real.
        # particle_sequence_mask:  (batch_size,1, 1)
        # combined_sequence_mask:  (batch_size, num_vectors + num_conditions + 1)
        # -----------------------------------------------------------------------------
        particle_sequence_mask = mask.new_ones(batch_size, 1, 1, dtype=torch.bool)
        combined_sequence_mask = torch.cat((particle_sequence_mask, mask), dim=1)

        # -----------------------------------------------------------------------------
        # Run all of the vectors through transformer encoder
        # combined_vectors: [T + 1, B, D]
        # particle_vector: [B, D]
        # encoded_vectors: [T, B, D]
        # -----------------------------------------------------------------------------
        combined_vectors = self.encoder(
            x= combined_vectors,
            padding_mask = combined_padding_mask,
            sequence_mask = combined_sequence_mask
        )
        combined_vectors = self.output_bridge(combined_vectors)

        particle_vector, encoded_vectors, condition_vectors = combined_vectors[:, 0, :], combined_vectors[:, 1: (1+num_vectors) , :], combined_vectors[:, (1 + num_vectors):, :]

        return encoded_vectors, condition_vectors, particle_vector
