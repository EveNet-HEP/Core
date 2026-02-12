from typing import List, Tuple
from torch import nn, Tensor
import torch
import numpy as np
from itertools import islice
from opt_einsum import contract_expression

from evenet.utilities.linear_form import create_symmetric_function, symmetric_tensor, contract_linear_form
from evenet.utilities.group_theory import complete_indices, symmetry_group
from evenet.network.body.object_encoder import ObjectEncoder


class SymmetricAttentionBase(nn.Module):
    WEIGHTS_INDEX_NAMES = "ijklmn"
    INPUT_INDEX_NAMES = "xyzwuv"
    DEFAULT_JET_COUNT = 16

    def __init__(self,
                 hidden_dim: int,
                 degree: int,
                 permutation_indices: List[Tuple[int, ...]] = None,
                 attention_dim: int = None):
        super(SymmetricAttentionBase, self).__init__()
        # Take hadronic top decay for example (t->bq1q2, symmetry [q1, q2])

        self.attention_dim = attention_dim
        if self.attention_dim is None:
            self.attention_dim = hidden_dim

        self.permutation_indices = [] if permutation_indices is None else permutation_indices # [[(1,2)]]
        self.features = hidden_dim
        self.features = hidden_dim
        self.degree = degree # degree: 3

        # Add any missing cycles to have a complete group
        self.permutation_indices = complete_indices(self.degree, self.permutation_indices)  # [[(1,2)], [(0,)]]
        self.permutation_group = symmetry_group(self.permutation_indices)  # [[0, 1, 2], [0, 2, 1]]
        self.no_identity_permutations = [p for p in self.permutation_group if sorted(p) != p] # [[0, 2, 1]]
        self.batch_no_identity_permutations = [(0,) + tuple(e + 1 for e in p) for p in self.no_identity_permutations] #[(0, 1, 3, 2)]

        self.weights_scale = torch.sqrt(torch.scalar_tensor(self.features)) ** self.degree


class SymmetricAttentionSplit(SymmetricAttentionBase):
    def __init__(
            self,
            hidden_dim: int,
            position_embedding_dim: int,
            num_heads: int,
            transformer_dim_scale: float,
            num_linear_layers: int,
            num_encoder_layers: int,
            dropout: float,
            degree: int,
            permutation_indices: List[Tuple[int, ...]] = None,
            attention_dim: int = None
    ):
        super(SymmetricAttentionSplit, self).__init__(
            hidden_dim=hidden_dim,
            degree=degree,
            permutation_indices=permutation_indices,
            attention_dim=attention_dim
        )

        # Each potential resonance daughter gets its own encoder to extract information for attention.

        self.encoders = nn.ModuleList(
            [ObjectEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                position_embedding_dim=position_embedding_dim,
                num_heads=num_heads,
                transformer_dim_scale=transformer_dim_scale,
                num_linear_layers=num_linear_layers,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
                conditioned=False)
                for _ in range(degree)]
        )

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, self.attention_dim) for _ in range(degree)]
        )

        self.symmetrize_tensor = create_symmetric_function(self.batch_no_identity_permutations)

        self.contraction_operation = self.make_contraction() # 'xbi,ybi,zbi,->bxyz'

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initiate parameters in the model.
        :return:
        """

        for p in self.linear_layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_contraction(self):
        input_index_names = np.array(list(self.INPUT_INDEX_NAMES)) # ['x', 'y', 'z', 'w', 'u', 'v']
        operations = map(lambda x: f"{x}bi", input_index_names) # ['xbi', 'ybi', 'zbi', 'wbi', 'ubi', 'vbi']
        operations = ','.join(islice(operations, self.degree)) # 'xbi,ybi,zbi'
        result = f"->b{''.join(input_index_names[:self.degree])}" # '->bxyz'
        return operations + result # 'xbi,ybi,zbi,->bxyz'

    def forward(self, x: Tensor, x_mask: Tensor, condition: Tensor, condition_mask: Tensor) -> Tuple[
        Tensor, List[Tensor]]:
        """ Perform symmetric attention on the hidden vectors and produce the output logits.

        This is the approximate version which learns embedding layers and computes a trivial linear form.

        Parameters
        ----------
        x : (batch_size, num_vectors, hidden_dim)
            Hidden activations after branch encoders.
        sequence_mask: (batch_size, num_vectors, 1)
            Positive mask indicating jet is real.

        Returns
        -------
        output : [T, T, ...]
            Prediction logits for this particle.
        """

        # ---------------------------------------------------------
        # Construct the transformed attention vectors for each jet.
        # ys: [(batch_size, num_vectors, hidden_dim), ...]
        # ---------------------------------------------------------
        ys = []
        daughter_vectors = []
        for encoder, linear_layer in zip(self.encoders, self.linear_layers):
            # ------------------------------------------------------
            # First pass the input through this jet's encoder stack.
            # y: (batch_size, num_vectors, hidden_dim)
            # ------------------------------------------------------
            y, y_cond, daughter_vector = encoder(
                encoded_vectors=x,
                mask=x_mask,
                condition_vectors=condition,
                condition_mask=condition_mask
            )
            # --------------------------------------------------------
            # Flatten and apply the final linear layer to each vector.
            # y: [T, B, D]
            # ---------------------------------------------------------
            y = linear_layer(y)
            y = y * x_mask
            y = y.transpose(0, 1) # [T, B, D]

            # Accumulate vectors into stack for each daughter of this particle.
            daughter_vectors.append(daughter_vector)
            ys.append(y)

        # -------------------------------------------------------
        # Construct the output logits via general self-attention.
        # output: [T, T, ...]
        # -------------------------------------------------------
        output = torch.einsum(self.contraction_operation, *ys) # [B, T, T, ...]
        output = output / self.weights_scale

        # ---------------------------------------------------
        # Symmetrize the output according to group structure.
        # output: [T, T, ...]
        # ---------------------------------------------------
        # TODO Perhaps make the encoder layers match in the symmetric dimensions.
        output = self.symmetrize_tensor(output)

        # TODO: Not exactly the same as paper, need index attention to be consistent with the paper.

        return output, daughter_vectors


class SymmetricAttentionFull(SymmetricAttentionBase):
    def __init__(self,
                 hidden_dim: int,
                 position_embedding_dim: int,
                 num_heads: int,
                 transformer_dim_scale: float,
                 num_linear_layers: int,
                 num_encoder_layers: int,
                 dropout: float,
                 degree: int,
                 permutation_indices: List[Tuple[int, ...]] = None,
                 attention_dim: int = None):
        super(SymmetricAttentionFull, self).__init__(
            hidden_dim=hidden_dim,
            degree=degree,
            permutation_indices=permutation_indices,
            attention_dim=attention_dim
        )

        self.weights_shape = [self.features] * self.degree
        self.weights = nn.Parameter(torch.randn(*self.weights_shape))

        self.output_operation = self.make_contraction()

        self.reset_parameters()

    def make_contraction(self):
        weights_index_names = np.array(list(self.WEIGHTS_INDEX_NAMES))
        input_index_names = np.array(list(self.INPUT_INDEX_NAMES))
        batch_index_name = 'b'

        operations = map(lambda x: batch_index_name + ''.join(x), zip(input_index_names, weights_index_names))
        operations = ','.join(islice(operations, self.degree))

        operand = f",{''.join(weights_index_names[:self.degree])}"
        result = f"->b{''.join(input_index_names[:self.degree])}"

        expression = operations + operand + result
        shapes = [(self.batch_size, self.DEFAULT_JET_COUNT, self.features)] * self.degree
        shapes.append((self.features,) * self.degree)
        return contract_expression(expression, *shapes, optimize='optimal')

    def reset_parameters(self) -> None:
        # bound = 1 / math.sqrt(self.weights.shape[1])
        # nn.init.uniform_(self.weights, -bound, bound)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: Tensor, x_mask: Tensor, condition: Tensor, condition_mask: Tensor) -> Tuple[
        Tensor, List[Tensor]]:
        """ Perform symmetric attention on the hidden vectors and produce the output logits.

        This is the full version which creates the N^D tensor and perfoms a general linear form calculation.

        Parameters
        ----------
        x : (batch_size, num_vectors, hidden_dim)
            Hidden activations after branch encoders.


        Returns
        -------
        output : [T, T, ...]
            Prediction logits for this particle.
        """

        # Enforce that symmetries of the particle permutation group
        # symmetric_weights: [D, D, ...] Symmetric layer weights
        symmetric_weights = symmetric_tensor(self.weights, self.no_identity_permutations)
        # symmetric_weights = symmetric_weights / self.weights_scale

        # symmetric_weights = symmetric_weights ** (1 / self.order)
        # Perform the generalized matrix multiplication operation.
        # output: [B, T, T, ...] Symmetric output distribution
        # output_operands = [x] * self.order + [symmetric_weights]
        # output = self.output_operation(*output_operands, backend='torch')
        output = contract_linear_form(symmetric_weights, x)

        output = symmetric_tensor(output, self.batch_no_identity_permutations)

        return output
