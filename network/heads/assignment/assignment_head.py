from typing import Tuple, List
from opt_einsum import contract_expression

import torch
from torch import nn, Tensor

from evenet.dataset.types import Symmetries
from evenet.network.body.object_encoder import ObjectEncoder
from evenet.network.heads.assignment.symmetric_attention import SymmetricAttentionSplit, SymmetricAttentionFull
from evenet.network.heads.classification.classification_head import BranchLinear
from evenet.network.layers.transformer import ClassifierTransformerBlockModule

from evenet.utilities.masked_softmax_no_gradient import masked_log_softmax
from evenet.network.body.normalizer import Normalizer
from evenet.network.layers.utils import RandomDrop
from evenet.network.layers.activation import create_activation, create_residual_connection
from typing import Dict
import re
from collections import OrderedDict


class AssignmentHead(nn.Module):
    WEIGHTS_INDEX_NAMES = "ijklmn"
    DEFAULT_JET_COUNT = 16

    def __init__(
            self,
            split_attention: bool,
            input_dim: int,
            hidden_dim: int,
            position_embedding_dim: int,
            num_heads: int,
            transformer_dim_scale: float,
            num_linear_layers: int,
            num_encoder_layers: int,
            num_jet_linear_layers: int,
            num_jet_encoder_layers: int,
            num_detection_layers: int,
            dropout: float,
            combinatorial_scale: float,
            product_names: List[str],
            product_symmetries: Symmetries,
            detection_output_dim: int = 1,
            softmax_output: bool = True,
            skip_connection: bool = False,
            encoder_skip_connection: bool = False,
    ):
        super(AssignmentHead, self).__init__()
        # Take hadronic top decay for example (t->bq1q2, symmetry [q1, q2])

        self.degree = product_symmetries.degree  # degree: 3
        self.product_names = product_names  # product_name: ['b', 'q1', 'q2']
        self.softmax_output = softmax_output

        self.combinatorial_scale = combinatorial_scale

        self.encoder = ObjectEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            position_embedding_dim=position_embedding_dim,
            num_heads=num_heads,
            transformer_dim_scale=transformer_dim_scale,
            num_linear_layers=num_linear_layers,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            conditioned=True,
            skip_connection=skip_connection,
            encoder_skip_connection=encoder_skip_connection,
        )

        attention_layer = SymmetricAttentionSplit if split_attention else SymmetricAttentionFull
        self.attention = attention_layer(
            hidden_dim=hidden_dim,
            position_embedding_dim=position_embedding_dim,
            num_heads=num_heads,
            transformer_dim_scale=transformer_dim_scale,
            num_linear_layers=num_jet_linear_layers,
            num_encoder_layers=num_jet_encoder_layers,
            dropout=dropout,
            degree=self.degree,
            permutation_indices=product_symmetries.permutations
        )

        self.detection_classifier = BranchLinear(
            num_layers=num_detection_layers,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_outputs=detection_output_dim,
            dropout=dropout,
            batch_norm=True
        )

        self.num_targets = len(self.attention.permutation_group)  # 2 = len([[0, 1, 2], [0, 2, 1]])
        self.permutation_indices = self.attention.permutation_indices  # [[(1,2)], [(0,)]

        self.padding_mask_operation = self.create_padding_mask_operation()
        self.diagonal_mask_operation = self.create_diagonal_mask_operation()
        self.diagonal_mask = {}

    def create_padding_mask_operation(self):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]  # 'ijk'
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))  # 'bi, bj, bk'
        expression = f"{operands}->b{weights_index_names}"  # 'bi, bj, bk-> 'bijk'
        return expression

    def create_diagonal_mask_operation(self):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]  # 'ijk'
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))  # 'bi, bj, bk'
        expression = f"{operands}->{weights_index_names}"  # 'bi, bj, bk-> ijk'
        return expression

    def create_output_mask(self, output: Tensor, mask: Tensor) -> Tensor:

        num_jets = output.shape[1]

        # batch_sequence_mask: [B, T, 1] Positive mask indicating jet is real.
        batch_sequence_mask = mask

        # =========================================================================================
        # Padding mask
        # =========================================================================================
        padding_mask_operands = [batch_sequence_mask.squeeze(-1) * 1] * self.degree
        padding_mask = torch.einsum(self.padding_mask_operation, *padding_mask_operands)
        padding_mask = padding_mask.bool()

        # =========================================================================================
        # Diagonal mask
        # =========================================================================================
        try:
            diagonal_mask = self.diagonal_mask[(num_jets, output.device)]
        except KeyError:
            identity = 1 - torch.eye(num_jets)  # num_jets x num_jets diagonal matrix
            identity = identity.type_as(output)

            diagonal_mask_operands = [identity * 1] * self.degree
            diagonal_mask = torch.einsum(self.diagonal_mask_operation, *diagonal_mask_operands)
            diagonal_mask = diagonal_mask.unsqueeze(0) < (num_jets + 1 - self.degree)
            self.diagonal_mask[(num_jets, output.device)] = diagonal_mask

        return (padding_mask & diagonal_mask).bool()

    def forward(
            self,
            point_cloud: Tensor,
            point_cloud_mask: Tensor,
            global_condition: Tensor,
            global_condition_mask: Tensor,
            cond_vector: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """ Create a distribution over jets for a given particle and a probability of its existence.

        Parameters
        ----------
        point_clould: Tensor, shape (batch_size, num_vectors, hidden_dim)
        point_cloud_mask: Tensor, shape (batch_size, num_vectors, 1)
        global_condition: Tensor, shape (batch_size, 1, hidden_dim)
        global_condition_mask: Tensor, shape (batch_size, 1, 1)
        cond_vector: Tensor, shape (batch_size, 1, hidden_dim)

        Returns
        -------
        selection : [TS, TS, ...]
            Distribution over sequential vectors for the target vectors.
        classification: [B]
            Probability of this particle existing in the data.
        """

        # ------------------------------------------------------
        # Apply the branch's independent encoder to each vector.
        # particle_vectors : (batch_size, num_vectors, hidden_dim)
        # ------------------------------------------------------


        encoded_vectors, encoded_global_cond, particle_vector = self.encoder(
            point_cloud, point_cloud_mask,
            global_condition, global_condition_mask,
            cond_vector
        )

        # -----------------------------------------------
        # Run the encoded vectors through the classifier.
        # detection: [B, 1]
        # -----------------------------------------------
        detection = self.detection_classifier(particle_vector).squeeze(-1)

        # --------------------------------------------------------
        # Extract sequential vectors only for the assignment step.
        # sequential_particle_vectors : [TS, B, D]
        # sequential_padding_mask : [B, TS]
        # sequential_sequence_mask : [TS, B, 1]
        # --------------------------------------------------------
        sequential_particle_vectors = encoded_vectors
        sequential_padding_mask = ~(point_cloud_mask.squeeze(-1)).bool()
        sequential_sequence_mask = point_cloud_mask

        # --------------------------------------------------------------------
        # Create the vector distribution logits and the correctly shaped mask.
        # assignment : [TS, TS, ...]
        # assignment_mask : [TS, TS, ...]
        # --------------------------------------------------------------------
        assignment, daughter_vectors = self.attention(
            x=sequential_particle_vectors,
            x_mask=sequential_sequence_mask,
            condition=encoded_global_cond,
            condition_mask=global_condition_mask
        )

        assignment_mask = self.create_output_mask(assignment, sequential_sequence_mask)

        # ---------------------------------------------------------------------------
        # Need to reshape output to make softmax-calculation easier.
        # We transform the mask and output into a flat representation.
        # Afterwards, we apply a masked log-softmax to create the final distribution.
        # output : [TS, TS, ...]
        # mask : [TS, TS, ...]
        # ---------------------------------------------------------------------------
        if self.softmax_output:
            original_shape = assignment.shape
            batch_size = original_shape[0]

            assignment = assignment.reshape(batch_size, -1)
            assignment_mask = assignment_mask.reshape(batch_size, -1)

            assignment = masked_log_softmax(assignment, assignment_mask)
            assignment = assignment.view(*original_shape)

            # mask = mask.view(*original_shape)
            # offset = torch.log(mask.sum((1, 2, 3), keepdims=True).float()) * self.combinatorial_scale
            # output = output + offset

        return assignment, detection, assignment_mask, particle_vector, daughter_vectors


class SharedAssignmentHead(nn.Module):
    def __init__(
            self,
            resonance_particle_properties_mean: Tensor,
            resonance_particle_properties_std: Tensor,
            pairing_topology: Dict,
            process_names: List[str],
            pairing_topology_category: Dict,
            event_particles: Dict,
            event_permutation: Dict,
            product_particles: Dict,
            product_symmetries: Dict,
            feature_drop: float,
            num_feature_keep: int,
            input_dim: int,
            hidden_dim: int,
            position_embedding_dim: int,
            num_attention_heads: int,
            transformer_dim_scale: float,
            num_linear_layers: int,
            num_encoder_layers: int,
            num_jet_embedding_layers: int,
            num_jet_encoder_layers: int,
            num_detection_layers: int,
            num_max_event_particles: int,
            dropout: float,
            combinatorial_scale: float,
            split_attention: bool,
            encode_event_token: bool,
            activation: str,
            skip_connection: bool,
            encoder_skip_connection: bool,
            device: str,
    ):
        super(SharedAssignmentHead, self).__init__()
        # Initialize the resonance particle condition

        self.device = device
        self.pairing_topology = pairing_topology
        self.pairing_topology_category = pairing_topology_category
        self.event_permutation = event_permutation
        self.event_particles = event_particles
        self.product_symmetries = product_symmetries
        self.product_particles = product_particles

        self.process_names = process_names

        self.hidden_dim = hidden_dim

        # Resonance structure conditioner
        self.resonance_particle_properties_mean = resonance_particle_properties_mean
        self.resonance_particle_properties_std = resonance_particle_properties_std
        self.num_resonance_particle_feature = self.resonance_particle_properties_mean.size(0)
        self.resonance_particle_properties = (
            nn.ParameterDict({
                topology_name:
                    nn.Parameter(
                        pairing_topology[topology_name][
                            "resonance_particle_properties"].to(self.device),
                        requires_grad=False)
                for topology_name in self.pairing_topology}
            )
        )
        self.resonance_particle_properties_normalizer = Normalizer(
            mean=resonance_particle_properties_mean.to(self.device),
            std=resonance_particle_properties_std.to(self.device),
            norm_mask=torch.ones_like(resonance_particle_properties_mean, device=self.device).bool(),
        )

        self.resonance_particle_embed = nn.Sequential(
            RandomDrop(feature_drop, num_feature_keep),
            nn.Linear(self.num_resonance_particle_feature, self.hidden_dim),
            create_activation(activation, self.hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # [5] Assignment Head
        self.multiprocess_assign_head = nn.ModuleDict({
            topology_name: AssignmentHead(
                split_attention=split_attention,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                position_embedding_dim=position_embedding_dim,
                num_heads=num_attention_heads,
                transformer_dim_scale=transformer_dim_scale,
                num_linear_layers=num_linear_layers,
                num_encoder_layers=num_encoder_layers,
                num_jet_linear_layers=num_jet_embedding_layers,
                num_jet_encoder_layers=num_jet_encoder_layers,
                num_detection_layers=num_detection_layers,
                dropout=dropout,
                combinatorial_scale=combinatorial_scale,
                product_names=pairing_topology_category[topology_name]["product_particles"].names,
                product_symmetries=pairing_topology_category[topology_name]["product_symmetry"],
                softmax_output=True,
                detection_output_dim=num_max_event_particles + 1,
                skip_connection=skip_connection,
                encoder_skip_connection=encoder_skip_connection
            )
            for topology_name in pairing_topology_category
        })

        # [event_token]
        self.encode_event_token = encode_event_token
        if encode_event_token:
            self.position_token = nn.Parameter(
                torch.randn(1, len(pairing_topology), position_embedding_dim)
            )
            self.mlp_for_event_point_cloud = nn.Linear(hidden_dim + position_embedding_dim, hidden_dim)
            self.class_transformer = ClassifierTransformerBlockModule(
                input_dim=input_dim,
                projection_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )

    def forward(self,
                x: Tensor,
                x_mask: Tensor,
                global_condition: Tensor,
                global_condition_mask: Tensor,
                event_token=None,
                return_type: str = "process_base"):
        """

        :param x: (batch_size, num_vectors, hidden_dim)
        :param x_mask: (batch_size, num_vectors, 1)
        :param global_condition: (batch_size, num_global_condition, hidden_dim)
        :param global_condition_mask: (batch_size, num_global_condition, 1)
        :param return_type: str

        :return:
        - return_type == "process_base":
            Assignment: Dict[process] = {
                List of assignments for each event particles result, shape: List([B, T, T, ...])
            }
            Detection: Dict[process] = {
                List of detections for each event particles result , shape: List([B, num_max_event_particles])
            }
        - if self.output_event_token:
            event_token: Transformer based encoded event token,
        - else:
            event_token: remain the same.
        """

        # Pass the shared hidden state to every decoder branch
        branch_decoder_result = dict()
        batch_size = x.shape[0]
        event_particle_vector_arrays = []
        for topology_name in self.pairing_topology:
            # Condition embedding for each assignment head
            topology_category_name = self.pairing_topology[topology_name]["pairing_topology_category"]
            condition_variable = self.resonance_particle_properties_normalizer(
                self.resonance_particle_properties[topology_name]
            )
            num_res = condition_variable.shape[-1]
            condition_variable = condition_variable.view(1, 1, num_res).expand(batch_size, 1, num_res)
            condition_variable = self.resonance_particle_embed(condition_variable)
            (
                assignment,
                detection,
                assignment_mask,
                event_particle_vector,
                product_particle_vectors
            ) = self.multiprocess_assign_head[topology_category_name](
                point_cloud=x,
                point_cloud_mask=x_mask,
                global_condition=global_condition,
                global_condition_mask=global_condition_mask,
                cond_vector=condition_variable,
            )
            event_particle_vector_arrays.append(event_particle_vector.unsqueeze(1))
            branch_decoder_result[topology_name] = {"assignment": assignment, "detection": detection}

        if event_token is not None and self.encode_event_token:
            event_particle_cloud = torch.cat(
                (
                    torch.cat(event_particle_vector_arrays, dim=1),
                    self.position_token.expand(batch_size, -1, -1)
                ),
                dim=-1
            )
            event_particle_cloud = self.mlp_for_event_point_cloud(event_particle_cloud)
            event_token = self.class_transformer(
                x=event_particle_cloud,
                class_token=event_token
            )

        assignments = OrderedDict()
        detections = OrderedDict()

        if return_type == "process_base":
            for process in self.process_names:
                assignments[process] = []
                detections[process] = []
                for event_particle_name, product_symmetry in self.product_symmetries[process].items():
                    topology_name = ''.join(self.product_particles[process][event_particle_name].names)
                    topology_name = f"{event_particle_name}/{topology_name}"
                    topology_name = re.sub(r'\d+', '', topology_name)
                    assignments[process].append(branch_decoder_result[topology_name]["assignment"])
                    detections[process].append(branch_decoder_result[topology_name]["detection"])

        else:
            for topology_name in branch_decoder_result:
                assignments[topology_name] = branch_decoder_result[topology_name]["assignment"]
                detections[topology_name] = branch_decoder_result[topology_name]["detection"]

        return assignments, detections, event_token
