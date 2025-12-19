from torch import nn, Tensor
from typing import Dict
from evenet.network.layers.linear_block import create_linear_block
from evenet.network.body.normalizer import Normalizer
from collections import OrderedDict
import torch

import numpy as np
from evenet.network.layers.activation import  create_residual_connection
from evenet.network.layers.transformer import ClassifierTransformerBlockModule


class BranchLinear(nn.Module):

    def __init__(
            self,
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            num_outputs: int = 1,
            dropout: float = 0.0,
            batch_norm: bool = False,
            skip_connection: bool = False
    ):
        super(BranchLinear, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.bridge = create_residual_connection(
            skip_connection=True,
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
        )

        self.hidden_layers = nn.ModuleList([
            create_linear_block(
                linear_block_type="GRU",
                input_dim=self.hidden_dim,
                hidden_dim_scale=2.0,
                output_dim=self.hidden_dim,
                normalization_type="LayerNorm",
                activation_type="gelu",
                dropout=dropout,
                skip_connection=skip_connection
            ) for _ in range(self.num_layers)])

        # TODO Play around with this normalization layer
        if batch_norm:
            self.output_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.output_norm = nn.Identity()

        self.output_layer = nn.Linear(hidden_dim, num_outputs)

    def forward(self, single_vector: Tensor) -> Tensor:
        """ Produce a single classification output for a sequence of vectors.

        Parameters
        ----------
        single_vector : [B, D]
            Hidden activations after central encoder.

        Returns
        -------
        classification: [B, O]
            Probability of this particle existing in the data.
        """
        single_vector = self.bridge(single_vector)
        batch_size, input_dim = single_vector.shape

        # -----------------------------------------------------------------------------
        # Convert our single vector into a sequence of length 1.
        # Mostly just to re-use previous code.
        # sequence_mask: [1, B, 1]
        # single_vector: [1, B, D]
        # -----------------------------------------------------------------------------
        sequence_mask = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=single_vector.device)
        single_vector = single_vector.view(batch_size, 1, input_dim)

        # ---------------------------------------------------------------------------
        # Run through hidden layer stack first, and then take the first timestep out.
        # hidden : [B, H]
        # ----------------------------------------------------------------------------
        for layer in self.hidden_layers:
            single_vector = layer(single_vector, sequence_mask)
        hidden = single_vector.view(batch_size, self.hidden_dim)

        # ------------------------------------------------------------
        # Run through the linear layer stack and output the result
        # classification : [B, O]
        # ------------------------------------------------------------
        classification = self.output_layer(self.output_norm(hidden))

        return classification


class ClassificationHead(nn.Module):
    def __init__(
            self,
            class_label,
            event_num_classes,
            num_attention_heads: int,
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            dropout: float = 0.0,
            skip_connection: bool = False,
    ):
        super(ClassificationHead, self).__init__()

        self.class_transformer = ClassifierTransformerBlockModule(
            input_dim=input_dim,
            projection_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        networks = OrderedDict()
        for name in class_label:
            num_classes = event_num_classes[name]
            networks[f"classification/{name}"] = BranchLinear(
                input_dim=hidden_dim,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_outputs=num_classes,
                dropout=dropout,
                batch_norm=True,
                skip_connection=skip_connection
            )
        self.networks = nn.ModuleDict(networks)

    def forward(self, x, x_mask, event_token) -> Dict[str, Tensor]:
        """
        :param x: input point cloud (batch_size, hidden_dim)
        :return: Dict[str, Tensor]
        """
        class_token = event_token.clone()
        class_token = self.class_transformer(
            x = x,
            class_token = class_token,
            mask = x_mask
        )
        class_token = event_token + class_token

        return {
            key: network(class_token)
            for key, network in self.networks.items()
        }


class RegressionHead(nn.Module):
    def __init__(
            self,
            regressions_target,
            regression_names,
            means: Dict[str, Tensor],
            stds: Dict[str, Tensor],
            num_layers: int,
            input_dim: int,
            hidden_dim: int,
            device,
            dropout: float = 0.0,
            skip_connection: bool = False
    ):
        super(RegressionHead, self).__init__()
        networks = OrderedDict()
        normalizers = OrderedDict()
        self.regressions_target = OrderedDict()
        for name in regressions_target:
            target_list = []
            for target in regression_names:
                # regression_names Format : "process_name/target_name"
                process_name = target.split("/")[0]
                if process_name == name:
                    target_list.append(target)
            num_regressions = len(target_list)
            if num_regressions == 0:
                continue
            networks[f"regression/{name}"] = BranchLinear(
                input_dim=input_dim,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_outputs=num_regressions,
                dropout=dropout,
                batch_norm=True,
                skip_connection=skip_connection
            )
            self.regressions_target[name] = target_list
            mean = torch.cat([means[target].unsqueeze(-1) for target in target_list], dim=-1)
            std = torch.cat([stds[target].unsqueeze(-1) for target in target_list], dim=-1)
            normalizers[f"regression/{name}"] = Normalizer(
                mean=mean.to(device),
                std=std.to(device),
                norm_mask=torch.ones_like(mean, dtype=torch.bool).to(device),
            )
        self.networks = nn.ModuleDict(networks)
        self.normalizers = nn.ModuleDict(normalizers)

    def forward(self, x) -> Dict[str, Tensor]:
        """

        :param x: event token, shape (batch_size, hidden_dim)
        :return:
        """

        regression = {
            key: network(x)
            for key, network in self.networks.items()
        }
        # Apply the normalizers to the regression outputs
        regression_denormalized = {
            key: normalizer.denormalize(regression[key])
            for key, normalizer in self.normalizers.items()
        }
        return regression_denormalized
