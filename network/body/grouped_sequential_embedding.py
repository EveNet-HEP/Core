from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GroupedSequentialNodeSpec:
    name: str
    output_dim: int
    index: tuple[int, ...] = ()
    inputs: tuple[Any, ...] = ()


def _build_node_spec(name: str, raw_node: dict) -> GroupedSequentialNodeSpec:
    inputs = []
    for child in raw_node.get("input", []):
        if isinstance(child, str):
            inputs.append(child)
            continue

        child_name, child_value = next(iter(child.items()))
        inputs.append(_build_node_spec(child_name, child_value))

    return GroupedSequentialNodeSpec(
        name=name,
        output_dim=int(raw_node["output_dim"]),
        index=tuple(int(index) for index in raw_node.get("index", [])),
        inputs=tuple(inputs),
    )


def build_grouped_node_specs(raw_config: dict) -> tuple[GroupedSequentialNodeSpec, ...]:
    groups = raw_config.get("groups", {})
    if not groups:
        return ()

    return tuple(
        _build_node_spec(group_name, group_cfg)
        for group_name, group_cfg in groups.items()
    )


class GroupedSequentialNode(nn.Module):
    def __init__(
        self,
        spec: GroupedSequentialNodeSpec,
        feature_index: dict[str, int],
        hidden_dim_scale: float,
        dropout: float,
    ):
        super().__init__()
        self.name = spec.name
        self.output_dim = spec.output_dim

        leaf_indices = []
        child_specs = []
        child_sequence = []
        for child in spec.inputs:
            if isinstance(child, str):
                if child not in feature_index:
                    raise ValueError(
                        f"Grouped sequential input '{self.name}' references unknown raw feature '{child}'."
                    )
                leaf_indices.append(feature_index[child])
                child_sequence.append(("leaf", len(leaf_indices) - 1))
            else:
                child_specs.append(child)
                child_sequence.append(("group", len(child_specs) - 1))

        input_dim = len(leaf_indices) + sum(child.output_dim for child in child_specs)
        if input_dim <= 0:
            raise ValueError(f"Grouped sequential input '{self.name}' has no usable children.")

        self.leaf_indices = tuple(leaf_indices)
        self.child_sequence = tuple(child_sequence)
        self.child_groups = nn.ModuleList(
            GroupedSequentialNode(
                spec=child_spec,
                feature_index=feature_index,
                hidden_dim_scale=hidden_dim_scale,
                dropout=dropout,
            )
            for child_spec in child_specs
        )

        self.use_identity = all(kind == "leaf" for kind, _ in self.child_sequence) and input_dim == self.output_dim
        if self.use_identity:
            self.encoder = nn.Identity()
        else:
            hidden_dim = max(self.output_dim, int(round(input_dim * hidden_dim_scale)))
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(approximate="none"),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
                nn.GELU(approximate="none"),
            )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Normalized raw sequential features with shape [batch, particle, raw_feature].
        mask:
            Sequential mask with shape [batch, particle, 1].
        """
        parts = []
        for kind, index in self.child_sequence:
            if kind == "leaf":
                leaf_index = self.leaf_indices[index]
                parts.append(x[..., leaf_index : leaf_index + 1])
            else:
                parts.append(self.child_groups[index](x, mask))

        encoded = torch.cat(parts, dim=-1)
        encoded = self.encoder(encoded)
        return encoded * mask.float()


class GroupedSequentialEmbedding(nn.Module):
    def __init__(
        self,
        raw_feature_names: list[str],
        grouped_config: dict,
        hidden_dim_scale: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        source_cfg = grouped_config.get("Source", grouped_config)
        projected_feature_names = tuple(source_cfg.get("projected_feature_names", ()))
        if not projected_feature_names:
            raise ValueError("Grouped sequential config is missing projected_feature_names.")

        self.output_dim = len(projected_feature_names)
        self.projected_feature_names = projected_feature_names

        feature_index = {
            feature_name: index
            for index, feature_name in enumerate(raw_feature_names)
        }
        group_specs = build_grouped_node_specs(source_cfg)
        if not group_specs:
            raise ValueError("Grouped sequential config is missing root groups.")

        self.group_nodes = nn.ModuleList(
            GroupedSequentialNode(
                spec=group_spec,
                feature_index=feature_index,
                hidden_dim_scale=hidden_dim_scale,
                dropout=dropout,
            )
            for group_spec in group_specs
        )
        self.group_indices = tuple(group_spec.index for group_spec in group_specs)

        occupied_slots: set[int] = set()
        for group_spec in group_specs:
            if len(group_spec.index) != group_spec.output_dim:
                raise ValueError(
                    f"Root grouped input '{group_spec.name}' has output_dim={group_spec.output_dim} "
                    f"but index width={len(group_spec.index)}."
                )
            for slot in group_spec.index:
                if slot < 0 or slot >= self.output_dim:
                    raise ValueError(
                        f"Root grouped input '{group_spec.name}' writes to invalid projected slot {slot}."
                    )
                if slot in occupied_slots:
                    raise ValueError(f"Projected sequential slot {slot} is assigned more than once.")
                occupied_slots.add(slot)
        if occupied_slots != set(range(self.output_dim)):
            raise ValueError(
                "Root grouped inputs must fill the full projected sequential range exactly once."
            )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Normalized raw sequential features with shape [batch, particle, raw_feature].
        mask:
            Sequential mask with shape [batch, particle, 1].

        Returns
        -------
        Tensor
            Projected sequential features with shape [batch, particle, projected_feature].
        """
        projected = x.new_zeros(x.size(0), x.size(1), self.output_dim)
        for group_indices, group_node in zip(self.group_indices, self.group_nodes):
            projected[..., list(group_indices)] = group_node(x, mask)
        return projected * mask.float()
