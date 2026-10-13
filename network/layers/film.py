"""FiLM conditioning for the TruthGeneration (neutrino) diffusion head.

The visible particles (and, optionally, the event global conditions) are summarised
into one event context. Per generator block, a linear head maps the context to FiLM
coefficients that update the noised invisible slots: ``h -> h * (1 + gamma) + beta``.

Configured by ``network.TruthGeneration.FiLM``; every option is required when FiLM is
enabled and the defaults live in ``share/network/*.yaml``. The heads are
zero-initialised, so an enabled FiLM leaves the generation head unchanged at the start.
"""
from typing import Optional

import torch
from torch import Tensor, nn

from evenet.network.layers.activation import create_activation

FILM_OPTIONS = (
    "enabled", "hidden_dim", "num_layers", "activation", "pooling",
    "use_multiplicity", "use_globals", "modulate_attention", "modulate_ffn", "scale",
)
POOLINGS = ("attention", "mean")


def film_kwargs(cfg, *, token_dim: int, block_dim: int, num_blocks: int, global_dim: int) -> Optional[dict]:
    """Constructor kwargs for FiLMConditioning from ``TruthGeneration.FiLM``; None when disabled."""
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(FILM_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown TruthGeneration.FiLM options: {sorted(unknown)}")
    if not cfg.get("enabled", False):
        return None
    missing = set(FILM_OPTIONS) - set(cfg)
    if missing:
        raise ValueError(f"TruthGeneration.FiLM is missing {sorted(missing)} (defaults: share/network/*.yaml)")
    for key in ("enabled", "use_multiplicity", "use_globals", "modulate_attention", "modulate_ffn", "scale"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"TruthGeneration.FiLM.{key} must be a boolean")
    kwargs = {key: cfg[key] for key in FILM_OPTIONS if key not in ("enabled", "use_globals")}
    return dict(token_dim=token_dim, block_dim=block_dim, num_blocks=num_blocks,
                global_dim=global_dim if cfg["use_globals"] else 0, **kwargs)


def _mlp(input_dim: int, hidden_dim: int, num_layers: int, activation: str) -> nn.Sequential:
    layers = []
    for index in range(num_layers):
        layers += [nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim),
                   create_activation(activation, hidden_dim)]
    return nn.Sequential(*layers)


class FiLMConditioning(nn.Module):
    """Visible particles (+ optional globals) -> FiLM coefficients for each generator block."""

    def __init__(self, *, token_dim: int, block_dim: int, num_blocks: int, global_dim: int,
                 hidden_dim: int, num_layers: int, activation: str, pooling: str, use_multiplicity: bool,
                 modulate_attention: bool, modulate_ffn: bool, scale: bool):
        super().__init__()
        if min(token_dim, block_dim, num_blocks, hidden_dim, num_layers) < 1 or global_dim < 0:
            raise ValueError("FiLM dimensions and layer counts must be positive")
        if pooling not in POOLINGS:
            raise ValueError(f"FiLM pooling must be one of {POOLINGS}")
        if not (modulate_attention or modulate_ffn):
            raise ValueError("FiLM must modulate the attention input, the FFN input, or both")

        self.block_dim = block_dim
        self.global_dim = global_dim
        self.use_multiplicity = use_multiplicity
        self.scale = scale
        # Where the coefficients are applied inside each GeneratorTransformerBlockModule.
        self.sites = [site for site, on in (("attention", modulate_attention), ("ffn", modulate_ffn)) if on]

        self.particle_encoder = _mlp(token_dim, hidden_dim, num_layers, activation)
        self.pool_score = nn.Linear(hidden_dim, 1, bias=False) if pooling == "attention" else None
        context_dim = hidden_dim + int(use_multiplicity) + global_dim
        self.event_encoder = nn.Sequential(
            _mlp(context_dim, hidden_dim, num_layers, activation),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )
        coefficients_per_site = 2 if scale else 1  # (gamma, beta) or beta only
        self.heads = nn.ModuleList(
            nn.Linear(hidden_dim, len(self.sites) * coefficients_per_site * block_dim) for _ in range(num_blocks)
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, tokens: Tensor, mask: Tensor, global_conditions: Optional[Tensor] = None,
                global_conditions_mask: Optional[Tensor] = None) -> list[dict]:
        """
        :param tokens: visible tokens as fed to PET (normalised and projected), [B, N, token_dim]
        :param mask: visible mask, [B, N] or [B, N, 1]
        :param global_conditions: normalised event globals, [B, 1, C] (required when global_dim > 0)
        :param global_conditions_mask: [B, 1, 1]
        :return: per generator block, {site: (gamma or None, beta)} with tensors of shape [B, block_dim]
        """
        valid = mask.reshape(tokens.shape[:2]).bool()
        encoded = self.particle_encoder(torch.where(valid[..., None], tokens, 0.0))
        count = valid.sum(dim=1, keepdim=True).to(encoded.dtype)

        if self.pool_score is not None:
            scores = self.pool_score(encoded).squeeze(-1).masked_fill(~valid, torch.finfo(encoded.dtype).min)
            weights = scores.softmax(dim=1) * valid
        else:
            weights = valid / count.clamp_min(1.0)
        pooled = (encoded * weights[..., None]).sum(dim=1)

        context = [pooled]
        if self.use_multiplicity:
            context.append(count.log1p())  # number of visible particles
        if self.global_dim:
            if global_conditions is None:
                raise ValueError("FiLM use_globals requires the normalised global conditions")
            globals_ = global_conditions.reshape(len(pooled), -1).to(pooled)
            if globals_.shape[-1] != self.global_dim:
                raise ValueError("FiLM global condition width mismatch")
            if global_conditions_mask is not None:
                globals_ = globals_ * global_conditions_mask.reshape(-1, 1).to(globals_)
            context.append(globals_)
        context = self.event_encoder(torch.cat(context, dim=-1))
        present = (count > 0).to(context)  # events without visible particles are not modulated

        blocks = []
        for head in self.heads:
            coefficients = (head(context) * present).split(self.block_dim, dim=-1)
            if self.scale:
                pairs = [(coefficients[2 * i], coefficients[2 * i + 1]) for i in range(len(self.sites))]
            else:
                pairs = [(None, beta) for beta in coefficients]
            blocks.append(dict(zip(self.sites, pairs)))
        return blocks


def apply_film(h: Tensor, coefficients: tuple, slot_mask: Tensor) -> Tensor:
    """``h * (1 + gamma) + beta`` on the slots selected by ``slot_mask`` [B, T, 1]; other slots unchanged."""
    gamma, beta = coefficients
    delta = beta.unsqueeze(1)
    if gamma is not None:
        delta = delta + h * gamma.unsqueeze(1)
    return torch.where(slot_mask.bool(), h + delta, h)
