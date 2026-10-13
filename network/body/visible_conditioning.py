"""Event-level FiLM conditioning for the TruthGeneration (neutrino) diffusion head.

Ported from the ``VisibleConditioning`` module in
YirenNTU/dgpo-omnifold-pipeline (branch ``feature/dgpo-omnifold-optimization``,
commit 65327c8, ``evenet_dgpo/evenet/network/body/visible_conditioning.py``),
keeping only the parts used by the epoch-287 FiLM diffusion model:

* per-particle encoder over the projected visible tokens, plus optional
  angular Fourier features and optional "kinematics" numerical features;
* attention pooling into one event context vector;
* one zero-initialised linear head per generator block that emits
  (attention gamma, attention beta, FFN gamma, FFN beta).

Dropped research options: token readout, reward probe, relation adapter,
pair-attention bias and classifier-side conditioning.  Configs that request
them are rejected rather than silently ignored.

New (default off): ``include_global_conditions`` appends the normalised event
global conditions (e.g. thrust axis, isolation) to the pooled context before
the modulation heads.

Parameter names match the source implementation, so a checkpoint produced
there (``TruthGeneration.visible_conditioning.*``) maps onto this module when
``include_global_conditions`` is off.

Because every modulation head starts at zero, an enabled module is an exact
identity at initialisation: the head output equals the unconditioned head.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn


FILM_GAIN_KEYS = ("attention_gamma", "attention_beta", "ffn_gamma", "ffn_beta")

# Options of the source implementation that are intentionally not ported.
_UNSUPPORTED_OPTIONS = (
    "diffusion_token_readout",
    "diffusion_reward_probe",
    "diffusion_relation_adapter",
    "diffusion_pair_attention",
)

_KNOWN_OPTIONS = {
    "diffusion_enabled", "classifier_enabled", "width", "harmonics", "theta_source", "phi_source",
    "feature_mode", "numerical_features", "numerical_frequencies", "log_diagnostics",
    "diffusion_fourier_enabled", "diffusion_film_scale_enabled", "diffusion_ffn_film_enabled",
    "diffusion_film_gains", "include_global_conditions", *_UNSUPPORTED_OPTIONS,
}


def diffusion_film_gains(options: Optional[dict]) -> dict:
    """Runtime multipliers on the four FiLM coefficients (no checkpoint tensors)."""
    options = {} if options is None else dict(options)
    unknown = set(options) - set(FILM_GAIN_KEYS)
    if unknown:
        raise ValueError(f"diffusion_film_gains accepts only {FILM_GAIN_KEYS}, got {sorted(unknown)}")
    gains = {key: options.get(key, 1.0) for key in FILM_GAIN_KEYS}
    for key, value in gains.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"FiLM gain {key} must be a finite non-negative number")
    return {key: float(value) for key, value in gains.items()}


def _flag(cfg: dict, key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"VisibleConditioning.{key} must be a boolean")
    return value


def visible_conditioning_spec(network_cfg, *, feature_names: Sequence[str], token_dim: int,
                              hidden_dim: int, num_layers: int, global_cond_dim: int = 0) -> Optional[dict]:
    """Build constructor kwargs from ``network.VisibleConditioning``; None when disabled."""
    cfg = dict(getattr(network_cfg, "VisibleConditioning", None) or {})
    unknown = set(cfg) - _KNOWN_OPTIONS
    if unknown:
        raise ValueError(f"Unknown VisibleConditioning options: {sorted(unknown)}")
    for key in _UNSUPPORTED_OPTIONS:
        if dict(cfg.get(key) or {}).get("enabled", False):
            raise ValueError(f"VisibleConditioning.{key} is not supported in this port")
    if _flag(cfg, "classifier_enabled", False):
        raise ValueError("VisibleConditioning.classifier_enabled is not supported in this port")
    if not _flag(cfg, "diffusion_enabled", False):
        return None

    spec = dict(
        feature_names=list(feature_names),
        token_dim=int(token_dim),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        width=int(cfg.get("width", 64)),
        harmonics=list(cfg.get("harmonics", [1, 2, 3, 4])),
        theta_source=str(cfg.get("theta_source", "Part_eta")),
        phi_source=str(cfg.get("phi_source", "Part_phi")),
        feature_mode=str(cfg.get("feature_mode", "angles")),
        log_diagnostics=_flag(cfg, "log_diagnostics", True),
        fourier_enabled=_flag(cfg, "diffusion_fourier_enabled", True),
        film_scale_enabled=_flag(cfg, "diffusion_film_scale_enabled", True),
        ffn_film_enabled=_flag(cfg, "diffusion_ffn_film_enabled", True),
        film_gains=diffusion_film_gains(cfg.get("diffusion_film_gains")),
        global_cond_dim=int(global_cond_dim) if _flag(cfg, "include_global_conditions", False) else 0,
    )
    if spec["feature_mode"] == "kinematics":
        spec["numerical_features"] = list(cfg.get("numerical_features", ["Part_energy", "Part_pt"]))
        spec["numerical_frequencies"] = list(cfg.get("numerical_frequencies", [0.25, 0.5, 1.0, 2.0]))
    elif spec["feature_mode"] != "angles":
        raise ValueError("VisibleConditioning.feature_mode must be 'angles' or 'kinematics'")
    return spec


class VisibleConditioning(nn.Module):
    """Visible-particle (and optionally global) event context -> per-block FiLM coefficients."""

    def __init__(self, *, feature_names: Sequence[str], token_dim: int, hidden_dim: int, num_layers: int,
                 width: int = 64, harmonics: Sequence[int] = (1, 2, 3, 4),
                 theta_source: str = "Part_eta", phi_source: str = "Part_phi",
                 feature_mode: str = "angles", numerical_features: Sequence[str] = (),
                 numerical_frequencies: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
                 log_diagnostics: bool = True, fourier_enabled: bool = True,
                 film_scale_enabled: bool = True, ffn_film_enabled: bool = True,
                 film_gains: Optional[dict] = None, global_cond_dim: int = 0):
        super().__init__()
        if min(token_dim, hidden_dim, num_layers) < 1 or width < 2 or global_cond_dim < 0:
            raise ValueError("VisibleConditioning dimensions must be positive (width >= 2)")
        if not harmonics or any(int(k) != k or k < 1 for k in harmonics) or len(set(harmonics)) != len(harmonics):
            raise ValueError("harmonics must be distinct positive integers")
        names = list(feature_names)
        if theta_source not in ("Part_eta", "Part_theta"):
            raise ValueError("theta_source must be Part_eta or Part_theta")
        if theta_source not in names or phi_source not in names:
            raise ValueError("VisibleConditioning angle sources are missing from the raw sequential features")

        self.feature_names = names
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.theta_source = theta_source
        self.theta_index, self.phi_index = names.index(theta_source), names.index(phi_source)
        self.feature_mode = feature_mode
        self.fourier_enabled = bool(fourier_enabled)
        self.film_scale_enabled = bool(film_scale_enabled)
        self.ffn_film_enabled = bool(ffn_film_enabled)
        self.film_gains = diffusion_film_gains(film_gains)
        self.global_cond_dim = int(global_cond_dim)
        self.log_diagnostics = bool(log_diagnostics)
        self.diagnostics: dict = {}

        extra_token_dim = 0
        if feature_mode == "kinematics":
            if (not numerical_features or len(set(numerical_features)) != len(numerical_features)
                    or any(n not in names or n in (theta_source, phi_source) for n in numerical_features)):
                raise ValueError("kinematic numerical features must be distinct known non-angle features")
            if (not numerical_frequencies or len(set(numerical_frequencies)) != len(numerical_frequencies)
                    or any(not math.isfinite(b) or b <= 0 for b in numerical_frequencies)):
                raise ValueError("numerical frequencies must be distinct finite positive values")
            self.numerical_indices = [names.index(n) for n in numerical_features]
            self.register_buffer("numerical_frequencies",
                                 torch.tensor(list(numerical_frequencies), dtype=torch.float32))
            extra_token_dim = len(numerical_features) * (1 + 2 * len(numerical_frequencies))
        elif feature_mode != "angles":
            raise ValueError("feature_mode must be 'angles' or 'kinematics'")
        self.register_buffer("harmonics", torch.tensor(list(harmonics), dtype=torch.float32))

        self.token_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.encoder = nn.Sequential(
            nn.Linear(token_dim + 4 * len(harmonics) + extra_token_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.pool_score = nn.Linear(width, 1, bias=False)
        self.event_encoder = nn.Sequential(
            nn.Linear(width + 1 + self.global_cond_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.LayerNorm(width, elementwise_affine=False),
        )
        # Output layout per block: [attention gamma | attention beta | FFN gamma | FFN beta].
        self.modulations = nn.ModuleList(nn.Linear(width, 4 * hidden_dim) for _ in range(num_layers))
        for head in self.modulations:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, raw: Tensor, tokens: Tensor, mask: Tensor, *,
                normalized_raw: Optional[Tensor] = None,
                global_conditions: Optional[Tensor] = None,
                global_conditions_mask: Optional[Tensor] = None) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        :param raw: raw (un-normalised) visible inputs ``x['x']``, [B, N, F_raw]
        :param tokens: normalised + projected visible tokens fed to PET, [B, N, token_dim]
        :param mask: visible mask, [B, N] or [B, N, 1]
        :param normalized_raw: normalised visible inputs before projection, [B, N, F_raw]
            (required for ``feature_mode='kinematics'``)
        :param global_conditions: normalised globals, [B, C] or [B, 1, C] (used when global_cond_dim > 0)
        :param global_conditions_mask: [B, 1] or [B, 1, 1]
        :return: one (attn_gamma, attn_beta, ffn_gamma, ffn_beta) tuple per generator block, each [B, hidden]
        """
        if raw.ndim != 3 or tokens.ndim != 3 or raw.shape[:2] != tokens.shape[:2]:
            raise ValueError("VisibleConditioning expects matching [batch, visible, features] inputs")
        if raw.shape[-1] != len(self.feature_names) or tokens.shape[-1] != self.token_dim:
            raise ValueError("VisibleConditioning feature width mismatch")
        valid = mask.squeeze(-1).bool() if mask.ndim == 3 else mask.bool()
        if valid.shape != raw.shape[:2] or raw.shape[1] == 0:
            raise ValueError("VisibleConditioning requires a [batch, visible] mask and at least one slot")

        raw = torch.where(valid[..., None], raw, 0.0)
        tokens = torch.where(valid[..., None], tokens, 0.0)

        theta = raw[..., self.theta_index]
        if self.theta_source == "Part_eta":
            small = 2.0 * torch.atan(torch.exp(-theta.abs()))
            theta = torch.where(theta >= 0.0, small, math.pi - small)
        phi = raw[..., self.phi_index]
        k = self.harmonics.to(raw)
        angles = torch.cat((theta[..., None] * k, phi[..., None] * k), dim=-1)
        fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).to(tokens)
        if not self.fourier_enabled:
            fourier = torch.zeros_like(fourier)
        parts = [self.token_norm(tokens), fourier]

        if self.feature_mode == "kinematics":
            if normalized_raw is None or normalized_raw.shape != raw.shape:
                raise ValueError("kinematics mode requires normalised visible inputs before projection")
            numeric = torch.where(valid[..., None], normalized_raw, 0.0)[..., self.numerical_indices].to(tokens)
            phase = (numeric[..., None] * self.numerical_frequencies.to(numeric)).flatten(-2)
            numeric_fourier = torch.cat((phase.sin(), phase.cos()), dim=-1)
            if not self.fourier_enabled:
                numeric_fourier = torch.zeros_like(numeric_fourier)
            parts.extend((numeric, numeric_fourier))

        encoded = self.encoder(torch.cat(parts, dim=-1))
        count = valid.sum(dim=1, keepdim=True)
        # Finite sentinel keeps empty events NaN-free.
        scores = self.pool_score(encoded).squeeze(-1).masked_fill(~valid, torch.finfo(encoded.dtype).min)
        weights = scores.softmax(dim=1) * valid.to(encoded.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (encoded * weights[..., None]).sum(dim=1)

        context_inputs = [pooled, count.to(pooled).log1p()]
        if self.global_cond_dim:
            if global_conditions is None:
                raise ValueError("include_global_conditions requires the normalised global conditions")
            g = global_conditions.reshape(global_conditions.shape[0], -1).to(pooled)
            if g.shape[-1] != self.global_cond_dim:
                raise ValueError("global condition width mismatch")
            if global_conditions_mask is not None:
                g = g * global_conditions_mask.reshape(g.shape[0], 1).to(g)
            context_inputs.append(g)
        context = self.event_encoder(torch.cat(context_inputs, dim=-1))
        present = (count > 0).to(context)

        outputs = [head(context) * present for head in self.modulations]
        if self.log_diagnostics:
            with torch.no_grad():
                packed = torch.stack(outputs).detach().float().reshape(len(outputs), raw.shape[0], 2, 2, self.hidden_dim)
                self.diagnostics = {
                    "context_rms": (context.detach().float() * present).square().mean().sqrt(),
                    "scale_rms": packed[..., 0, :].square().mean().sqrt(),
                    "shift_rms": packed[..., 1, :].square().mean().sqrt(),
                }
        return [self._policy(out.chunk(4, dim=-1)) for out in outputs]

    def _policy(self, modulation: tuple) -> tuple:
        """Apply the FFN/scale switches and runtime gains exactly once."""
        mod = list(modulation)
        if not self.ffn_film_enabled:
            mod[2], mod[3] = mod[2] * 0.0, mod[3] * 0.0
        if not self.film_scale_enabled:
            mod[0], mod[2] = torch.zeros_like(mod[0]), torch.zeros_like(mod[2])
        return tuple(m if self.film_gains[key] == 1.0 else m * self.film_gains[key]
                     for m, key in zip(mod, FILM_GAIN_KEYS))


def modulate_visible_condition(normed: Tensor, scale: Tensor, shift: Tensor, mask: Tensor) -> Tensor:
    """``normed * (1 + scale) + shift`` on masked (valid invisible) slots only; others unchanged."""
    if scale.ndim == 2:
        scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
    delta = normed * scale + shift
    return normed + torch.where(mask.bool(), delta, torch.zeros_like(delta))
