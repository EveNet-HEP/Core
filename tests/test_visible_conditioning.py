"""CPU unit tests for the TruthGeneration FiLM conditioning (visible_conditioning.py)."""
import copy
from types import SimpleNamespace

import pytest
import torch

from evenet.network.body.visible_conditioning import (
    VisibleConditioning, modulate_visible_condition, visible_conditioning_spec,
)
from evenet.network.heads.generation.generation_head import EventGenerationHead

NAMES = ["Part_energy", "Part_pt", "Part_eta", "Part_phi", "tag"]


def spec(**kw):
    out = dict(feature_names=NAMES, token_dim=5, hidden_dim=8, num_layers=2, width=12)
    out.update(kw)
    return out


def visible(batch=3, n=4):
    torch.manual_seed(0)
    raw, tokens = torch.randn(batch, n, 5), torch.randn(batch, n, 5)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)[..., None]
    return raw, tokens, mask


def open_heads(module, std=0.05):
    with torch.no_grad():
        for head in module.modulations:
            head.weight.normal_(std=std)
            head.bias.fill_(0.02)


def test_zero_init_masking_and_permutation():
    module = VisibleConditioning(**spec())
    raw, tokens, mask = visible()
    raw[~mask.expand_as(raw)] = float("nan")       # padded slots must be ignored
    tokens[~mask.expand_as(tokens)] = float("nan")
    out = module(raw, tokens, mask)
    assert len(out) == 2 and all(len(layer) == 4 for layer in out)
    assert all(t.count_nonzero() == 0 for layer in out for t in layer)  # identity at init
    sum(t.sum() for layer in out for t in layer).backward()
    assert module.modulations[0].weight.grad.norm() > 0                # heads still learn

    open_heads(module)
    out = module(raw, tokens, mask)
    flipped = module(raw.flip(1), tokens.flip(1), mask.flip(1))
    for a_layer, b_layer in zip(out, flipped):
        for a, b in zip(a_layer, b_layer):
            assert torch.isfinite(a).all()
            assert a[1].count_nonzero() == 0                            # empty event -> no modulation
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)      # permutation invariant


def test_kinematics_mode_requires_and_uses_normalized_inputs():
    module = VisibleConditioning(**spec(feature_mode="kinematics", numerical_features=["Part_energy", "Part_pt"]))
    open_heads(module)
    raw, tokens, mask = visible()
    with pytest.raises(ValueError):
        module(raw, tokens, mask)
    a = module(raw, tokens, mask, normalized_raw=raw)[0][0]
    changed = raw.clone(); changed[..., 0] += 1.0
    b = module(raw, tokens, mask, normalized_raw=changed)[0][0]
    assert not torch.allclose(a[0], b[0])


def test_switches_and_gains():
    raw, tokens, mask = visible()
    base = VisibleConditioning(**spec()); open_heads(base)
    ref = base(raw, tokens, mask)

    no_ffn = VisibleConditioning(**spec(ffn_film_enabled=False)); no_ffn.load_state_dict(base.state_dict())
    no_scale = VisibleConditioning(**spec(film_scale_enabled=False)); no_scale.load_state_dict(base.state_dict())
    gains = dict(attention_gamma=0.5, attention_beta=1.0, ffn_gamma=0.0, ffn_beta=2.0)
    gained = VisibleConditioning(**spec(film_gains=gains)); gained.load_state_dict(base.state_dict())

    for r, f, s, g in zip(ref, no_ffn(raw, tokens, mask), no_scale(raw, tokens, mask), gained(raw, tokens, mask)):
        torch.testing.assert_close(f[0], r[0]); torch.testing.assert_close(f[1], r[1])
        assert f[2].count_nonzero() == 0 and f[3].count_nonzero() == 0
        assert s[0].count_nonzero() == 0 and s[2].count_nonzero() == 0
        torch.testing.assert_close(s[1], r[1]); torch.testing.assert_close(s[3], r[3])
        for value, reference, key in zip(g, r, ("attention_gamma", "attention_beta", "ffn_gamma", "ffn_beta")):
            torch.testing.assert_close(value, reference * gains[key])


def test_global_conditions_enter_context():
    raw, tokens, mask = visible()
    module = VisibleConditioning(**spec(global_cond_dim=3)); open_heads(module)
    g = torch.randn(3, 1, 3)
    with pytest.raises(ValueError):
        module(raw, tokens, mask)
    a = module(raw, tokens, mask, global_conditions=g)[0][0]
    b = module(raw, tokens, mask, global_conditions=g + 1.0)[0][0]
    assert not torch.allclose(a[0], b[0])
    # A masked-out global condition is equivalent to zeros.
    off = torch.zeros(3, 1, 1)
    c = module(raw, tokens, mask, global_conditions=g, global_conditions_mask=off)[0][0]
    d = module(raw, tokens, mask, global_conditions=torch.zeros_like(g))[0][0]
    torch.testing.assert_close(c, d)


def test_modulate_only_masked_slots():
    normed = torch.randn(2, 5, 8)
    scale, shift = torch.randn(2, 8), torch.randn(2, 8)
    slot_mask = torch.zeros(2, 5, 1, dtype=torch.bool); slot_mask[:, 3:] = True
    out = modulate_visible_condition(normed, scale, shift, slot_mask)
    torch.testing.assert_close(out[:, :3], normed[:, :3])
    torch.testing.assert_close(out[:, 3:], normed[:, 3:] * (1 + scale[:, None]) + shift[:, None])


def make_head(film):
    torch.manual_seed(1)
    return EventGenerationHead(
        input_dim=8, projection_dim=8, num_global_cond=8, num_classes=3, output_dim=2,
        num_layers=2, num_heads=2, dropout=0.0, layer_scale=False, layer_scale_init=1.0,
        drop_probability=0.0, feature_drop=0.0,
        visible_conditioning=spec(token_dim=5, hidden_dim=8) if film else None,
    )


def head_inputs():
    raw, tokens, vmask = visible()
    b, n = vmask.shape[:2]
    x_mask = torch.cat((vmask, torch.ones(b, 2, 1, dtype=torch.bool)), dim=1)
    time_masking = torch.cat((torch.zeros(b, n, 1), torch.ones(b, 2, 1)), dim=1)
    return dict(
        x=torch.randn(b, n + 2, 8), global_cond=torch.randn(b, 1, 8), global_cond_mask=torch.ones(b, 1, 1),
        num_x=None, x_mask=x_mask.float(), time=torch.rand(b), label=torch.zeros(b, 1, dtype=torch.long),
        time_masking=time_masking,
    ), dict(visible_raw=raw, visible_tokens=tokens, visible_mask=vmask)


def test_generation_head_identity_at_init_and_effect_when_trained():
    plain, film = make_head(False).eval(), make_head(True).eval()
    missing, unexpected = film.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("visible_conditioning.") for k in missing)
    common, extra = head_inputs()
    torch.testing.assert_close(film(**common, **extra), plain(**common))   # exact identity at init
    with pytest.raises(ValueError):
        film(**common)                                                      # FiLM inputs are required
    open_heads(film.visible_conditioning)
    out = film(**common, **extra)
    n_vis = extra["visible_mask"].shape[1]
    assert not torch.allclose(out[:, n_vis:], plain(**common)[:, n_vis:])


def test_spec_parsing():
    cfg = lambda **vc: SimpleNamespace(VisibleConditioning=vc)
    kwargs = dict(feature_names=NAMES, token_dim=5, hidden_dim=8, num_layers=2, global_cond_dim=4)
    assert visible_conditioning_spec(SimpleNamespace(), **kwargs) is None
    assert visible_conditioning_spec(cfg(diffusion_enabled=False), **kwargs) is None
    # The epoch-287 FiLM network block is accepted as-is.
    epoch287 = dict(diffusion_enabled=True, classifier_enabled=False, diffusion_fourier_enabled=False,
                    diffusion_film_scale_enabled=True, width=64, harmonics=[1, 2, 3, 4],
                    theta_source="Part_eta", phi_source="Part_phi", feature_mode="kinematics",
                    numerical_features=["Part_energy", "Part_pt"], numerical_frequencies=[0.25, 0.5, 1.0, 2.0],
                    log_diagnostics=True,
                    diffusion_film_gains=dict(attention_gamma=1.0, attention_beta=1.0, ffn_gamma=1.0, ffn_beta=1.0))
    parsed = visible_conditioning_spec(cfg(**epoch287), **kwargs)
    assert parsed["global_cond_dim"] == 0 and parsed["fourier_enabled"] is False
    VisibleConditioning(**parsed)
    assert visible_conditioning_spec(cfg(diffusion_enabled=True, include_global_conditions=True), **kwargs)["global_cond_dim"] == 4
    for bad in (dict(diffusion_token_readout=dict(enabled=True)), dict(classifier_enabled=True),
                dict(typo_option=1), dict(diffusion_film_gains=dict(ffn_gamma=-1.0)), dict(feature_mode="x")):
        with pytest.raises(ValueError):
            visible_conditioning_spec(cfg(diffusion_enabled=True, **bad), **kwargs)
