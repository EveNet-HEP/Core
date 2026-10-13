"""CPU unit tests for the TruthGeneration FiLM conditioning (network/layers/film.py)."""
import pytest
import torch

from evenet.network.heads.generation.generation_head import EventGenerationHead
from evenet.network.layers.film import FILM_OPTIONS, FiLMConditioning, apply_film, film_kwargs

CFG = dict(enabled=True, hidden_dim=12, num_layers=2, activation="silu", pooling="attention",
           use_multiplicity=True, use_globals=False, modulate_attention=True, modulate_ffn=True, scale=True)


def make_film(global_dim=0, **overrides):
    cfg = {**CFG, **overrides, "use_globals": global_dim > 0}
    return FiLMConditioning(**film_kwargs(cfg, token_dim=5, block_dim=8, num_blocks=2, global_dim=global_dim))


def visible():
    torch.manual_seed(0)
    tokens = torch.randn(3, 4, 5)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)[..., None]
    return tokens, mask


def open_heads(film, std=0.05):
    with torch.no_grad():
        for head in film.heads:
            head.weight.normal_(std=std)
            head.bias.fill_(0.02)


def coefficients(blocks):
    return [t for block in blocks for pair in block.values() for t in pair if t is not None]


@pytest.mark.parametrize("pooling", ["attention", "mean"])
def test_zero_init_masking_and_permutation(pooling):
    film = make_film(pooling=pooling)
    tokens, mask = visible()
    tokens[~mask.expand_as(tokens)] = float("nan")              # padded slots must be ignored
    blocks = film(tokens, mask)
    assert len(blocks) == 2 and all(set(b) == {"attention", "ffn"} for b in blocks)
    assert all(t.count_nonzero() == 0 for t in coefficients(blocks))  # identity at init
    sum(t.sum() for t in coefficients(blocks)).backward()
    assert film.heads[0].weight.grad.norm() > 0                   # heads still learn

    open_heads(film)
    out, flipped = film(tokens, mask), film(tokens.flip(1), mask.flip(1))
    for a, b in zip(coefficients(out), coefficients(flipped)):
        assert torch.isfinite(a).all()
        assert a[1].count_nonzero() == 0                          # empty event -> no modulation
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)    # permutation invariant


def test_sites_and_scale_switches():
    tokens, mask = visible()
    assert set(make_film(modulate_ffn=False)(tokens, mask)[0]) == {"attention"}
    assert set(make_film(modulate_attention=False)(tokens, mask)[0]) == {"ffn"}
    shift_only = make_film(scale=False)
    assert all(gamma is None for block in shift_only(tokens, mask) for gamma, _ in block.values())
    assert shift_only.heads[0].out_features == 2 * 8               # only betas are learned


def test_multiplicity_and_globals_enter_context():
    tokens, mask = visible()
    assert make_film(use_multiplicity=False).event_encoder[0][0].in_features == 12
    film = make_film(global_dim=3); open_heads(film)
    assert film.event_encoder[0][0].in_features == 12 + 1 + 3
    g = torch.randn(3, 1, 3)
    with pytest.raises(ValueError):
        film(tokens, mask)                                         # globals are required when enabled
    a = film(tokens, mask, g)[0]["attention"][1]
    b = film(tokens, mask, g + 1.0)[0]["attention"][1]
    assert not torch.allclose(a, b)
    # A masked-out global condition is equivalent to zeros.
    c = film(tokens, mask, g, torch.zeros(3, 1, 1))[0]["attention"][1]
    torch.testing.assert_close(c, film(tokens, mask, torch.zeros_like(g))[0]["attention"][1])


def test_apply_film_only_on_masked_slots():
    h = torch.randn(2, 5, 8)
    gamma, beta = torch.randn(2, 8), torch.randn(2, 8)
    slots = torch.zeros(2, 5, 1, dtype=torch.bool); slots[:, 3:] = True
    out = apply_film(h, (gamma, beta), slots)
    torch.testing.assert_close(out[:, :3], h[:, :3])
    torch.testing.assert_close(out[:, 3:], h[:, 3:] * (1 + gamma[:, None]) + beta[:, None])
    torch.testing.assert_close(apply_film(h, (None, beta), slots)[:, 3:], h[:, 3:] + beta[:, None])


def make_head(film):
    torch.manual_seed(1)
    kwargs = film_kwargs(CFG, token_dim=5, block_dim=8, num_blocks=2, global_dim=0) if film else None
    return EventGenerationHead(
        input_dim=8, projection_dim=8, num_global_cond=8, num_classes=3, output_dim=2,
        num_layers=2, num_heads=2, dropout=0.0, layer_scale=False, layer_scale_init=1.0,
        drop_probability=0.0, feature_drop=0.0, film=kwargs,
    )


def head_inputs():
    tokens, vmask = visible()
    vmask[1, 0] = True  # every event has a visible particle (the stock head is NaN for an empty event)
    b, n = vmask.shape[:2]
    x_mask = torch.cat((vmask, torch.ones(b, 2, 1, dtype=torch.bool)), dim=1)
    time_masking = torch.cat((torch.zeros(b, n, 1), torch.ones(b, 2, 1)), dim=1)
    # As in EveNetModel: visible queries do not attend to the invisible slots.
    is_invisible = torch.cat((torch.zeros(n, dtype=torch.bool), torch.ones(2, dtype=torch.bool)))
    attn_mask = (~is_invisible[:, None]) & is_invisible[None, :]
    return dict(
        x=torch.randn(b, n + 2, 8), global_cond=torch.randn(b, 1, 8), global_cond_mask=torch.ones(b, 1, 1),
        num_x=None, x_mask=x_mask.float(), time=torch.rand(b), label=torch.zeros(b, 1, dtype=torch.long),
        time_masking=time_masking, attn_mask=attn_mask,
    ), dict(visible_tokens=tokens, visible_mask=vmask)


def test_generation_head_identity_at_init_and_effect_when_trained():
    plain, film = make_head(False).eval(), make_head(True).eval()
    missing, unexpected = film.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("film.") for k in missing)
    common, extra = head_inputs()
    torch.testing.assert_close(film(**common, **extra), plain(**common))   # exact identity at init
    with pytest.raises(ValueError):
        film(**common)                                                      # FiLM inputs are required
    open_heads(film.film)
    out = film(**common, **extra)
    n_vis = extra["visible_mask"].shape[1]
    assert torch.isfinite(out[:, n_vis:]).all()
    torch.testing.assert_close(out[:, :n_vis], plain(**common)[:, :n_vis])  # visible slots untouched
    assert not torch.allclose(out[:, n_vis:], plain(**common)[:, n_vis:])


def test_config_parsing():
    dims = dict(token_dim=5, block_dim=8, num_blocks=2, global_dim=4)
    assert film_kwargs(None, **dims) is None
    assert film_kwargs({**CFG, "enabled": False}, **dims) is None
    assert film_kwargs(CFG, **dims)["global_dim"] == 0
    assert film_kwargs({**CFG, "use_globals": True}, **dims)["global_dim"] == 4
    missing = {k: v for k, v in CFG.items() if k != "pooling"}
    for bad in (missing, {**CFG, "typo": 1}, {**CFG, "scale": 1}):
        with pytest.raises(ValueError):
            film_kwargs(bad, **dims)
    for bad in (dict(pooling="max"), dict(modulate_attention=False, modulate_ffn=False), dict(num_layers=0)):
        with pytest.raises(ValueError):
            make_film(**bad)
    assert set(CFG) == set(FILM_OPTIONS)
