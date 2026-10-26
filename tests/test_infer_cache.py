import numpy as np
import pytest

from danyai.infer import Engine
from danyai.model import GPT

CFG = {
    "name": "tiny",
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "block_size": 16,
    "vocab_size": 64,
    "norm": "rms",
    "ffn": "swiglu",
    "rope": True,
    "n_kv_head": 1,
    "rope_base": 10000.0,
}


@pytest.fixture()
def engine():
    model = GPT(CFG)
    return Engine(CFG, model.state())


def test_fused_matches_plain(engine):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 64, size=(2, 8)).astype(np.int64)
    fused = engine.forward(ids)
    qkv, gu = engine._qkv, engine._gu
    engine._qkv, engine._gu = {}, {}
    try:
        plain = engine.forward(ids)
    finally:
        engine._qkv, engine._gu = qkv, gu
    assert np.allclose(fused, plain, atol=1e-5)


def test_cache_matches_full_forward(engine):
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 64, size=12).astype(np.int64)
    full = engine.forward(ids[None, :], None, logits_last=True)[0, -1]

    cache = engine.new_cache(1)
    engine.forward(ids[:8][None, :], cache, logits_last=True)
    assert cache["pos"] == 8
    inc = engine.forward(ids[8:12][None, :], cache, logits_last=True)[0, -1]
    assert np.allclose(full, inc, atol=1e-5)


def test_sliding_window_matches_same_window(engine):
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 64, size=40).astype(np.int64)
    cache = engine.new_cache(1)
    engine.forward(ids[:15][None, :], cache, logits_last=True)
    for t in ids[15:]:
        engine.forward(np.asarray([[t]]), cache, logits_last=True)
        assert cache["pos"] <= engine.block

    assert cache["pos"] == engine.block
    assert cache["base"] == 40 - engine.block

    window = ids[-engine.block:]
    fresh = engine.new_cache(1)
    fresh["base"] = cache["base"]
    engine.forward(window[None, :], fresh, logits_last=True)
    assert np.allclose(cache["k"][0], fresh["k"][0], atol=1e-5)
    assert np.allclose(cache["v"][0], fresh["v"][0], atol=1e-5)
    for i in range(1, engine.n_layer):
        assert cache["k"][i].shape == (1, engine.n_head, engine.block, engine.hd)
        assert np.isfinite(cache["k"][i]).all()


def test_logits_last_shape(engine):
    rng = np.random.default_rng(3)
    ids = rng.integers(0, 64, size=(1, 9)).astype(np.int64)
    out = engine.forward(ids, None, logits_last=True)
    assert out.shape == (1, 1, CFG["vocab_size"])
