import math
import numpy as np
from .backend import xp
from .tensor import (
    T, add, mul, matmul, gelu, silu, rope, repeat_heads,
    layernorm, rmsnorm, softmax, embedding, cross_entropy, reshape, transpose,
)


class GPT:
    def __init__(self, cfg):
        self.cfg = cfg
        V = cfg["vocab_size"]
        D = cfg["n_embd"]
        L = cfg["n_layer"]
        H = cfg["n_head"]
        rng = np.random.default_rng(cfg.get("seed", 1337))
        P = self.params = {}
        self.norm_kind = cfg.get("norm", "ln")
        self.ffn_kind = cfg.get("ffn", "gelu")
        self.rope = bool(cfg.get("rope", False))
        hd = D // H
        if self.rope and hd % 2:
            self.rope = False
        kv = int(cfg.get("n_kv_head", H))
        if kv < 1 or H % kv:
            kv = H
        self.n_kv = kv
        if self.n_kv != H and D % H:
            self.n_kv = H
        self._mask = None
        self._rope = None

        def w(*shape, scale=0.02):
            return T(rng.normal(0.0, scale, shape).astype(np.float32), requires_grad=True)

        def ones(n):
            return T(np.ones(n, dtype=np.float32), requires_grad=True)

        def zeros(n):
            return T(np.zeros(n, dtype=np.float32), requires_grad=True)

        P["wte"] = w(V, D)
        if not self.rope:
            P["wpe"] = w(cfg["block_size"], D, scale=0.01)
        for i in range(L):
            P[f"h{i}.ln1.g"] = ones(D)
            if self.norm_kind != "rms":
                P[f"h{i}.ln1.b"] = zeros(D)
            P[f"h{i}.wq"] = w(D, D)
            P[f"h{i}.wk"] = w(D, kv * hd)
            P[f"h{i}.wv"] = w(D, kv * hd)
            P[f"h{i}.wo"] = w(D, D, scale=0.02 / math.sqrt(2 * L))
            P[f"h{i}.ln2.g"] = ones(D)
            if self.norm_kind != "rms":
                P[f"h{i}.ln2.b"] = zeros(D)
            if self.ffn_kind == "swiglu":
                P[f"h{i}.w1"] = w(D, 4 * D)
                P[f"h{i}.w3"] = w(D, 4 * D)
            else:
                P[f"h{i}.w1"] = w(D, 4 * D)
                P[f"h{i}.b1"] = zeros(4 * D)
            P[f"h{i}.w2"] = w(4 * D, D, scale=0.02 / math.sqrt(2 * L))
            P[f"h{i}.b2"] = zeros(D)
        P["lnf.g"] = ones(D)
        if self.norm_kind != "rms":
            P["lnf.b"] = zeros(D)

    def mask(self, Tn):
        if self._mask is None or self._mask.shape[0] < Tn:
            m = xp.triu(xp.full((Tn, Tn), -1e9, dtype=xp.float32), 1)
            self._mask = m
        return self._mask[:Tn, :Tn]

    def rope_table(self, Tn):
        if self._rope is not None and self._rope.shape[0] >= Tn:
            return self._rope[:Tn]
        D = self.cfg["n_embd"]
        H = self.cfg["n_head"]
        hd = D // H
        half = hd // 2
        base = float(self.cfg.get("rope_base", 10000.0))
        inv = base ** (-xp.arange(half, dtype=xp.float64) * 2.0 / hd)
        pos = xp.arange(Tn, dtype=xp.float64)[:, None]
        ang = pos * inv[None, :]
        self._rope = xp.stack([xp.cos(ang), xp.sin(ang)], axis=0).astype(xp.float32)
        return self._rope

    def norm(self, x, name):
        P = self.params
        if self.norm_kind == "rms":
            return rmsnorm(x, P[name + ".g"])
        return layernorm(x, P[name + ".g"], P[name + ".b"])

    def attn(self, x, i, mask, B, Tn):
        D = self.cfg["n_embd"]
        H = self.cfg["n_head"]
        kv = self.n_kv
        hd = D // H
        P = self.params
        q = transpose(reshape(matmul(x, P[f"h{i}.wq"]), (B, Tn, H, hd)), (0, 2, 1, 3))
        k = transpose(reshape(matmul(x, P[f"h{i}.wk"]), (B, Tn, kv, hd)), (0, 2, 1, 3))
        v = transpose(reshape(matmul(x, P[f"h{i}.wv"]), (B, Tn, kv, hd)), (0, 2, 1, 3))
        if self.rope:
            cos, sin = self.rope_table(Tn)
            q = rope(q, cos, sin)
            k = rope(k, cos, sin)
        if kv != H:
            k = repeat_heads(k, H // kv)
            v = repeat_heads(v, H // kv)
        att = mul(matmul(q, k, tb=True), 1.0 / math.sqrt(hd))
        att = add(att, mask)
        p = softmax(att)
        o = matmul(p, v)
        o = reshape(transpose(o, (0, 2, 1, 3)), (B, Tn, D))
        return matmul(o, P[f"h{i}.wo"])

    def mlp(self, x, i):
        P = self.params
        if self.ffn_kind == "swiglu":
            a = matmul(x, P[f"h{i}.w1"])
            b = matmul(x, P[f"h{i}.w3"])
            return add(matmul(mul(silu(a), b), P[f"h{i}.w2"]), P[f"h{i}.b2"])
        h = add(matmul(x, P[f"h{i}.w1"]), P[f"h{i}.b1"])
        return add(matmul(gelu(h), P[f"h{i}.w2"]), P[f"h{i}.b2"])

    def forward(self, idx):
        idx = xp.asarray(idx, dtype=xp.int64)
        B, Tn = idx.shape
        P = self.params
        if self.rope:
            x = embedding(P["wte"], idx)
        else:
            x = add(embedding(P["wte"], idx), embedding(P["wpe"], np.arange(Tn)))
        mask = self.mask(Tn)
        for i in range(self.cfg["n_layer"]):
            x = add(x, self.attn(self.norm(x, f"h{i}.ln1"), i, mask, B, Tn))
            x = add(x, self.mlp(self.norm(x, f"h{i}.ln2"), i))
        x = self.norm(x, "lnf")
        return matmul(x, P["wte"], tb=True)

    def loss(self, idx, targets):
        return cross_entropy(self.forward(idx), targets)

    def named_params(self):
        return list(self.params.items())

    def state(self):
        return {k: v.data.copy() for k, v in self.params.items()}

    def load_state(self, state):
        for k, v in self.params.items():
            if k not in state:
                raise KeyError(f"missing parameter in checkpoint: {k}")
            src = state[k]
            if src.shape != v.data.shape:
                raise ValueError(f"shape mismatch for {k}: {src.shape} vs {v.data.shape}")
            v.data[...] = xp.asarray(src)

    def num_params(self):
        return int(sum(v.data.size for v in self.params.values()))