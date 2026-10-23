import math
import numpy as np
from .tensor import (
    T, add, mul, matmul, gelu, layernorm, softmax,
    embedding, cross_entropy, reshape, transpose,
)


class GPT:
    def __init__(self, cfg):
        self.cfg = cfg
        V = cfg["vocab_size"]
        D = cfg["n_embd"]
        L = cfg["n_layer"]
        rng = np.random.default_rng(cfg.get("seed", 1337))
        P = self.params = {}

        def w(*shape, scale=0.02):
            return T(rng.normal(0.0, scale, shape).astype(np.float32), requires_grad=True)

        def ones(n):
            return T(np.ones(n, dtype=np.float32), requires_grad=True)

        def zeros(n):
            return T(np.zeros(n, dtype=np.float32), requires_grad=True)

        P["wte"] = w(V, D)
        P["wpe"] = w(cfg["block_size"], D, scale=0.01)
        for i in range(L):
            P[f"h{i}.ln1.g"] = ones(D)
            P[f"h{i}.ln1.b"] = zeros(D)
            P[f"h{i}.wq"] = w(D, D)
            P[f"h{i}.wk"] = w(D, D)
            P[f"h{i}.wv"] = w(D, D)
            P[f"h{i}.wo"] = w(D, D, scale=0.02 / math.sqrt(2 * L))
            P[f"h{i}.ln2.g"] = ones(D)
            P[f"h{i}.ln2.b"] = zeros(D)
            P[f"h{i}.w1"] = w(D, 4 * D)
            P[f"h{i}.b1"] = zeros(4 * D)
            P[f"h{i}.w2"] = w(4 * D, D, scale=0.02 / math.sqrt(2 * L))
            P[f"h{i}.b2"] = zeros(D)
        P["lnf.g"] = ones(D)
        P["lnf.b"] = zeros(D)
        self._mask = None

    def mask(self, Tn):
        if self._mask is None or self._mask.shape[0] < Tn:
            m = np.triu(np.full((Tn, Tn), -1e9, dtype=np.float32), 1)
            self._mask = m
        return self._mask[:Tn, :Tn]

    def attn(self, x, i, mask, B, Tn):
        D = self.cfg["n_embd"]
        H = self.cfg["n_head"]
        hd = D // H
        P = self.params
        q = transpose(reshape(matmul(x, P[f"h{i}.wq"]), (B, Tn, H, hd)), (0, 2, 1, 3))
        k = transpose(reshape(matmul(x, P[f"h{i}.wk"]), (B, Tn, H, hd)), (0, 2, 1, 3))
        v = transpose(reshape(matmul(x, P[f"h{i}.wv"]), (B, Tn, H, hd)), (0, 2, 1, 3))
        att = mul(matmul(q, k, tb=True), 1.0 / math.sqrt(hd))
        att = add(att, mask)
        p = softmax(att)
        o = matmul(p, v)
        o = reshape(transpose(o, (0, 2, 1, 3)), (B, Tn, D))
        return matmul(o, P[f"h{i}.wo"])

    def mlp(self, x, i):
        P = self.params
        h = add(matmul(x, P[f"h{i}.w1"]), P[f"h{i}.b1"])
        return add(matmul(gelu(h), P[f"h{i}.w2"]), P[f"h{i}.b2"])

    def forward(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        B, Tn = idx.shape
        P = self.params
        x = add(embedding(P["wte"], idx), embedding(P["wpe"], np.arange(Tn)))
        mask = self.mask(Tn)
        for i in range(self.cfg["n_layer"]):
            x = add(x, self.attn(layernorm(x, P[f"h{i}.ln1.g"], P[f"h{i}.ln1.b"]), i, mask, B, Tn))
            x = add(x, self.mlp(layernorm(x, P[f"h{i}.ln2.g"], P[f"h{i}.ln2.b"]), i))
        x = layernorm(x, P["lnf.g"], P["lnf.b"])
        return matmul(x, P["wte"], tb=True)

    def loss(self, idx, targets):
        return cross_entropy(self.forward(idx), targets)

    def named_params(self):
        return list(self.params.items())

    def state(self):
        return {k: v.data.copy() for k, v in self.params.items()}

    def load_state(self, state):
        for k, v in self.params.items():
            v.data[...] = state[k]

    def num_params(self):
        return int(sum(v.data.size for v in self.params.values()))