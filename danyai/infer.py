import json
import numpy as np


def _gelu(x):
    c = np.float32(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def apply_penalties(logits, ids, repetition=1.0, presence=0.0, frequency=0.0, window=256):
    lg = np.array(logits, dtype=np.float64, copy=True)
    if not ids:
        return lg
    hist = list(ids)[-window:] if window and window > 0 else list(ids)
    counts = {}
    for t in hist:
        t = int(t)
        counts[t] = counts.get(t, 0) + 1
    idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    cnt = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    idx = np.clip(idx, 0, lg.size - 1)
    if repetition and repetition != 1.0:
        v = lg[idx]
        lg[idx] = np.where(v > 0, v / repetition, v * repetition)
    if presence:
        lg[idx] -= presence
    if frequency:
        lg[idx] -= frequency * cnt
    return lg


def sample_token(logits, rng, temp=0.8, top_k=40, top_p=1.0):
    lg = np.asarray(logits, dtype=np.float64)
    if temp is not None and temp <= 0:
        return int(lg.argmax())
    if temp != 1.0:
        lg = lg / temp
    if top_k and 0 < top_k < lg.size:
        thr = np.partition(lg, -top_k)[-top_k]
        lg = np.where(lg < thr, -np.inf, lg)
    if top_p and 0 < top_p < 1.0:
        order = np.argsort(-lg)
        sorted_lg = lg[order]
        e = np.exp(sorted_lg - sorted_lg[0])
        cdf = np.cumsum(e / e.sum())
        cut = int(np.searchsorted(cdf, top_p) + 1)
        keep = order[:cut]
        masked = np.full_like(lg, -np.inf)
        masked[keep] = lg[keep]
        lg = masked
    m = lg.max()
    p = np.exp(lg - m)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return int(np.argmax(logits))
    p /= s
    return int(rng.choice(p.size, p=p))


class Engine:
    def __init__(self, cfg, params):
        self.cfg = cfg
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        self.n_layer = int(cfg["n_layer"])
        self.n_head = int(cfg["n_head"])
        self.n_embd = int(cfg["n_embd"])
        self.hd = self.n_embd // self.n_head
        self.n_kv = int(cfg.get("n_kv_head", self.n_head))
        if self.n_kv < 1 or self.n_head % self.n_kv:
            self.n_kv = self.n_head
        self.norm_kind = cfg.get("norm", "ln")
        self.ffn_kind = cfg.get("ffn", "gelu")
        self.rope = bool(cfg.get("rope", False)) and self.hd % 2 == 0
        self.rope_base = float(cfg.get("rope_base", 10000.0))
        self.block = int(cfg["block_size"])
        self.vocab = int(cfg["vocab_size"])
        self.quant = bool(cfg.get("quantized", False))
        self._rope = None
        self._bias = None

    @classmethod
    def from_file(cls, path):
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["cfg"]))
        qcfg = {}
        if "quant" in z.files:
            qcfg = json.loads(str(z["quant"]))
        params = {}
        for k in z.files:
            if k in ("cfg", "quant", "step", "best"):
                continue
            if k.endswith(".scale"):
                continue
            arr = z[k]
            if qcfg and k in qcfg:
                params[k] = arr.astype(np.float32) * np.float32(qcfg[k])
            else:
                params[k] = arr.astype(np.float32)
        if qcfg:
            cfg = dict(cfg)
            cfg["quantized"] = True
        return cls(cfg, params), cfg

    def new_cache(self, batch=1):
        return {"k": [None] * self.n_layer, "v": [None] * self.n_layer, "pos": 0}

    def _norm(self, x, name):
        g = self.p[name + ".g"]
        if self.norm_kind == "rms":
            return x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(1e-5)) * g
        b = self.p[name + ".b"]
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        rstd = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + np.float32(1e-5))
        return xc * rstd * g + b

    def _tables(self, T):
        if self._rope is not None and self._rope[0].shape[0] >= T:
            return self._rope
        half = self.hd // 2
        inv = self.rope_base ** (-np.arange(half, dtype=np.float64) * 2.0 / self.hd)
        ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
        self._rope = (np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32))
        return self._rope

    @staticmethod
    def _rotate(x, cos, sin):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def _bias_for(self, Tq, Tk):
        key = (Tq, Tk)
        if self._bias is None or self._bias[0] != key:
            q = np.arange(Tq)[:, None] + (Tk - Tq)
            k = np.arange(Tk)[None, :]
            b = np.where(k <= q, np.float32(0.0), np.float32(-1e9)).astype(np.float32)
            self._bias = (key, b)
        return self._bias[1]

    def _attn(self, x, i, cache, B, Tn):
        P = self.p
        H = self.n_head
        kv = self.n_kv
        hd = self.hd
        q = (x @ P[f"h{i}.wq"]).reshape(B, Tn, H, hd).transpose(0, 2, 1, 3)
        k = (x @ P[f"h{i}.wk"]).reshape(B, Tn, kv, hd).transpose(0, 2, 1, 3)
        v = (x @ P[f"h{i}.wv"]).reshape(B, Tn, kv, hd).transpose(0, 2, 1, 3)
        if self.rope:
            off = cache["pos"] if cache is not None else 0
            cos, sin = self._tables(off + Tn)
            cos = cos[off:off + Tn][None, None, :, :]
            sin = sin[off:off + Tn][None, None, :, :]
            q = self._rotate(q, cos, sin)
            k = self._rotate(k, cos, sin)
        if cache is not None:
            if cache["k"][i] is not None:
                k = np.concatenate([cache["k"][i], k], axis=2)
                v = np.concatenate([cache["v"][i], v], axis=2)
            cache["k"][i] = k
            cache["v"][i] = v
        if kv != H:
            k = np.repeat(k, H // kv, axis=1)
            v = np.repeat(v, H // kv, axis=1)
        att = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(hd))
        att = att + self._bias_for(Tn, k.shape[2])
        att = att - att.max(-1, keepdims=True)
        e = np.exp(att)
        e /= e.sum(-1, keepdims=True)
        o = (e @ v).transpose(0, 2, 1, 3).reshape(B, Tn, self.n_embd)
        return o @ P[f"h{i}.wo"]

    def _mlp(self, x, i):
        P = self.p
        if self.ffn_kind == "swiglu":
            a = x @ P[f"h{i}.w1"]
            b = x @ P[f"h{i}.w3"]
            h = _silu(a) * b
        else:
            h = _gelu(x @ P[f"h{i}.w1"] + P[f"h{i}.b1"])
        return h @ P[f"h{i}.w2"] + P[f"h{i}.b2"]

    def forward(self, idx, cache=None):
        idx = np.asarray(idx, dtype=np.int64)
        B, Tn = idx.shape
        P = self.p
        if self.rope:
            x = P["wte"][idx]
        else:
            off = cache["pos"] if cache is not None else 0
            x = P["wte"][idx] + P["wpe"][off:off + Tn]
        for i in range(self.n_layer):
            x = x + self._attn(self._norm(x, f"h{i}.ln1"), i, cache, B, Tn)
            x = x + self._mlp(self._norm(x, f"h{i}.ln2"), i)
        x = self._norm(x, "lnf")
        logits = x @ P["wte"].T
        if cache is not None:
            cache["pos"] += Tn
        return logits

    def next_logits(self, ids, cache):
        return self.forward(np.asarray([ids], dtype=np.int64), cache)[0, -1]

    def generate(self, tok, prompt, tokens=200, temp=0.8, top_k=40, top_p=1.0,
                 repetition=1.0, presence=0.0, frequency=0.0, seed=0,
                 on_token=None, stop=None):
        rng = np.random.default_rng(seed)
        ids = tok.encode(prompt) if isinstance(prompt, str) else [int(t) for t in prompt]
        if not ids:
            ids = [10]
        cache = self.new_cache(1)
        head = ids[-self.block:-1]
        self.forward(np.asarray([head], dtype=np.int64), cache)
        out = bytearray()
        prev = ""
        for _ in range(int(tokens)):
            if cache["pos"] >= self.block:
                cache = self.new_cache(1)
                self.forward(np.asarray([ids[-self.block:-1]], dtype=np.int64), cache)
            lg = self.next_logits([ids[-1]], cache)
            lg = apply_penalties(lg, ids, repetition, presence, frequency)
            nxt = sample_token(lg, rng, temp, top_k, top_p)
            ids.append(nxt)
            piece = tok.vocab.get(int(nxt))
            if piece:
                out.extend(piece)
                cur = out.decode("utf-8", errors="replace")
                if cur.startswith(prev):
                    delta = cur[len(prev):]
                    prev = cur
                    if delta and on_token is not None:
                        on_token(delta, nxt)
            if stop and any(s in prev for s in stop):
                break
        return tok.decode(ids)

    def stream(self, tok, prompt, **kw):
        buf = []
        kw = dict(kw)
        kw["on_token"] = lambda delta, _n: buf.append(delta)
        text = self.generate(tok, prompt, **kw)
        for d in buf:
            yield d

    @staticmethod
    def _ce(logits, targets):
        x = logits - logits.max(-1, keepdims=True)
        lse = np.log(np.exp(x).sum(-1))
        flat_x = x.reshape(-1, x.shape[-1])
        y = targets.reshape(-1)
        rows = np.arange(y.size)
        return float((-flat_x[rows, y] + lse.reshape(-1)).mean())

    def perplexity(self, data, block=None, batch=8, limit=None):
        block = int(block or self.block)
        data = np.asarray(data)
        n = data.size
        pos = 0
        tot = 0.0
        seen = 0
        while pos + block + 1 <= n and (limit is None or seen < limit):
            rows = []
            while len(rows) < batch and pos + block + 1 <= n:
                rows.append(pos)
                pos += block
            x = np.stack([data[a:a + block] for a in rows]).astype(np.int64)
            y = np.stack([data[a + 1:a + block + 1] for a in rows]).astype(np.int64)
            logits = self.forward(x)
            tot += self._ce(logits, y) * y.size
            seen += y.size
        if seen == 0:
            return float("nan")
        return float(np.exp(tot / seen))