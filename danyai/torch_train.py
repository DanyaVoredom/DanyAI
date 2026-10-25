import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


class ParamDict(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._keys = []
        self._list = torch.nn.ParameterList()

    def __setitem__(self, key, value):
        if key in self._keys:
            raise KeyError("duplicate parameter: " + key)
        self._keys.append(key)
        self._list.append(value)

    def __getitem__(self, key):
        return self._list[self._keys.index(key)]

    def __contains__(self, key):
        return key in self._keys

    def keys(self):
        return list(self._keys)

    def items(self):
        return list(zip(self._keys, list(self._list)))

    def values(self):
        return list(self._list)


class GPT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        V = int(cfg["vocab_size"])
        D = int(cfg["n_embd"])
        L = int(cfg["n_layer"])
        H = int(cfg["n_head"])
        hd = D // H
        self.norm_kind = cfg.get("norm", "ln")
        self.ffn_kind = cfg.get("ffn", "gelu")
        self.rope = bool(cfg.get("rope", False)) and hd % 2 == 0
        kv = int(cfg.get("n_kv_head", H))
        if kv < 1 or H % kv:
            kv = H
        self.n_kv = kv
        self.n_layer = L
        self.n_head = H
        self.n_embd = D
        self.hd = hd
        self.rope_base = float(cfg.get("rope_base", 10000.0))
        self.block = int(cfg["block_size"])

        g = torch.Generator().manual_seed(int(cfg.get("seed", 1337)))
        self.params = ParamDict()

        def w(name, *shape, scale=0.02):
            t = torch.empty(*shape).normal_(0.0, scale, generator=g)
            self.params[name] = torch.nn.Parameter(t)

        def ones(name, n):
            self.params[name] = torch.nn.Parameter(torch.ones(n))

        def zeros(name, n):
            self.params[name] = torch.nn.Parameter(torch.zeros(n))

        w("wte", V, D)
        if not self.rope:
            w("wpe", self.block, D, scale=0.01)
        for i in range(L):
            ones(f"h{i}.ln1.g", D)
            if self.norm_kind != "rms":
                zeros(f"h{i}.ln1.b", D)
            w(f"h{i}.wq", D, D)
            w(f"h{i}.wk", D, kv * hd)
            w(f"h{i}.wv", D, kv * hd)
            w(f"h{i}.wo", D, D, scale=0.02 / math.sqrt(2 * L))
            ones(f"h{i}.ln2.g", D)
            if self.norm_kind != "rms":
                zeros(f"h{i}.ln2.b", D)
            if self.ffn_kind == "swiglu":
                w(f"h{i}.w1", D, 4 * D)
                w(f"h{i}.w3", D, 4 * D)
            else:
                w(f"h{i}.w1", D, 4 * D)
                zeros(f"h{i}.b1", 4 * D)
            w(f"h{i}.w2", 4 * D, D, scale=0.02 / math.sqrt(2 * L))
            zeros(f"h{i}.b2", D)
        ones("lnf.g", D)
        if self.norm_kind != "rms":
            zeros("lnf.b", D)

        half = hd // 2
        inv = self.rope_base ** (-np.arange(half, dtype=np.float64) * 2.0 / hd)
        ang = np.arange(self.block, dtype=np.float64)[:, None] * inv[None, :]
        self.register_buffer("rope_cos", torch.tensor(np.cos(ang), dtype=torch.float32), persistent=False)
        self.register_buffer("rope_sin", torch.tensor(np.sin(ang), dtype=torch.float32), persistent=False)
        mask = torch.triu(torch.full((self.block, self.block), -1e9), 1)
        self.register_buffer("causal", mask, persistent=False)

    def norm(self, x, name):
        g = self.params[name + ".g"]
        if self.norm_kind == "rms":
            xf = x.float()
            rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
            return (xf * rstd).to(x.dtype) * g
        b = self.params[name + ".b"]
        xf = x.float()
        mu = xf.mean(-1, keepdim=True)
        xc = xf - mu
        rstd = torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + 1e-5)
        return ((xc * rstd).to(x.dtype) * g + b)

    def attn(self, x, i, Tn):
        P = self.params
        B = x.shape[0]
        H, hd, kv = self.n_head, self.hd, self.n_kv
        q = (x @ P[f"h{i}.wq"]).view(B, Tn, H, hd).transpose(1, 2)
        k = (x @ P[f"h{i}.wk"]).view(B, Tn, kv, hd).transpose(1, 2)
        v = (x @ P[f"h{i}.wv"]).view(B, Tn, kv, hd).transpose(1, 2)
        if self.rope:
            cos = self.rope_cos[:Tn].to(x.device)
            sin = self.rope_sin[:Tn].to(x.device)
            q = self.rotate(q, cos, sin)
            k = self.rotate(k, cos, sin)
        if kv != H:
            k = k.repeat_interleave(H // kv, dim=1)
            v = v.repeat_interleave(H // kv, dim=1)
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(hd))
        att = att + self.causal[:Tn, :Tn].to(att.dtype)
        p = torch.softmax(att.float(), dim=-1).to(att.dtype)
        o = (p @ v).transpose(1, 2).reshape(B, Tn, self.n_embd)
        return o @ P[f"h{i}.wo"]

    @staticmethod
    def rotate(x, cos, sin):
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def mlp(self, x, i):
        P = self.params
        if self.ffn_kind == "swiglu":
            a = x @ P[f"h{i}.w1"]
            b = x @ P[f"h{i}.w3"]
            return (F.silu(a) * b) @ P[f"h{i}.w2"] + P[f"h{i}.b2"]
        h = x @ P[f"h{i}.w1"] + P[f"h{i}.b1"]
        h = 0.5 * h * (1.0 + torch.tanh(0.7978845608 * (h + 0.044715 * h * h * h)))
        return h @ P[f"h{i}.w2"] + P[f"h{i}.b2"]

    def forward(self, idx):
        P = self.params
        B, Tn = idx.shape
        x = P["wte"][idx]
        if not self.rope:
            x = x + P["wpe"][:Tn]
        for i in range(self.n_layer):
            x = x + self.attn(self.norm(x, f"h{i}.ln1"), i, Tn)
            x = x + self.mlp(self.norm(x, f"h{i}.ln2"), i)
        x = self.norm(x, "lnf")
        return x @ P["wte"].T

    def loss(self, idx, targets):
        logits = self.forward(idx)
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               targets.view(-1), ignore_index=-100)

    def state_numpy(self):
        return {k: v.detach().float().cpu().numpy() for k, v in self.params.items()}

    def load_numpy(self, state):
        with torch.no_grad():
            for k, v in self.params.items():
                if k not in state:
                    raise KeyError("missing parameter in checkpoint: " + k)
                src = torch.from_numpy(np.ascontiguousarray(state[k])).float()
                if tuple(src.shape) != tuple(v.shape):
                    raise ValueError(f"shape mismatch for {k}: {tuple(src.shape)} vs {tuple(v.shape)}")
                v.copy_(src.to(v.device))

    def num_params(self):
        return int(sum(v.numel() for v in self.params.values()))


def get_batch(data, block, batch, rng):
    ix = rng.integers(0, len(data) - block - 1, size=batch)
    x = np.stack([data[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + block + 1] for i in ix]).astype(np.int64)
    return x, y


def lr_at(step, steps, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def build_optimizer(model, lr, wd, betas=(0.9, 0.95)):
    decay, nodecay = [], []
    for name, p in model.params.items():
        (decay if p.ndim >= 2 else nodecay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": nodecay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=fused)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


def save(model, cfg, path, step=0, best=float("inf")):
    state = model.state_numpy()
    np.savez(path, cfg=np.array(json.dumps(cfg)), step=np.array(step),
             best=np.array(best), **state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/nano")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--amp", type=int, default=1)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(args.data, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg["vocab_size"] = meta["vocab_size"]
    dt = DTYPES.get(meta.get("dtype", "uint16"), np.uint16)

    if args.block:
        cfg["block_size"] = args.block
    if args.batch:
        cfg["batch_size"] = args.batch
    if args.lr:
        cfg["lr"] = args.lr

    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=dt, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=dt, mode="r")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = GPT(cfg).to(device)
    start = 0
    best = float("inf")
    if args.resume and os.path.exists(args.resume):
        z = np.load(args.resume, allow_pickle=False)
        model.load_numpy({k: z[k] for k in z.files if k not in ("cfg", "step", "best")})
        start = int(z["step"]) if "step" in z.files else 0
        if "best" in z.files:
            best = float(z["best"])
        z.close()
    print(f"{cfg.get('name', 'DanyAI')} params {model.num_params():,} start {start} device {device}", flush=True)

    opt = build_optimizer(model, cfg["lr"], cfg.get("weight_decay", 0.1))
    rng = np.random.default_rng(cfg.get("seed", 1337) + args.seed)
    steps = args.steps or cfg["steps"]
    warmup = args.warmup or cfg.get("warmup", max(10, steps // 100))
    clip = cfg.get("grad_clip", 1.0)
    os.makedirs(args.out, exist_ok=True)

    use_amp = bool(args.amp) and device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
    t0 = time.time()
    pad_id = 0

    for step in range(start + 1, steps + 1):
        lr = lr_at(step, steps, cfg["lr"], warmup)
        for grp in opt.param_groups:
            grp["lr"] = lr
        x, y = get_batch(train, cfg["block_size"], cfg["batch_size"], rng)
        xb = torch.from_numpy(x).to(device, non_blocking=True)
        yb = torch.from_numpy(y).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast:
            loss = model.loss(xb, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()), clip))
        scaler.step(opt)
        scaler.update()

        if step % cfg.get("log_every", 10) == 0:
            el = time.time() - t0
            print(f"step {step} loss {float(loss.detach()):.4f} lr {lr:.2e} gn {gn:.2f} {el:.0f}s", flush=True)
        if step % cfg.get("eval_every", 500) == 0 or step == steps:
            xv, yv = get_batch(val, cfg["block_size"], cfg["batch_size"], rng)
            xb = torch.from_numpy(xv).to(device, non_blocking=True)
            yb = torch.from_numpy(yv).to(device, non_blocking=True)
            model.eval()
            with torch.no_grad(), autocast:
                vl = float(model.loss(xb, yb))
            model.train()
            ppl = math.exp(min(20.0, vl))
            flag = ""
            if vl < best:
                best = vl
                save(model, cfg, os.path.join(args.out, "best.npz"), step, best)
                flag = " best"
            print(f"eval step {step} val {vl:.4f} ppl {ppl:.2f}{flag}", flush=True)
        if step % cfg.get("save_every", 500) == 0 or step == steps:
            save(model, cfg, os.path.join(args.out, "ckpt.npz"), step, best)
            save(model, cfg, os.path.join(args.out, f"ckpt_{step}.npz"), step, best)

    print(f"done {time.time() - t0:.0f}s best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
