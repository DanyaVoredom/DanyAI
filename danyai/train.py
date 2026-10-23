import argparse
import json
import math
import os
import time
import numpy as np
from .model import GPT
from .tensor import backward

DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


class AdamW:
    def __init__(self, named, lr, betas=(0.9, 0.95), eps=1e-8, wd=0.1):
        self.named = named
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = wd
        self.t = 0
        self.m = [np.zeros_like(p.data) for _, p in named]
        self.v = [np.zeros_like(p.data) for _, p in named]

    def zero(self):
        for _, p in self.named:
            p.grad = None

    def clip(self, max_norm):
        total = 0.0
        for _, p in self.named:
            if p.grad is not None:
                total += float((p.grad.astype(np.float64) ** 2).sum())
        total = math.sqrt(total)
        if total > max_norm:
            s = max_norm / (total + 1e-6)
            for _, p in self.named:
                if p.grad is not None:
                    p.grad *= s
        return total

    def step(self, lr):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k, (_, p) in enumerate(self.named):
            if p.grad is None:
                continue
            g = p.grad
            self.m[k] *= self.b1
            self.m[k] += (1 - self.b1) * g
            self.v[k] *= self.b2
            self.v[k] += (1 - self.b2) * g * g
            if p.data.ndim >= 2:
                p.data -= lr * self.wd * p.data
            p.data -= (lr / c1) * self.m[k] / (np.sqrt(self.v[k] / c2) + self.eps)


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


def save(model, cfg, path, step=0, best=float("inf")):
    np.savez(path, cfg=np.array(json.dumps(cfg)), step=np.array(step), best=np.array(best), **model.state())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/nano")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(args.data, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg["vocab_size"] = meta["vocab_size"]
    dt = DTYPES.get(meta.get("dtype", "uint16"), np.uint16)

    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=dt, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=dt, mode="r")

    model = GPT(cfg)
    start = 0
    best = float("inf")
    if args.resume and os.path.exists(args.resume):
        z = np.load(args.resume, allow_pickle=False)
        model.load_state({k: z[k] for k in z.files if k not in ("cfg", "step", "best")})
        start = int(z["step"]) if "step" in z.files else 0
        if "best" in z.files:
            best = float(z["best"])
    print(f"{cfg.get('name', 'DanyAI')} params {model.num_params():,} start {start}", flush=True)

    opt = AdamW(model.named_params(), cfg["lr"], wd=cfg.get("weight_decay", 0.1))
    rng = np.random.default_rng(cfg.get("seed", 1337) + args.seed)
    steps = args.steps or cfg["steps"]
    warmup = cfg.get("warmup", max(10, steps // 100))
    clip = cfg.get("grad_clip", 1.0)
    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    t0 = time.time()

    for step in range(start + 1, steps + 1):
        lr = lr_at(step, steps, cfg["lr"], warmup)
        x, y = get_batch(train, cfg["block_size"], cfg["batch_size"], rng)
        loss = model.loss(x, y)
        opt.zero()
        backward(loss)
        gn = opt.clip(clip)
        opt.step(lr)

        if step % cfg.get("log_every", 10) == 0:
            el = time.time() - t0
            print(f"step {step} loss {float(loss.data):.4f} lr {lr:.2e} gn {gn:.2f} {el:.0f}s", flush=True)
        if step % cfg.get("eval_every", 200) == 0:
            xv, yv = get_batch(val, cfg["block_size"], cfg["batch_size"], rng)
            vl = float(model.loss(xv, yv).data)
            flag = ""
            if vl < best:
                best = vl
                save(model, cfg, os.path.join(args.out, "best.npz"), step, best)
                flag = " best"
            print(f"eval step {step} val {vl:.4f}{flag}", flush=True)
        if step % cfg.get("save_every", 500) == 0 or step == steps:
            save(model, cfg, os.path.join(args.out, "ckpt.npz"), step, best)
            save(model, cfg, os.path.join(args.out, f"ckpt_{step}.npz"), step, best)

    print(f"done {time.time() - t0:.0f}s best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()