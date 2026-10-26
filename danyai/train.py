import argparse
import json
import math
import os
import time
import numpy as np
from .backend import xp, to_numpy, to_scalar
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
        self.m = [xp.zeros_like(p.data) for _, p in named]
        self.v = [xp.zeros_like(p.data) for _, p in named]

    def zero(self):
        for _, p in self.named:
            p.grad = None

    def clip(self, max_norm):
        total = 0.0
        for _, p in self.named:
            if p.grad is not None:
                total += float(to_numpy((p.grad.astype(xp.float64) ** 2).sum()))
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
            p.data -= (lr / c1) * self.m[k] / (xp.sqrt(self.v[k] / c2) + self.eps)


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


def _layout(named):
    offs = []
    off = 0
    for _, p in named:
        n = p.data.size
        offs.append((p.data.shape, off, n))
        off += n
    return offs, off


def _write_data(dst, named, offs):
    buf = np.frombuffer(dst, dtype=np.float32)
    for (_, p), (shape, off, n) in zip(named, offs):
        buf[off:off + n] = np.asarray(p.data, dtype=np.float32).reshape(-1)


def _read_flat(src, named, offs):
    buf = np.frombuffer(src, dtype=np.float32)
    for (_, p), (shape, off, n) in zip(named, offs):
        p.data[...] = buf[off:off + n].reshape(shape)


def _write_flat(dst, named, offs):
    buf = np.frombuffer(dst, dtype=np.float32)
    for (_, p), (shape, off, n) in zip(named, offs):
        g = p.grad
        if g is None:
            buf[off:off + n] = 0.0
        else:
            buf[off:off + n] = np.asarray(g, dtype=np.float32).reshape(-1)


def _par_loop(rank, w):
    cfg, args = w["cfg"], w["args"]
    model = w.get("model")
    if model is None:
        model = GPT(cfg)
        model.load_state(w["state"])
    opt = AdamW(model.named_params(), cfg["lr"], wd=cfg.get("weight_decay", 0.1))
    train = np.memmap(os.path.join(w["data"], "train.bin"), dtype=w["dtype"], mode="r")
    val = np.memmap(os.path.join(w["data"], "val.bin"), dtype=w["dtype"], mode="r")
    named = model.named_params()
    offs, total = w["offs"], w["total"]
    workers = args.workers
    micro = max(1, cfg["batch_size"] // workers)
    steps, warmup, clip = w["steps"], w["warmup"], w["clip"]
    barrier, gshms, pshm = w["barrier"], w["gshms"], w["pshm"]
    rng = np.random.default_rng(cfg.get("seed", 1337) + args.seed + 101 * rank)
    out = args.out
    best = w["best"]
    t0 = time.time()
    gn = 0.0

    for step in range(w["start"] + 1, steps + 1):
        lr = lr_at(step, steps, cfg["lr"], warmup)
        x, y = get_batch(train, cfg["block_size"], micro, rng)
        loss = model.loss(x, y)
        opt.zero()
        backward(loss)
        _write_flat(gshms[rank], named, offs)
        barrier.wait()

        if rank == 0:
            acc = np.zeros(total, dtype=np.float32)
            for g in gshms:
                acc += np.frombuffer(g, dtype=np.float32)
            acc /= workers
            for (_, p), (shape, off, n) in zip(named, offs):
                p.grad = acc[off:off + n].reshape(shape).copy()
            gn = opt.clip(clip)
            opt.step(lr)
            _write_data(pshm, named, offs)
        barrier.wait()
        _read_flat(pshm, named, offs)

        if rank != 0:
            continue

        if step % cfg.get("log_every", 10) == 0:
            el = time.time() - t0
            print(f"step {step} loss {to_scalar(loss.data):.4f} lr {lr:.2e} gn {gn:.2f} {el:.0f}s", flush=True)
        if step % cfg.get("eval_every", 200) == 0 or step == steps:
            xv, yv = get_batch(val, cfg["block_size"], max(1, cfg["batch_size"] // 2), rng)
            vl = to_scalar(model.loss(xv, yv).data)
            ppl = math.exp(min(20.0, vl))
            flag = ""
            if vl < best:
                best = vl
                save(model, cfg, os.path.join(out, "best.npz"), step, best)
                flag = " best"
            print(f"eval step {step} val {vl:.4f} ppl {ppl:.2f}{flag}", flush=True)
        if step % cfg.get("save_every", 500) == 0 or step == steps:
            save(model, cfg, os.path.join(out, "ckpt.npz"), step, best)
            save(model, cfg, os.path.join(out, f"ckpt_{step}.npz"), step, best)

    if rank == 0:
        print(f"done {time.time() - t0:.0f}s best val {best:.4f}", flush=True)
        if w.get("queue") is not None:
            w["queue"].put(best)


def parallel_train(cfg, args, model, opt, train, val, start, steps, warmup, clip, best, dtype):
    import multiprocessing as mp

    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    try:
        ctx = mp.get_context("fork")
        shared_model = True
    except ValueError:
        ctx = mp.get_context("spawn")
        shared_model = False

    cpu = os.cpu_count() or 1
    if args.workers > cpu - 1:
        print(f"workers {args.workers} above cpu-1, clamping to {cpu - 1}", flush=True)
        args.workers = max(1, cpu - 1)
    if args.workers < 2:
        return None

    named = model.named_params()
    offs, total = _layout(named)
    gshms = [ctx.RawArray("f", total) for _ in range(args.workers)]
    pshm = ctx.RawArray("f", total)
    _write_data(pshm, named, offs)
    barrier = ctx.Barrier(args.workers)
    queue = None if shared_model else ctx.Queue()
    state = None if shared_model else {k: np.array(v) for k, v in model.state().items()}

    w = {"cfg": cfg, "args": args, "model": model if shared_model else None, "state": state,
         "data": args.data, "dtype": dtype, "offs": offs, "total": total, "steps": steps,
         "warmup": warmup, "clip": clip, "barrier": barrier, "gshms": gshms, "pshm": pshm,
         "start": start, "best": best, "queue": queue}

    procs = [ctx.Process(target=_par_loop, args=(r, w), daemon=True) for r in range(1, args.workers)]
    for p in procs:
        p.start()
    try:
        _par_loop(0, w)
    finally:
        for p in procs:
            p.join(timeout=120)
    if queue is not None:
        try:
            return queue.get(timeout=30)
        except Exception:
            return best
    return best


def save(model, cfg, path, step=0, best=float("inf")):
    state = {k: to_numpy(v) for k, v in model.state().items()}
    np.savez(path, cfg=np.array(json.dumps(cfg)), step=np.array(step), best=np.array(best), **state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/nano")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
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
    t0 = time.time()

    if args.workers > 1:
        done = parallel_train(cfg, args, model, opt, train, val, start, steps, warmup, clip, best, dt)
        if done is not None:
            return

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
            print(f"step {step} loss {to_scalar(loss.data):.4f} lr {lr:.2e} gn {gn:.2f} {el:.0f}s", flush=True)
        if step % cfg.get("eval_every", 200) == 0 or step == steps:
            xv, yv = get_batch(val, cfg["block_size"], cfg["batch_size"], rng)
            vl = to_scalar(model.loss(xv, yv).data)
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