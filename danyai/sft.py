import argparse
import json
import os
import time
import numpy as np
from .model import GPT
from .train import AdamW, lr_at, save
from .backend import to_scalar
from .tensor import backward
from .tokenizer import BPETokenizer

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_ids(tok, row, block):
    text = USER + row["user"] + "\n" + ASSISTANT + row["assistant"] + END
    ids = tok.encode(text)
    prompt = tok.encode(USER + row["user"] + "\n" + ASSISTANT)
    n = min(len(ids), block + 1)
    if len(ids) < 3:
        return None
    if len(ids) > block + 1:
        ids = ids[:block + 1]
        prompt = prompt[:max(1, len(prompt))]
    x = np.array(ids[:-1], dtype=np.int64)
    y = np.array(ids[1:], dtype=np.int64)
    cut = min(len(prompt) - 1, y.size)
    y[:max(0, cut)] = -100
    if cut >= y.size:
        return None
    return x, y


def pad(batch):
    n = max(len(x) for x, _ in batch)
    xs = np.zeros((len(batch), n), dtype=np.int64)
    ys = np.full((len(batch), n), -100, dtype=np.int64)
    for i, (x, y) in enumerate(batch):
        xs[i, :x.size] = x
        ys[i, :y.size] = y
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/best.npz")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--eval", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.ckpt, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    model = GPT(cfg)
    model.load_state({k: z[k] for k in z.files if k not in ("cfg", "step", "best")})
    tok = BPETokenizer.load(args.tok)

    rows = load_jsonl(args.data)
    rng = np.random.default_rng(cfg.get("seed", 1337) + args.seed)
    order = rng.permutation(len(rows))
    cut = max(1, int(len(rows) * args.eval))
    val_rows = [rows[i] for i in order[:cut]]
    train_rows = [rows[i] for i in order[cut:]] or val_rows

    def make(rs):
        out = []
        for r in rs:
            b = build_ids(tok, r, args.block)
            if b is not None:
                out.append(b)
        return out

    train = make(train_rows)
    val = make(val_rows)
    if not train:
        raise SystemExit("no usable sft examples")

    opt = AdamW(model.named_params(), args.lr, wd=cfg.get("weight_decay", 0.1))
    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    t0 = time.time()
    log_every = max(1, args.steps // 50)

    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        batch = [train[i] for i in rng.integers(0, len(train), size=args.batch)]
        x, y = pad(batch)
        loss = model.loss(x, y)
        opt.zero()
        backward(loss)
        gn = opt.clip(cfg.get("grad_clip", 1.0))
        opt.step(lr)
        if step % log_every == 0:
            print(f"step {step} loss {to_scalar(loss.data):.4f} lr {lr:.2e} gn {gn:.2f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if val and (step % max(1, args.steps // 5) == 0 or step == args.steps):
            eb = max(1, min(args.batch, 8))
            tot = 0.0
            cnt = 0
            for i in range(0, len(val), eb):
                xv, yv = pad(val[i:i + eb])
                n = int((yv >= 0).sum())
                if n == 0:
                    continue
                tot += to_scalar(model.loss(xv, yv).data) * n
                cnt += n
            vl = tot / cnt if cnt else float("inf")
            flag = ""
            if vl < best:
                best = vl
                save(model, cfg, os.path.join(args.out, "best.npz"), step, best)
                flag = " best"
            print(f"eval step {step} val {vl:.4f}{flag}", flush=True)

    save(model, cfg, os.path.join(args.out, "ckpt.npz"), args.steps, best)
    print(f"done {time.time() - t0:.0f}s best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()