import argparse
import json
import os
import time

import numpy as np
import torch

from .tokenizer import BPETokenizer
from .torch_train import GPT, build_optimizer, lr_at, save

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
    if len(ids) > block + 1:
        ids = ids[:block + 1]
    if len(ids) < 3:
        return None
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
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--eval", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=int, default=1)
    args = ap.parse_args()

    z = np.load(args.ckpt, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    cfg["block_size"] = args.block
    tok = BPETokenizer.load(args.tok)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    model = GPT(cfg).to(device)
    model.load_numpy({k: z[k] for k in z.files if k not in ("cfg", "step", "best")})
    z.close()

    rows = load_jsonl(args.data)
    rng = np.random.default_rng(cfg.get("seed", 1337) + args.seed)
    order = rng.permutation(len(rows))
    cut = max(1, int(len(rows) * args.eval))
    val_rows = [rows[i] for i in order[:cut]]
    train_rows = [rows[i] for i in order[cut:]] or val_rows

    print(f"sft rows {len(train_rows)} val {len(val_rows)}", flush=True)

    def make(rs):
        out = []
        for r in rs:
            b = build_ids(tok, r, args.block)
            if b is not None:
                out.append(b)
        return out

    t0 = time.time()
    train = make(train_rows)
    val = make(val_rows)
    if not train:
        raise SystemExit("no usable sft examples")
    print(f"tokenized {len(train)} examples {time.time() - t0:.0f}s", flush=True)

    opt = build_optimizer(model, args.lr, cfg.get("weight_decay", 0.1))
    os.makedirs(args.out, exist_ok=True)
    use_amp = bool(args.amp) and device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    best = float("inf")
    log_every = max(1, args.steps // 50)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for grp in opt.param_groups:
            grp["lr"] = lr
        batch = [train[i] for i in rng.integers(0, len(train), size=args.batch)]
        x, y = pad(batch)
        xb = torch.from_numpy(x).to(device, non_blocking=True)
        yb = torch.from_numpy(y).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast:
            loss = model.loss(xb, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0))
        scaler.step(opt)
        scaler.update()

        if step % log_every == 0:
            print(f"step {step} loss {float(loss.detach()):.4f} lr {lr:.2e} gn {gn:.2f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if val and (step % max(1, args.steps // 5) == 0 or step == args.steps):
            tot = 0.0
            cnt = 0
            model.eval()
            with torch.no_grad():
                for i in range(0, len(val), max(1, args.batch)):
                    xv, yv = pad(val[i:i + args.batch])
                    n = int((yv >= 0).sum())
                    if n == 0:
                        continue
                    xb = torch.from_numpy(xv).to(device)
                    yb = torch.from_numpy(yv).to(device)
                    with autocast:
                        l = float(model.loss(xb, yb))
                    tot += l * n
                    cnt += n
            model.train()
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
