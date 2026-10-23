import time
import numpy as np


def bench(model, block=128, batch=32, steps=3):
    rng = np.random.default_rng(0)
    x = rng.integers(0, model.cfg["vocab_size"], size=(batch, block))
    y = rng.integers(0, model.cfg["vocab_size"], size=(batch, block))
    from .tensor import backward
    model.loss(x, y)
    for _, p in model.named_params():
        p.grad = None
    backward(model.loss(x, y))
    t = time.time()
    for _ in range(steps):
        loss = model.loss(x, y)
        for _, p in model.named_params():
            p.grad = None
        backward(loss)
    dt = (time.time() - t) / steps
    return dt


def main():
    import argparse
    import json
    from .model import GPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nano.json")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    cfg["vocab_size"] = args.vocab
    cfg["block_size"] = args.block
    model = GPT(cfg)
    dt = bench(model, args.block, args.batch, args.steps)
    tok_s = args.block * args.batch / dt
    print(f"{cfg.get('name')} params {model.num_params():,}")
    print(f"{dt:.2f} s/step  {tok_s:.0f} tok/s")


if __name__ == "__main__":
    main()