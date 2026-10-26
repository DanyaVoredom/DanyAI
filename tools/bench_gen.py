import argparse
import json
import time

import numpy as np

from danyai.infer import Engine
from danyai.tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phone_handoff/ckpt.npz")
    ap.add_argument("--tok", default="data2/tokenizer.json")
    ap.add_argument("--prompt", default="Привет, как дела?")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.15)
    args = ap.parse_args()

    eng, cfg = Engine.from_file(args.ckpt)
    tok = BPETokenizer.load(args.tok)
    best = 0.0
    for i in range(args.runs):
        t = time.time()
        eng.generate(tok, args.prompt, tokens=args.tokens, temp=args.temp,
                     top_k=args.top_k, top_p=args.top_p, repetition=args.rep, seed=i)
        dt = time.time() - t
        tps = args.tokens / dt
        best = max(best, tps)
        print(f"run {i}: {dt:.2f}s {tps:.1f} tok/s", flush=True)
    print(f"best {best:.1f} tok/s")


if __name__ == "__main__":
    main()
