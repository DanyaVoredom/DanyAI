import argparse
import sys
import numpy as np
from .sample import load_ckpt, generate
from .tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/ckpt.npz")
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=200)
    args = ap.parse_args()

    model, cfg = load_ckpt(args.ckpt)
    tok = BPETokenizer.load(args.tok)
    print(cfg.get("name", "DanyAI"), "ready. пустая строка для выхода.", flush=True)
    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            break
        print(generate(model, tok, prompt, args.tokens, args.temp, args.top_k,
                       seed=int(np.random.randint(1 << 30))), flush=True)


if __name__ == "__main__":
    sys.exit(main())