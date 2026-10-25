import argparse
import sys
import numpy as np
from .infer import Engine
from .tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/ckpt.npz")
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition", type=float, default=1.05)
    ap.add_argument("--presence", type=float, default=0.0)
    ap.add_argument("--frequency", type=float, default=0.0)
    ap.add_argument("--tokens", type=int, default=200)
    args = ap.parse_args()

    engine, cfg = Engine.from_file(args.ckpt)
    tok = BPETokenizer.load(args.tok)
    print(cfg.get("name", "DanyAI"), "ready. empty line to exit.", flush=True)

    def emit(delta, _n):
        sys.stdout.write(delta)
        sys.stdout.flush()

    while True:
        try:
            prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            break
        engine.generate(tok, prompt, args.tokens, args.temp, args.top_k,
                        args.top_p, args.repetition, args.presence,
                        args.frequency, int(np.random.randint(1 << 30)),
                        on_token=emit)
        print(flush=True)


if __name__ == "__main__":
    sys.exit(main())