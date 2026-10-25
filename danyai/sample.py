import argparse
import json
import numpy as np
from .infer import Engine, apply_penalties, sample_token
from .tokenizer import BPETokenizer


def load_ckpt(path):
    engine, cfg = Engine.from_file(path)
    return engine, cfg


def generate(engine, tok, prompt, tokens=200, temp=0.8, top_k=40, top_p=1.0,
             repetition=1.0, presence=0.0, frequency=0.0, seed=0):
    return engine.generate(tok, prompt, tokens, temp, top_k, top_p,
                           repetition, presence, frequency, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/best.npz")
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition", type=float, default=1.0)
    ap.add_argument("--presence", type=float, default=0.0)
    ap.add_argument("--frequency", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine, _ = load_ckpt(args.ckpt)
    tok = BPETokenizer.load(args.tok)
    print(engine.generate(tok, args.prompt, args.tokens, args.temp, args.top_k,
                          args.top_p, args.repetition, args.presence,
                          args.frequency, args.seed))


if __name__ == "__main__":
    main()