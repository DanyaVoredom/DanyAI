import argparse
import json
import numpy as np
from .model import GPT
from .tokenizer import BPETokenizer


def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["cfg"]))
    model = GPT(cfg)
    model.load_state({k: z[k] for k in z.files if k != "cfg"})
    return model, cfg


def generate(model, tok, prompt, tokens=200, temp=0.8, top_k=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt) or [10]
    block = model.cfg["block_size"]
    for _ in range(tokens):
        ctx = np.array([ids[-block:]], dtype=np.int64)
        logits = model.forward(ctx).data[0, -1].astype(np.float64)
        if temp <= 0:
            nxt = int(logits.argmax())
        else:
            logits /= temp
            if top_k and top_k < logits.size:
                thr = np.sort(logits)[-top_k]
                logits[logits < thr] = -np.inf
            p = np.exp(logits - logits.max())
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
        ids.append(nxt)
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/best.npz")
    ap.add_argument("--tok", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, _ = load_ckpt(args.ckpt)
    tok = BPETokenizer.load(args.tok)
    print(generate(model, tok, args.prompt, args.tokens, args.temp, args.top_k, args.seed))


if __name__ == "__main__":
    main()