import argparse
import json
import os
import sys
import time
import numpy as np
from .tokenizer import BPETokenizer


def read_corpus(root):
    parts = []
    for cur, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.lower().endswith(".txt"):
                with open(os.path.join(cur, fn), "r", encoding="utf-8", errors="ignore") as f:
                    parts.append(f.read())
    return "\n".join(parts)


def pick_dtype(vocab):
    if vocab <= 65536:
        return np.uint16, 65536
    return np.uint32, 2 ** 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--val", type=float, default=0.02)
    ap.add_argument("--max-words", type=int, default=30000)
    args = ap.parse_args()

    if args.vocab < 257 or args.vocab > 2 ** 32:
        raise SystemExit("vocab out of range")

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    text = read_corpus(args.input)
    if not text:
        raise SystemExit("no txt in " + args.input)
    print(f"corpus {len(text)} chars {time.time() - t0:.1f}s", flush=True)

    tok = BPETokenizer()
    tok.train(text, args.vocab, args.max_words)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    print(f"tokenizer {tok.vocab_size} tokens {time.time() - t0:.1f}s", flush=True)

    ids = tok.encode(text)
    dtype, _ = pick_dtype(tok.vocab_size)
    arr = np.array(ids, dtype=dtype)
    print(f"encoded {len(arr)} tokens {time.time() - t0:.1f}s", flush=True)

    n = int(len(arr) * (1 - args.val))
    arr[:n].tofile(os.path.join(args.out, "train.bin"))
    arr[n:].tofile(os.path.join(args.out, "val.bin"))

    meta = {
        "vocab_size": tok.vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": int(n),
        "val_tokens": int(len(arr) - n),
        "chars": len(text),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    sys.exit(main())