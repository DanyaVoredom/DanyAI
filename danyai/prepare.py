import argparse
import html
import json
import os
import re
import sys
import time
import numpy as np
from .tokenizer import BPETokenizer

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
WS_RE = re.compile(r"[ \t\r\f\v]+")
NL_RE = re.compile(r"\n{3,}")


def clean_text(text):
    if "<" in text and ">" in text:
        text = SCRIPT_RE.sub(" ", text)
        text = TAG_RE.sub("\n", text)
        text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\ufeff", "")
    text = WS_RE.sub(" ", text)
    text = NL_RE.sub("\n\n", text)
    return text.strip()


def read_corpus(root):
    parts = []
    skipped = 0
    for cur, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.lower().endswith(".txt") or fn.startswith("_"):
                skipped += 1
                continue
            with open(os.path.join(cur, fn), "r", encoding="utf-8", errors="ignore") as f:
                t = clean_text(f.read())
            if t:
                parts.append(t)
            else:
                skipped += 1
    if skipped:
        print(f"skipped {skipped} files", flush=True)
    return "\n\n".join(parts)


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