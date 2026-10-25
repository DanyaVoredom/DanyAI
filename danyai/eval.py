import argparse
import json
import os
import numpy as np
from .infer import Engine


def load_val_tokens(data_dir):
    npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=False)
        key = "val" if "val" in z.files else z.files[0]
        return z[key].astype(np.int64)
    bin_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(bin_path):
        dtype = np.uint16
        meta_path = os.path.join(data_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                dtype = np.dtype(json.load(f).get("dtype", "uint16"))
        return np.fromfile(bin_path, dtype=dtype).astype(np.int64)
    txt_path = os.path.join(data_dir, "val.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return np.frombuffer(f.read().encode("utf-8"), dtype=np.uint8).astype(np.int64)
    raise SystemExit(f"no validation data in {data_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nano/ckpt.npz")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    engine, cfg = Engine.from_file(args.ckpt)
    if args.split == "val":
        data = load_val_tokens(args.data)
    else:
        p = os.path.join(args.data, args.split + ".npz")
        z = np.load(p, allow_pickle=False)
        data = z[z.files[0]].astype(np.int64)

    limit = args.limit if args.limit > 0 else None
    ppl = engine.perplexity(data, block=args.block or None, batch=args.batch, limit=limit)
    tokens = int(data.size)
    print(json.dumps({
        "model": cfg.get("name", "DanyAI"),
        "params": int(sum(v.size for v in engine.p.values())),
        "split": args.split,
        "tokens": tokens,
        "block": int(args.block or engine.block),
        "perplexity": ppl,
        "bits_per_token": float(np.log2(ppl)) if ppl == ppl else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()