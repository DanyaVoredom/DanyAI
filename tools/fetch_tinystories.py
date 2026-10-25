import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
DATASET = "roneneldan/TinyStories"
PAGE = 100


def fetch_rows(split, offset, length):
    q = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": length}
    )
    req = urllib.request.Request(API + "?" + q, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def collect(split, limit, out_path):
    texts = []
    offset = 0
    while offset < limit:
        n = min(PAGE, limit - offset)
        try:
            data = fetch_rows(split, offset, n)
        except Exception as e:
            print(f"rows {split} {offset} fail {e}", flush=True)
            break
        rows = data.get("rows") or []
        if not rows:
            break
        for item in rows:
            t = (item.get("row") or {}).get("text")
            if t:
                texts.append(t.strip())
        offset += len(rows)
        print(f"rows {split} {offset}/{limit}", flush=True)
        if len(rows) < n:
            break
    if not texts:
        return 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))
    print(f"tinystories {split} {len(texts)} stories {os.path.getsize(out_path)} bytes", flush=True)
    return len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/tinystories")
    ap.add_argument("--valid", type=int, default=5000)
    ap.add_argument("--train", type=int, default=40000)
    args = ap.parse_args()
    total = 0
    if args.valid:
        total += collect("validation", args.valid, os.path.join(args.out, "TinyStories-valid.txt"))
    if args.train:
        total += collect("train", args.train, os.path.join(args.out, "TinyStories-train.txt"))
    print(f"total stories {total}", flush=True)
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()