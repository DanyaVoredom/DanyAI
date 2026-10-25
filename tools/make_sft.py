import argparse
import json
import os
import re

SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def stories(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for block in text.replace("<|endoftext|>", "\n\n").split("\n\n"):
        t = block.strip()
        if len(t) >= 200:
            yield t


def pairs(path, per_story=2):
    for t in stories(path):
        parts = SPLIT_RE.split(t, maxsplit=1)
        if len(parts) != 2:
            continue
        head, rest = parts[0].strip(), parts[1].strip()
        if len(head) < 20 or len(rest) < 80:
            continue
        yield {"user": "Continue the story: " + head, "assistant": rest}
        words = head.split()
        if len(words) >= 4:
            yield {"user": "Write a short story that starts with: " + " ".join(words[:4]), "assistant": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/tinystories/TinyStories-valid.txt")
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--limit", type=int, default=8000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in pairs(args.src):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n >= args.limit:
                break
    print(f"sft rows {n} bytes {os.path.getsize(args.out)}", flush=True)
    if n == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()