import argparse
import json
import os
import random


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", default="data/instruct_en.jsonl")
    ap.add_argument("--ru", default="data/instruct_ru.jsonl")
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = []
    for path in (args.en, args.ru):
        if os.path.exists(path):
            part = load(path)
            print(os.path.basename(path), len(part), flush=True)
            rows.extend(part)
    if not rows:
        raise SystemExit("no instruct data")
    random.Random(args.seed).shuffle(rows)
    if args.limit > 0:
        rows = rows[:args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"sft rows {len(rows)} bytes {os.path.getsize(args.out)}", flush=True)


if __name__ == "__main__":
    main()
