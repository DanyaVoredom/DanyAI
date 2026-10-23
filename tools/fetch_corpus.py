import argparse
import os
import subprocess
import sys

BOOKS = [1342, 11, 1661, 2701, 84, 98, 345, 5200, 16328, 2600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--ids", default=",".join(str(i) for i in BOOKS))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for raw in args.ids.split(","):
        i = raw.strip()
        if not i:
            continue
        url = f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt"
        dst = os.path.join(args.out, f"pg{i}.txt")
        r = subprocess.run(["curl.exe", "-k", "-s", "-L", "--max-time", "120", "-A", "Mozilla/5.0", url, "-o", dst])
        size = os.path.getsize(dst) if os.path.exists(dst) else 0
        print(f"{i} {size} {'ok' if r.returncode == 0 and size > 1000 else 'fail'}")

    if not any(f.endswith(".txt") for f in os.listdir(args.out)):
        sys.exit(1)


if __name__ == "__main__":
    main()