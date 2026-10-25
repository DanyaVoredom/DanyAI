import argparse
import os
import re
import subprocess
import sys

BOOKS = [1342, 11, 1661, 2701, 84, 98, 345, 5200, 16328, 2600]

TINYSTORIES = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
]

TINYSTORIES_TRAIN = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"

SECTIONS = ["ANEKDOTY", "RUFANT", "MEMUARY", "RUSS_DETEKTIW", "PROZA"]
BASE = "https://lib.ru/"


def curl(url, dst, timeout=180):
    cmd = ["curl.exe", "-k", "-s", "-L", "--max-time", str(timeout), "-A", "Mozilla/5.0", url, "-o", dst]
    r = subprocess.run(cmd)
    size = os.path.getsize(dst) if os.path.exists(dst) else 0
    ok = r.returncode == 0 and size > 1000
    if not ok and os.path.exists(dst):
        os.remove(dst)
        size = 0
    return ok, size


def cyr_score(t):
    return sum(1 for ch in t if ch == "ё" or "а" <= ch.lower() <= "я")


def decode_ru(raw):
    head = raw[:4096].lower()
    if b"koi8" in head:
        return raw.decode("koi8-r", errors="replace")
    if b"windows-1251" in head or b"cp1251" in head:
        return raw.decode("cp1251", errors="replace")
    best = None
    for enc in ("utf-8", "koi8-r", "cp1251"):
        try:
            t = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        n = cyr_score(t)
        if best is None or n > best[0]:
            best = (n, t)
    if best is None:
        return raw.decode("utf-8", errors="ignore")
    return best[1]


def ascii_variant(url):
    if "?" in url:
        return url + "&format=_Ascii.txt"
    return url + "?format=_Ascii.txt"


def gutenberg(out, ids):
    gdir = os.path.join(out, "gutenberg")
    os.makedirs(gdir, exist_ok=True)
    ok = 0
    for i in ids:
        dst = os.path.join(gdir, f"pg{i}.txt")
        good, size = curl(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt", dst)
        print(f"gutenberg {i} {size} {'ok' if good else 'fail'}", flush=True)
        ok += int(good)
    return ok


def tinystories(out, train=False):
    tdir = os.path.join(out, "tinystories")
    os.makedirs(tdir, exist_ok=True)
    urls = list(TINYSTORIES)
    if train:
        urls.append(TINYSTORIES_TRAIN)
    ok = 0
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        dst = os.path.join(tdir, name)
        good, size = curl(url, dst, timeout=1800)
        print(f"tinystories {name} {size} {'ok' if good else 'fail'}", flush=True)
        ok += int(good)
    return ok


def collect_links(html, page_url):
    links = re.findall(r'href=["\']?([^"\'> ]+\.txt)["\']?', html, re.I)
    out = []
    for link in links:
        if link.startswith("http"):
            out.append(link)
        else:
            out.append(page_url + link.lstrip("./"))
    return out


def libru(out, per_section=60, total_bytes=40 * 1024 * 1024):
    rdir = os.path.join(out, "russian")
    os.makedirs(rdir, exist_ok=True)
    total = 0
    files = 0
    for section in SECTIONS:
        page_url = f"{BASE}{section}/"
        tmp = os.path.join(rdir, f"_{section}.html")
        good, _ = curl(page_url, tmp, timeout=120)
        if not good:
            print(f"russian {section} index fail", flush=True)
            continue
        with open(tmp, "rb") as f:
            html = decode_ru(f.read())
        links = collect_links(html, page_url)
        print(f"russian {section} links {len(links)}", flush=True)
        taken = 0
        for link in links:
            if taken >= per_section or total >= total_bytes:
                break
            name = f"{section}_{os.path.basename(link)}"
            dst = os.path.join(rdir, name)
            if os.path.exists(dst):
                taken += 1
                continue
            ok, size = curl(link, dst, timeout=120)
            if not ok:
                continue
            with open(dst, "rb") as f:
                text = decode_ru(f.read())
            with open(dst, "w", encoding="utf-8") as f:
                f.write(text)
            total += size
            files += 1
            taken += 1
        print(f"russian {section} taken {taken} total {total}", flush=True)
    print(f"russian files {files} bytes {total}", flush=True)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--ids", default=",".join(str(i) for i in BOOKS))
    ap.add_argument("--skip-gutenberg", action="store_true")
    ap.add_argument("--skip-tinystories", action="store_true")
    ap.add_argument("--tinystories-train", action="store_true")
    ap.add_argument("--skip-russian", action="store_true")
    ap.add_argument("--per-section", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.skip_gutenberg:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        gutenberg(args.out, ids)
    if not args.skip_tinystories:
        tinystories(args.out, args.tinystories_train)
    if not args.skip_russian:
        libru(args.out, args.per_section)

    found = 0
    total = 0
    for cur, _, files in os.walk(args.out):
        for fn in files:
            if fn.lower().endswith(".txt"):
                size = os.path.getsize(os.path.join(cur, fn))
                if size > 1000:
                    found += 1
                    total += size
    print(f"total files {found} bytes {total}", flush=True)
    if found == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()