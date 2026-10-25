import argparse
import html
import io
import os
import re
import sys

PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S | re.I)
FORM_RE = re.compile(r"<form.*?</form>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
NL_RE = re.compile(r"\n{3,}")


def cyr_score(t):
    return sum(1 for ch in t if ch == "ё" or "а" <= ch.lower() <= "я")


def decode(raw):
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    best = None
    for enc in ("koi8-r", "cp1251"):
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


def strip_html(text):
    if "<pre" not in text.lower():
        return text
    m = PRE_RE.search(text)
    body = m.group(1) if m else text
    body = FORM_RE.sub(" ", body)
    body = TAG_RE.sub("\n", body)
    body = html.unescape(body)
    body = body.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    body = NL_RE.sub("\n\n", body)
    return body.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/russian")
    args = ap.parse_args()
    fixed = 0
    removed = 0
    for fn in sorted(os.listdir(args.dir)):
        path = os.path.join(args.dir, fn)
        if fn.lower().endswith(".html") or fn.startswith("_"):
            os.remove(path)
            removed += 1
            continue
        if not fn.lower().endswith(".txt"):
            continue
        with open(path, "rb") as f:
            text = decode(f.read())
        clean = strip_html(text)
        if cyr_score(clean) < 500:
            os.remove(path)
            removed += 1
            print(f"drop {fn} {cyr_score(clean)}", flush=True)
            continue
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean)
        fixed += 1
        print(f"fix {fn} chars {len(clean)} cyr {cyr_score(clean)}", flush=True)
    print(f"fixed {fixed} removed {removed}", flush=True)
    if fixed == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()