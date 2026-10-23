#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SETUP = """!pip -q install numpy
!rm -rf /content/DanyAI
!git clone -q https://github.com/DanyaVoredom/DanyAI.git /content/DanyAI
%cd /content/DanyAI
!pip -q install -e .
!nvidia-smi --query-gpu=name,memory.total --format=csv || true
print("setup done")"""

TRAIN = """import os, json, subprocess
os.makedirs("data/raw", exist_ok=True)
ids = [1342, 11, 1661, 2701, 84, 98, 345, 5200, 16328, 2600]
for i in ids:
    subprocess.run(["curl", "-s", "-L",
                    f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt",
                    "-o", f"data/raw/pg{i}.txt"], check=False)
print("corpus files", len(os.listdir("data/raw")))
!python -m danyai.prepare --input data/raw --out data --vocab __VOCAB__ --val 0.02
cfg = json.load(open("configs/mini.json"))
cfg.update(block_size=__BLOCK__, batch_size=__BATCH__, steps=__STEPS__, lr=0.0015, warmup=__WARMUP__)
cfg["name"] = "DanyAI-Mini-Colab"
json.dump(cfg, open("configs/colab.json", "w"), indent=2)
print(cfg)
!python -m danyai.train --config configs/colab.json --data data --out runs/colab
!python -m danyai.sample --ckpt runs/colab/best.npz --tok data/tokenizer.json --prompt "The house" --tokens 200 --temp 0.8
print("train done")"""

PACK = """import base64, pathlib
for name in ["runs/colab/best.npz", "data/tokenizer.json", "data/meta.json", "configs/colab.json"]:
    p = pathlib.Path(name)
    if not p.exists():
        print("@@MISSING", name)
        continue
    b = base64.b64encode(p.read_bytes()).decode()
    print(f"@@BEGIN {name} {len(b)}")
    for i in range(0, len(b), 50000):
        print(b[i:i + 50000])
    print(f"@@END {name}")
print("pack done")"""


def log(msg):
    print(msg, flush=True)


def notebook(cells):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [],
    }
    for src in cells:
        lines = src.splitlines(True)
        nb["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": lines,
        })
    return nb


def build(train_cell, use_git, artifacts):
    cells = []
    if use_git:
        cells.append(SETUP)
    cells.append(train_cell)
    if artifacts:
        cells.append(PACK)
    return notebook(cells)


def publish(path, desc):
    out = subprocess.run(
        ["gh", "gist", "create", "--public", "--desc", desc, str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "gist create failed")
    url = out.stdout.strip().splitlines()[-1].strip()
    return url


def gist_colab_url(gist_url):
    m = re.search(r"gist\.github\.com/([^/]+)/([0-9a-f]+)", gist_url)
    if not m:
        raise RuntimeError(f"cannot parse gist url: {gist_url}")
    user, gid = m.group(1), m.group(2)
    return f"https://colab.research.google.com/gist/{user}/{gid}/train_danyai.ipynb"


RUN_LABELS = ["Выполнить", "Run anyway", "Run", "Выполнить все", "Run all"]


def alive(ctx):
    try:
        return len(ctx.pages) > 0 and not ctx.pages[0].is_closed()
    except Exception:
        return False


def accept_run(page):
    for label in RUN_LABELS:
        try:
            loc = page.get_by_role("button", name=label, exact=True)
            for i in range(min(loc.count(), 4)):
                b = loc.nth(i)
                if b.is_visible():
                    b.click(timeout=2000, force=True)
                    log(f"confirmed: {label}")
                    page.wait_for_timeout(800)
                    return True
        except Exception:
            continue
    return False


def frames_text(page):
    parts = []
    for frame in page.frames:
        try:
            blocks = frame.locator("pre")
            for i in range(blocks.count()):
                try:
                    t = blocks.nth(i).inner_text(timeout=500)
                except Exception:
                    continue
                if t:
                    parts.append(t)
        except Exception:
            continue
    return "\n".join(parts)


def wait_ready(page, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if page.locator("textarea.inputarea").count() > 0 and page.locator(".cell").count() > 0:
                return True
        except Exception:
            pass
        page.wait_for_timeout(1500)
    return False


def stream(page, ctx, stop_when, limit, dump):
    end = time.time() + limit
    last = 0
    marker_at = None
    while time.time() < end:
        if not alive(ctx):
            return False
        try:
            accept_run(page)
            text = frames_text(page)
            if len(text) > last:
                for line in text[last:].splitlines():
                    if not line.startswith("@@") and len(line) < 400:
                        log(line)
                last = len(text)
                dump.parent.mkdir(parents=True, exist_ok=True)
                dump.write_text(text, encoding="utf-8")
            if stop_when in text and marker_at is None:
                marker_at = time.time()
            try:
                running = page.locator(".cell.running").count()
            except Exception:
                running = 1
            if marker_at and running == 0 and time.time() - marker_at > 8:
                return True
            page.wait_for_timeout(3000)
        except Exception as e:
            if not alive(ctx):
                return False
            log(f"stream: {type(e).__name__}")
            page.wait_for_timeout(2000)
    return False


def extract(text, out):
    saved = []
    for m in re.finditer(r"@@BEGIN (\S+) (\d+)\n(.*?)@@END \1", text, re.S):
        name, size, payload = m.group(1), int(m.group(2)), m.group(3)
        b64 = "".join(payload.split())
        if len(b64) < size:
            log(f"artifact {name}: truncated {len(b64)}/{size}")
            continue
        dest = out / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(b64[:size]))
        saved.append((str(dest), len(b64) * 3 // 4))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--notebook", default="colab/train_danyai.ipynb")
    ap.add_argument("--profile", default=str(Path(os.environ.get("LOCALAPPDATA", ".")) / "DanyAI" / "colab-profile"))
    ap.add_argument("--out", default="runs/mini")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--timeout", type=int, default=21600)
    ap.add_argument("--ready-wait", type=int, default=180)
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--dump", default="runs/colab_output.txt")
    ap.add_argument("--channel", default="chrome")
    args = ap.parse_args()

    out = Path(args.out)
    nb_path = Path(args.notebook)

    if args.url:
        url = args.url
    else:
        train = (TRAIN
                 .replace("__VOCAB__", str(args.vocab))
                 .replace("__BLOCK__", str(args.block))
                 .replace("__BATCH__", str(args.batch))
                 .replace("__STEPS__", str(args.steps))
                 .replace("__WARMUP__", str(args.warmup)))
        nb = build(train, not args.no_git, not args.no_artifacts)
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        gist = publish(nb_path, "danyai colab run")
        log(f"gist {gist}")
        url = gist_colab_url(gist)

    log(f"open {url}")

    with sync_playwright() as p:
        kw = dict(
            user_data_dir=args.profile,
            headless=not args.headed,
            viewport={"width": 1440, "height": 960},
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            ctx = p.chromium.launch_persistent_context(channel=args.channel, **kw)
        except Exception:
            ctx = p.chromium.launch_persistent_context(**kw)

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        if not wait_ready(page, args.ready_wait * 1000):
            log("error: notebook did not open")
            try:
                page.screenshot(path=str(out / "failed.png"))
            except Exception:
                pass
            if args.headed:
                try:
                    while alive(ctx):
                        page.wait_for_timeout(5000)
                except KeyboardInterrupt:
                    pass
            ctx.close()
            return 1
        log("notebook ready")
        page.wait_for_timeout(1500)
        page.keyboard.press("Control+F9")
        page.wait_for_timeout(4000)
        log("run all started")

        stop = "pack done" if not args.no_artifacts else "train done"
        ok = stream(page, ctx, stop, args.timeout, Path(args.dump))
        log("finished" if ok else "timed out")

        if ok and not args.no_artifacts:
            text = frames_text(page)
            Path(args.dump).write_text(text, encoding="utf-8")
            for path, size in extract(text, out):
                log(f"saved {path} ~{size} bytes")

        if args.keep_open or args.headed:
            log("browser open, Ctrl+C to exit")
            try:
                while alive(ctx):
                    page.wait_for_timeout(5000)
            except KeyboardInterrupt:
                pass
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())