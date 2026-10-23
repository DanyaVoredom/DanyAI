#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PWTimeout

URL = "https://colab.research.google.com/github/DanyaVoredom/DanyAI/blob/dev/colab/train_danyai.ipynb"

PACK = (
    "!cd /content/DanyAI && tar czf /content/danyai_artifacts.tgz "
    "runs/colab data/tokenizer.json data/meta.json configs/colab.json "
    "&& cp runs/colab/best.npz /content/best_colab.npz && ls -lh /content"
)

DIALOGS = ["Run anyway", "Connect", "Yes", "OK", "Continue"]

FILES_BTN = [
    'button[aria-label="Files"]',
    '[data-tooltip="Files"]',
    'mwc-icon-button[title="Files"]',
]


def log(msg):
    print(msg, flush=True)


def nudge(page):
    for name in DIALOGS:
        try:
            btn = page.get_by_role("button", name=name, exact=True)
            if btn.count() and btn.first.is_visible():
                btn.first.click(timeout=2000)
                log(f"dialog: {name}")
                page.wait_for_timeout(500)
        except Exception:
            pass


def wait_notebook(page, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if page.locator("textarea.inputarea").count():
                return True
        except Exception:
            pass
        nudge(page)
        page.wait_for_timeout(1000)
    return False


def insert_cell(page, code):
    btns = [
        page.get_by_role("button", name="Code", exact=True),
        page.locator("text=+ Code"),
        page.locator('[aria-label="Add code cell"]'),
    ]
    clicked = False
    for b in btns:
        try:
            if b.count():
                b.last.click(timeout=5000)
                clicked = True
                break
        except Exception:
            continue
    if not clicked:
        return False
    page.wait_for_timeout(1200)
    areas = page.locator("textarea.inputarea")
    areas.nth(areas.count() - 1).click()
    page.wait_for_timeout(400)
    page.keyboard.type(code, delay=0)
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    return True


def cell_logs(page):
    out = []
    try:
        blocks = page.locator(".output_subarea pre")
        for i in range(blocks.count()):
            try:
                t = blocks.nth(i).inner_text(timeout=1000)
            except Exception:
                continue
            if t:
                out.append(t)
    except Exception:
        pass
    return out


def stream(page, seen, stop_when, limit):
    end = time.time() + limit
    idle_since = None
    marker = False
    while time.time() < end:
        nudge(page)
        cur = cell_logs(page)
        for i, text in enumerate(cur):
            if i >= len(seen):
                seen.extend([""] * (i + 1 - len(seen)))
            old = seen[i]
            if len(text) > len(old):
                chunk = text[len(old):]
                for line in chunk.splitlines():
                    log(line)
                if stop_when and stop_when in text:
                    marker = True
                seen[i] = text
        try:
            running = page.locator(".cell.running").count()
        except Exception:
            running = 0
        if running == 0:
            idle_since = idle_since or time.time()
            if marker and time.time() - idle_since > 20:
                return True
        else:
            idle_since = None
        page.wait_for_timeout(2000)
    return False


def run_all(page):
    page.keyboard.press("Control+F9")
    page.wait_for_timeout(3000)
    nudge(page)


def pick_gpu(page):
    try:
        page.get_by_role("menuitem", name="Runtime").click(timeout=5000)
    except Exception:
        try:
            page.locator("text=Runtime").first.click(timeout=5000)
        except Exception:
            return False
    page.wait_for_timeout(600)
    try:
        page.locator("text=Change runtime type").first.click(timeout=5000)
    except Exception:
        return False
    page.wait_for_timeout(1500)
    for sel in ['select', 'mwc-select', '[role="listbox"]']:
        try:
            box = page.locator(sel).first
            if box.count():
                box.click(timeout=3000)
                page.wait_for_timeout(500)
                page.locator("text=T4 GPU").first.click(timeout=3000)
                page.wait_for_timeout(500)
                break
        except Exception:
            continue
    for name in ["Save", "OK"]:
        try:
            page.get_by_role("button", name=name, exact=True).first.click(timeout=3000)
            break
        except Exception:
            continue
    log("gpu: T4 requested")
    return True


def open_files(page):
    for sel in FILES_BTN:
        try:
            el = page.locator(sel).first
            if el.count():
                el.click(timeout=5000)
                page.wait_for_timeout(1500)
                return True
        except Exception:
            continue
    return False


def download(page, name, dest):
    if not open_files(page):
        log("files: pane not found")
        return False
    page.wait_for_timeout(1500)
    try:
        row = page.get_by_text(name, exact=True).first
        row.scroll_into_view_if_needed(timeout=10000)
        row.hover(timeout=5000)
        page.wait_for_timeout(700)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with page.expect_download(timeout=180000) as info:
            btn = page.locator('[aria-label="Download"]').first
            if not btn.count():
                btn = page.locator('[title="Download"]').first
            btn.click(timeout=10000)
        info.value.save_as(str(dest))
        log(f"saved: {dest}")
        return True
    except Exception as e:
        log(f"download failed: {type(e).__name__}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--profile", default=str(Path(os.environ.get("LOCALAPPDATA", ".")) / "DanyAI" / "colab-profile"))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="runs/mini/best_colab.npz")
    ap.add_argument("--file", default="best_colab.npz")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--channel", default="chrome")
    args = ap.parse_args()

    with sync_playwright() as p:
        kw = dict(
            user_data_dir=args.profile,
            headless=not args.headed,
            accept_downloads=True,
            viewport={"width": 1440, "height": 960},
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            ctx = p.chromium.launch_persistent_context(channel=args.channel, **kw)
        except Exception:
            log(f"channel {args.channel} unavailable, using bundled chromium")
            ctx = p.chromium.launch_persistent_context(**kw)

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=120000)
        if not wait_notebook(page, 120000):
            log("error: notebook editor did not load")
            ctx.close()
            return 1
        log("notebook loaded")
        nudge(page)
        page.wait_for_timeout(2000)

        if args.gpu:
            pick_gpu(page)

        if not insert_cell(page, PACK):
            log("error: could not insert artifact cell")
        else:
            log("artifact cell added")

        run_all(page)
        log("run all started")
        ok = stream(page, [], "best_colab.npz", args.timeout)
        log("run finished" if ok else "run timed out or marker not found")

        if not args.no_download:
            if not download(page, args.file, Path(args.out)):
                log("manual: download from the open file pane")

        if args.keep_open or args.headed:
            log("browser left open, Ctrl+C to exit")
            try:
                while True:
                    page.wait_for_timeout(5000)
            except KeyboardInterrupt:
                pass
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())