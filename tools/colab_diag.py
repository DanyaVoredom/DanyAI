import json
import os
import urllib.request

from playwright.sync_api import sync_playwright

EP = os.environ.get("COLAB_CDP_ENDPOINT", "http://127.0.0.1:9333")
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp(EP)
    ctx = b.contexts[0]
    print("pages", len(ctx.pages))
    for pg in ctx.pages:
        print("url", pg.url)
    pg = ctx.pages[0]
    try:
        body = pg.evaluate("() => (document.body && document.body.innerText || '').slice(0, 1200)")
        print("---body---")
        print(body)
    except Exception as e:
        print("body err", e)
    try:
        print("---frames---")
        for f in pg.frames:
            t = ""
            try:
                t = f.evaluate("() => (document.body && document.body.innerText || '').slice(-800)")
            except Exception:
                pass
            if t.strip():
                print("FRAME", f.url[:80])
                print(t.strip()[-600:])
    except Exception as e:
        print("frames err", e)
    try:
        pg.screenshot(path="/home/lipton/DanyAI/runs/state.png")
        print("shot saved")
    except Exception as e:
        print("shot err", e)
