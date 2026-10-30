import json
import os
import re
import subprocess
import urllib.request

from playwright.sync_api import sync_playwright


def cdp_endpoint():
    env = os.environ.get("COLAB_CDP_ENDPOINT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["pgrep", "-af", "chrome"], capture_output=True, text=True
        ).stdout
    except Exception:
        return "http://127.0.0.1:9333"
    for port in reversed(re.findall(r"--remote-debugging-port=(\d+)", out)):
        try:
            urllib.request.urlopen("http://127.0.0.1:%s/json/version" % port, timeout=2)
            return "http://127.0.0.1:%s" % port
        except Exception:
            continue
    return "http://127.0.0.1:9333"


EP = cdp_endpoint()
print("cdp", EP)
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

