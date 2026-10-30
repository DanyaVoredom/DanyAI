import os
import re
import subprocess
import urllib.request

from playwright.sync_api import sync_playwright

MARK = (
    "due to usage limits",
    "из-за лимитов",
    "Connect without GPU",
    "Подключиться без ускорителя",
    "No GPU backend",
)


def port():
    out = subprocess.run(["pgrep", "-af", "chrome"], capture_output=True, text=True).stdout
    for p in reversed(re.findall(r"--remote-debugging-port=(\d+)", out)):
        try:
            urllib.request.urlopen("http://127.0.0.1:%s/json/version" % p, timeout=2)
            return p
        except Exception:
            continue
    return "9333"


with sync_playwright() as p:
    b = p.chromium.connect_over_cdp("http://127.0.0.1:%s" % port())
    ctx = b.contexts[0]
    page = ctx.pages[0]
    print("url", page.url[:90])
    body = page.evaluate("() => (document.body && document.body.innerText || '')")
    print("body len", len(body))
    hits = [m for m in MARK if m in body]
    print("markers", hits)
    idx = body.find("лимит")
    if idx < 0:
        idx = body.find("limit")
    if idx >= 0:
        print("ctx:", body[max(0, idx - 200) : idx + 300].replace("\n", " | "))
    for fr in page.frames:
        try:
            t = fr.evaluate("() => (document.body && document.body.innerText || '')")
        except Exception:
            continue
        h = [m for m in MARK if m in t]
        if h:
            print("frame", fr.url[:60], h)
