import os
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, "/home/lipton/DanyAI")
from playwright.sync_api import sync_playwright

import tools.colab_cli as cc


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
    page = b.contexts[0].pages[0]
    body = cc.body_text(page)
    print("len", len(body))
    print("hits", [m for m in cc.GPU_BLOCK if m in body])
    print("frames_text len", len(cc.frames_text(page)))
