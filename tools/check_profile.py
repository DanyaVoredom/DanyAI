import os
import sys

from playwright.sync_api import sync_playwright

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "/home/lipton/DanyAI/colab-profile-ff"
exe = None
for cand in ("/usr/bin/google-chrome", "/opt/google/chrome/chrome"):
    if os.path.exists(cand):
        exe = cand
        break

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        PROFILE,
        headless=True,
        executable_path=exe,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://myaccount.google.com/", wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(7000)
    body = page.evaluate("() => (document.body && document.body.innerText || '').slice(0, 800)")
    print("url", page.url[:80])
    print("signed_out", "Sign in" in body or "Войти" in body)
    for line in body.splitlines():
        if "@" in line:
            print("line", line.strip()[:80])
    print("head", " | ".join(body.splitlines()[:6])[:200])
    ctx.close()
