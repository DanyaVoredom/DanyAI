import json
import os
import shutil
import sys

from playwright.sync_api import sync_playwright

COOKIES = sys.argv[1] if len(sys.argv) > 1 else "/home/lipton/ff_cookies.json"
PROFILE = sys.argv[2] if len(sys.argv) > 2 else "/home/lipton/DanyAI/colab-profile-ff"

data = json.load(open(COOKIES, encoding="utf-8"))
shutil.rmtree(PROFILE, ignore_errors=True)
os.makedirs(PROFILE, exist_ok=True)

with sync_playwright() as p:
    exe = None
    for cand in ("/usr/bin/google-chrome", "/opt/google/chrome/chrome"):
        if os.path.exists(cand):
            exe = cand
            break
    ctx = p.chromium.launch_persistent_context(
        PROFILE,
        headless=True,
        executable_path=exe,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--disable-features=BlockThirdPartyCookies,TrackingProtection3pcd,ThirdPartyStoragePartitioning",
        ],
    )
    ctx.add_cookies(data)
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://colab.research.google.com/", wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(8000)
    body = page.evaluate("() => (document.body && document.body.innerText || '').slice(0, 4000)")
    mail = page.evaluate(
        "() => { const m = document.body.innerText.match(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/); return m ? m[0] : ''; }"
    )
    print("email", mail)
    print("signin_dialog", "You must be logged in" in body or "войдите" in body.lower())
    print("title", page.title()[:80])
    ctx.close()
print("profile", PROFILE)
