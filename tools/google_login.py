import os
import sys
import time

from playwright.sync_api import sync_playwright

EMAIL = sys.argv[1]
PASSWORD = sys.argv[2]
PROFILE = sys.argv[3] if len(sys.argv) > 3 else "/home/lipton/DanyAI/colab-profile-g2"

exe = None
for cand in ("/usr/bin/google-chrome", "/opt/google/chrome/chrome"):
    if os.path.exists(cand):
        exe = cand
        break


def click_any(page, labels):
    for lab in labels:
        try:
            loc = page.get_by_role("button", name=lab, exact=False)
            if loc.count():
                loc.first.click(timeout=4000)
                return lab
        except Exception:
            continue
    for lab in labels:
        try:
            loc = page.get_by_text(lab, exact=True)
            if loc.count():
                loc.first.click(timeout=4000)
                return lab
        except Exception:
            continue
    return ""


with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        PROFILE,
        headless=True,
        executable_path=exe,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--lang=en-US",
        ],
        locale="en-US",
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(
        "https://accounts.google.com/ServiceLogin?service=colab&continue=https://colab.research.google.com/",
        wait_until="domcontentloaded",
        timeout=120000,
    )
    page.wait_for_timeout(4000)
    print("step1 url", page.url[:80])
    try:
        page.fill("input[type=email]", EMAIL, timeout=20000)
    except Exception:
        try:
            page.fill("#identifierId", EMAIL, timeout=10000)
        except Exception as e:
            print("email fill failed", type(e).__name__)
    print("clicked", click_any(page, ["Next", "Далее", "Дальше"]))
    page.wait_for_timeout(6000)
    print("step2 url", page.url[:80])
    try:
        page.fill("input[type=password]", PASSWORD, timeout=20000)
        print("pass filled")
    except Exception as e:
        print("pass fill failed", type(e).__name__)
        body = page.evaluate("() => (document.body && document.body.innerText || '').slice(0,600)")
        print("body", " | ".join(body.splitlines())[:400])
    print("clicked2", click_any(page, ["Next", "Далее", "Sign in", "Войти"]))
    page.wait_for_timeout(12000)
    print("final url", page.url[:100])
    body = page.evaluate("() => (document.body && document.body.innerText || '').slice(0, 1500)")
    print("body", " | ".join(body.splitlines())[:700])
    page.screenshot(path="/home/lipton/DanyAI/runs/login.png")
    ctx.close()
