#!/usr/bin/env python3
import argparse
import atexit
import base64
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

SETUP = """!pip -q install numpy
!pip -q uninstall -y cupy-cuda12x || true
!pip -q install cupy-cuda13x || true
%cd /content
!rm -rf /content/DanyAI
!git clone -q https://github.com/DanyaVoredom/DanyAI.git /content/DanyAI
%cd /content/DanyAI
!pip -q install -e .
!nvidia-smi --query-gpu=name,memory.total --format=csv || true
print("setup done")"""

SETUP_TORCH = """!pip -q install numpy
%cd /content
!rm -rf /content/DanyAI
!git clone -q https://github.com/DanyaVoredom/DanyAI.git /content/DanyAI
%cd /content/DanyAI
!pip -q install -e .
!nvidia-smi --query-gpu=name,memory.total --format=csv || true
import torch
print("torch", torch.__version__, torch.cuda.is_available())
print("setup done")"""

SETUP_LORA = """!pip -q install -U transformers peft accelerate
!pip -q install flash-linear-attention 2>&1 | tail -2
!pip -q install --only-binary=:all: causal-conv1d 2>&1 | tail -1
!python -c "import fla; print('fla', fla.__version__)" || !pip uninstall -y -q flash-linear-attention
%cd /content
!rm -rf /content/DanyAI
!git clone -q https://github.com/DanyaVoredom/DanyAI.git /content/DanyAI
%cd /content/DanyAI
!pip -q install -e .
!nvidia-smi --query-gpu=name,memory.total --format=csv || true
import torch
print("torch", torch.__version__, torch.cuda.is_available())
print("setup done")"""

BUNDLE = """import base64, os, pathlib, urllib.request
try:
    os.chdir("/content/DanyAI")
except Exception:
    pass
data_raw = "__DATA_RAW__"
text = urllib.request.urlopen(data_raw).read().decode()
os.makedirs("data", exist_ok=True)
cur = None
buf = []
def flush():
    global cur, buf
    if cur is None:
        return
    b64 = "".join(buf)
    pathlib.Path("data", cur).write_bytes(base64.b64decode(b64))
    print("wrote", cur, len(b64) * 3 // 4)
    cur = None
    buf = []
for line in text.splitlines():
    if line.startswith("@@FILE "):
        flush()
        cur = line.split()[1]
    elif line.startswith("@@END"):
        flush()
    else:
        buf.append(line.strip())
flush()
print("data", sorted(os.listdir("data")))"""

BUNDLE_RELEASE = """import os, pathlib, urllib.request
try:
    os.chdir("/content/DanyAI")
except Exception:
    pass
base = "__BASE__"
for name in ["train.bin", "val.bin", "tokenizer.json", "meta.json", "sft.jsonl", "tools.jsonl"]:
    dst = pathlib.Path("data", name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(base + name, dst)
    print("got", name, dst.stat().st_size)
print("data", sorted(os.listdir("data")))"""

TRAIN = """import json, os
os.environ["DANYAI_GPU"] = "1"
import numpy as np
from danyai.backend import GPU
print("GPU", GPU)
cfg = json.load(open("__CONFIG__"))
cfg.update(block_size=__BLOCK__, batch_size=__BATCH__, steps=__STEPS__, warmup=__WARMUP__)
cfg["log_every"] = 1
json.dump(cfg, open("configs/colab.json", "w"), indent=2)
print(cfg, flush=True)
!DANYAI_GPU=1 python -m danyai.train --config configs/colab.json --data data --out __RUNS__ $( [ -f __RUNS__/ckpt.npz ] && echo --resume __RUNS__/ckpt.npz )
!DANYAI_GPU=1 python -m danyai.sample --ckpt __RUNS__/best.npz --tok data/tokenizer.json --prompt "Once upon a time" --tokens 120 --temp 0.8
print("train done")"""

SFT = """!DANYAI_GPU=1 python -m danyai.sft --ckpt __RUNS__/best.npz --data data/sft.jsonl --tok data/tokenizer.json --out __RUNS__/sft --steps 1200 --batch 8 --lr 1e-4 --block 256
!DANYAI_GPU=1 python -m danyai.sample --ckpt __RUNS__/sft/best.npz --tok data/tokenizer.json --prompt "<|user|>Write a short story about a cat.<|assistant|>" --tokens 120 --temp 0.7
print("sft done")"""

TRAIN_TORCH = """import json, os
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
cfg = json.load(open("__CONFIG__"))
cfg.update(block_size=__BLOCK__, batch_size=__BATCH__, steps=__STEPS__, warmup=__WARMUP__)
cfg["log_every"] = 10
json.dump(cfg, open("configs/colab.json", "w"), indent=2)
print(cfg, flush=True)
!python -m danyai.torch_train --config configs/colab.json --data data --out __RUNS__ $( [ -f __RUNS__/ckpt.npz ] && echo --resume __RUNS__/ckpt.npz )
!python -m danyai.sample --ckpt __RUNS__/best.npz --tok data/tokenizer.json --prompt "Once upon a time" --tokens 120 --temp 0.8
print("train done")"""

SFT_TORCH = """!python -m danyai.torch_sft --ckpt __RUNS__/best.npz --data data/sft.jsonl --tok data/tokenizer.json --out __RUNS__/sft --steps 1500 --batch 8 --lr 1e-4 --block 384
!python -m danyai.sample --ckpt __RUNS__/sft/best.npz --tok data/tokenizer.json --prompt "<|user|>Придумай короткую историю про кота.<|assistant|>" --tokens 120 --temp 0.7
print("sft done")"""

TRAIN_LORA = """!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u tools/finetune_lora.py --model __MODEL__ --data data/sft.jsonl,data/tools.jsonl --out __RUNS__/lora --max-len 512 --batch 2 --grad-accum 8 --grad-ckpt 0 --sort-window 128 --loss-chunk 96 --lr 1e-4 --warmup 30 --epochs 1 --eval-share 0.005 --eval-every 200 --eval-limit 32 --limit __LIMIT__ --snapshot 1 __RESUME__ --fallback Qwen/Qwen2.5-1.5B-Instruct
print("lora done")"""

PUSH_CELL = """import os, pathlib
p = pathlib.Path("/content/colab_push")
p.write_text('''__KEY__''')
os.chmod(p, 0o600)
print("push key ready", p.stat().st_size)"""

RESUME_CELL = """import shutil, urllib.request, zipfile
url = "__RESUME__"
zp = "/content/resume.zip"
urllib.request.urlretrieve(url, zp)
shutil.rmtree("/content/resume", ignore_errors=True)
with zipfile.ZipFile(zp) as z:
    z.extractall("/content/resume")
import os
print("resume files", sorted(os.listdir("/content/resume"))[:8])"""

PACK_LORA = """import base64, pathlib, time, uuid, urllib.request, zipfile
src = pathlib.Path("__RUNS__/lora")
zp = pathlib.Path("__RUNS__/lora.zip")
if not src.exists():
    print("@@MISSING __RUNS__/lora")
else:
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(src))
    data = zp.read_bytes()
    print("zip bytes", len(data), flush=True)

    def post(url, fields, field, filename, payload, timeout=600):
        boundary = "----danyai" + uuid.uuid4().hex
        body = b""
        for k, v in fields:
            body += ("--" + boundary + "\\r\\nContent-Disposition: form-data; name=\\"" + k + "\\"\\r\\n\\r\\n" + v + "\\r\\n").encode()
        body += ("--" + boundary + "\\r\\nContent-Disposition: form-data; name=\\"" + field + "\\"; filename=\\"" + filename + "\\"\\r\\nContent-Type: application/octet-stream\\r\\n\\r\\n").encode()
        body += payload + ("\\r\\n--" + boundary + "--\\r\\n").encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode(errors="replace").strip()

    urls = []
    try:
        import json as _json
        with urllib.request.urlopen("https://api.gofile.io/servers", timeout=60) as r:
            srv = _json.load(r)["data"]["servers"][0]["name"]
        u = post("https://%s.gofile.io/contents/uploadfile" % srv, [], "file", "lora.zip", data, timeout=1800)
        page = _json.loads(u).get("data", {}).get("downloadPage", "")
        if page.startswith("http"):
            urls.append(("gofile", page))
    except Exception as e:
        print("gofile fail", type(e).__name__, str(e)[:120], flush=True)
    for attempt in range(3):
        try:
            u = post("https://litterbox.catbox.moe/resources/internals/api.php",
                     [("reqtype", "fileupload"), ("time", "72h")], "fileToUpload", "lora.zip", data)
            if u.startswith("http"):
                urls.append(("litterbox", u))
                break
        except Exception as e:
            print("litterbox fail", type(e).__name__, str(e)[:120], flush=True)
            time.sleep(5 * (attempt + 1))
    if not urls:
        try:
            u = post("https://0x0.st", [], "file", "lora.zip", data)
            if u.startswith("http"):
                urls.append(("0x0", u.split()[0]))
        except Exception as e:
            print("0x0 fail", type(e).__name__, str(e)[:120], flush=True)
    if not urls:
        try:
            u = post("https://catbox.moe/user/api.php",
                     [("reqtype", "fileupload")], "fileToUpload", "lora.zip", data)
            if u.startswith("http"):
                urls.append(("catbox", u))
        except Exception as e:
            print("catbox fail", type(e).__name__, str(e)[:120], flush=True)
    if not urls and len(data) > 4 * 1024 * 1024:
        chunk = 8 * 1024 * 1024
        part = 0
        for i in range(0, len(data), chunk):
            blob = data[i:i + chunk]
            for attempt in range(3):
                try:
                    u = post("https://litterbox.catbox.moe/resources/internals/api.php",
                             [("reqtype", "fileupload"), ("time", "72h")], "fileToUpload",
                             "lora_%02d.bin" % part, blob)
                    if u.startswith("http"):
                        urls.append(("part%02d" % part, u))
                        break
                except Exception as e:
                    print("part fail", part, type(e).__name__, str(e)[:100], flush=True)
                    time.sleep(5)
            part += 1
    for host, u in urls:
        print("URL", host, u, flush=True)
    if not urls:
        b = base64.b64encode(data).decode()
        print("@@BEGIN __RUNS__/lora.zip " + str(len(b)))
        for i in range(0, len(b), 50000):
            print(b[i:i + 50000])
        print("@@END __RUNS__/lora.zip")
print("pack lora done")"""

PACK = """import base64, pathlib
for name in ["data/tokenizer.json", "data/meta.json", "configs/colab.json"]:
    p = pathlib.Path(name)
    if not p.exists():
        print("@@MISSING", name)
        continue
    b = base64.b64encode(p.read_bytes()).decode()
    print(f"@@BEGIN {name} {len(b)}")
    for i in range(0, len(b), 50000):
        print(b[i:i + 50000])
    print(f"@@END {name}")
import base64, pathlib
p = pathlib.Path("__RUNS__/best.npz")
if p.exists():
    b = base64.b64encode(p.read_bytes()).decode()
    print(f"@@BEGIN __RUNS__/best.npz {len(b)}")
    for i in range(0, len(b), 50000):
        print(b[i:i + 50000])
    print("@@END __RUNS__/best.npz")
else:
    print("@@MISSING __RUNS__/best.npz")
print("pack a done")"""

PACK_SFT = """import base64, pathlib
p = pathlib.Path("__RUNS__/sft/best.npz")
if p.exists():
    b = base64.b64encode(p.read_bytes()).decode()
    print(f"@@BEGIN __RUNS__/sft/best.npz {len(b)}")
    for i in range(0, len(b), 50000):
        print(b[i:i + 50000])
    print("@@END __RUNS__/sft/best.npz")
else:
    print("@@MISSING __RUNS__/sft/best.npz")
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
        nb["cells"].append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": lines,
            }
        )
    return nb


def build(train_cell, use_git, artifacts):
    cells = []
    if use_git:
        cells.append(SETUP)
    if artifacts:
        cells.append(train_cell + "\n\n" + PACK)
    else:
        cells.append(train_cell)
    return notebook(cells)


def publish(path, desc):
    out = subprocess.run(
        ["gh", "gist", "create", "--public", "--desc", desc, str(path)],
        capture_output=True,
        text=True,
        check=False,
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


CHROME_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-features=BlockThirdPartyCookies,TrackingProtection3pcd,ThirdPartyStoragePartitioning,PrivacySandboxSettings4",
    "--disable-site-isolation-trials",
]

SIGNIN_WORDS = (
    "Требуется вход",
    "sign-in required",
    "Sign in required",
    "Войдите в аккаунт",
)

GITHUB_WORDS = ("Запросить авторизацию", "Authorize with GitHub")


def prepare_profile(profile):
    pref = Path(profile) / "Default" / "Preferences"
    pref.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if pref.exists():
        try:
            data = json.loads(pref.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    prof = data.setdefault("profile", {})
    prof["cookie_controls_mode"] = 0
    prof["block_third_party_cookies"] = False
    prof["exit_type"] = "Normal"
    prof["exited_cleanly"] = True
    pref.write_text(json.dumps(data), encoding="utf-8")


def page_text(page):
    parts = []
    for frame in page.frames:
        try:
            parts.append(frame.inner_text("body"))
        except Exception:
            continue
    return "\n".join(parts)


def blocker(page):
    text = page_text(page)
    low = text.lower()
    for w in SIGNIN_WORDS:
        if w.lower() in low:
            return "signin"
    for w in GITHUB_WORDS:
        if w.lower() in low:
            return "github"
    if "Блокнот не найден" in text:
        return "missing"
    return None


def login_dialog(page):
    for frame in page.frames:
        try:
            if frame.evaluate(LOGIN_DLG_JS):
                return True
        except Exception:
            continue
    return False


def alive(ctx):
    try:
        return len(ctx.pages) > 0 and not ctx.pages[0].is_closed()
    except Exception:
        return False


def click_label(root, labels):
    for label in labels:
        try:
            loc = root.get_by_role("button", name=label, exact=True)
            if not loc.count():
                continue
            b = loc.first
            if b.is_visible() and b.is_enabled():
                b.click(timeout=3000)
                return label
        except Exception:
            continue
    return None


ACCEPT_JS = """() => {
  const labels = ['\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c', 'Run anyway', 'Run', '\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c \u0432\u0441\u0435', 'Run all'];
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const visible = b => {
    const r = b.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    const st = window.getComputedStyle(b);
    return st.display !== 'none' && st.visibility !== 'hidden';
  };
  const dlg = b => b.closest('[role=dialog], colab-dialog, .cdk-overlay-pane, mat-dialog-container, .dialog, md-dialog-container');
  let hit = null;
  for (const b of document.querySelectorAll('button, [role=button]')) {
    const t = norm(b.innerText || b.textContent);
    if (labels.includes(t) && visible(b) && dlg(b)) { hit = b; break; }
  }
  if (!hit) {
    for (const b of document.querySelectorAll('button, [role=button]')) {
      const t = norm(b.innerText || b.textContent);
      if (labels.includes(t) && visible(b)) { hit = b; break; }
    }
  }
  if (!hit) return null;
  hit.click();
  return norm(hit.innerText || hit.textContent);
}"""

CONNECT_JS = """() => {
  const labels = ['\u041f\u043e\u0434\u043a\u043b\u044e\u0447\u0438\u0442\u044c\u0441\u044f', 'Connect'];
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  for (const b of document.querySelectorAll('button, [role=button], [role=menuitem]')) {
    const t = norm(b.innerText || b.textContent);
    if (!labels.includes(t)) continue;
    const r = b.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    b.click();
    return t;
  }
  return null;
}"""


def frame_call(page, js):
    for frame in page.frames:
        try:
            res = frame.evaluate(js)
        except Exception:
            continue
        if res:
            return res
    return None


SIGNIN_JS = """() => {
  const t = (document.body.innerText || "").toLowerCase();
  if (t.indexOf("sign-in required") >= 0) return true;
  if (t.indexOf("\u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044f \u0432\u0445\u043e\u0434") >= 0) return true;
  for (const a of document.querySelectorAll("a[href*=\"accounts.google.com\"]")) {
    const r = a.getBoundingClientRect();
    if (r.width && r.height) return true;
  }
  return false;
}"""


def signed_out(page):
    for frame in page.frames:
        try:
            if frame.evaluate(SIGNIN_JS):
                return True
        except Exception:
            continue
    return False


LOGIN_DLG_JS = """() => {
  const t = (document.body.innerText || "").toLowerCase();
  if (t.indexOf("\\u0432\\u043e\\u0439\\u0434\\u0438\\u0442\\u0435 \\u0432 \\u0430\\u043a\\u043a\\u0430\\u0443\\u043d\\u0442 google") < 0) return false;
  for (const d of document.querySelectorAll("dialog, [role=dialog], md-dialog, .cdk-overlay-pane")) {
    const r = d.getBoundingClientRect();
    if (r.width > 150 && r.height > 80) return true;
  }
  return false;
}"""


HAS_DLG_JS = """() => {
  for (const d of document.querySelectorAll("dialog, [role=dialog]")) {
    const r = d.getBoundingClientRect();
    if (r.width > 200 && r.height > 100) return true;
  }
  return false;
}"""

PICK_BTNS_JS = """() => {
  let dlg = null;
  for (const d of document.querySelectorAll("dialog, [role=dialog]")) {
    const r = d.getBoundingClientRect();
    if (r.width > 200 && r.height > 100) { dlg = r; break; }
  }
  if (!dlg) return null;
  const cands = [];
  const walk = root => {
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) walk(el.shadowRoot);
      if (el.tagName !== "BUTTON" && el.getAttribute("role") !== "button") continue;
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      const cx = r.x + r.width / 2;
      const cy = r.y + r.height / 2;
      if (cx < dlg.x || cx > dlg.x + dlg.width) continue;
      if (cy < dlg.y || cy > dlg.y + dlg.height) continue;
      const st = window.getComputedStyle(el);
      if (st.display === "none" || st.visibility === "hidden") continue;
      cands.push({x: cx, y: cy, rx: r.x});
    }
  };
  walk(document);
  if (!cands.length) {
    return [{x: dlg.x + dlg.width * 0.88, y: dlg.y + dlg.height * 0.81, rx: 0}];
  }
  cands.sort((a, b) => a.rx - b.rx);
  return cands;
}"""


CLICK_LABEL_JS = r"""() => {
  const labels = ['\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c', 'Run anyway', '\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c \u0432\u0441\u0451', '\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c \u0432\u0441\u0435', 'Run all', 'Run'];
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  let hit = null;
  const walk = root => {
    if (hit) return;
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) walk(el.shadowRoot);
      if (hit) return;
      if (!el.tagName.endsWith('BUTTON')) continue;
      const t = norm(el.innerText || el.textContent);
      if (!labels.includes(t)) continue;
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      if (r.y < 160) continue;
      const st = window.getComputedStyle(el);
      if (st.display === 'none' || st.visibility === 'hidden' || st.pointerEvents === 'none') continue;
      hit = el;
      return;
    }
  };
  walk(document);
  if (!hit) return null;
  const label = norm(hit.innerText || hit.textContent);
  hit.click();
  return label;
}"""


CELLS_JS = r"""() => {
  const out = [];
  document.querySelectorAll('.cell.code').forEach(c => {
    const ta = c.querySelector('textarea.inputarea');
    if (!ta) return;
    const v = (ta.value || ta.textContent || '').trim();
    if (v) out.push(v);
  });
  return out;
}"""

CELLS_TA_JS = r"""() => {
  const out = [];
  document.querySelectorAll('textarea.inputarea').forEach(ta => {
    let root = ta;
    while (root && root.classList && !root.classList.contains('cell')) root = root.parentElement;
    if (root && root.classList.contains('markdown')) return;
    const v = (ta.value || ta.textContent || '').trim();
    if (v) out.push(v);
  });
  return out;
}"""

RUN_ALL_JS = r"""() => {
  const labels = ['\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c \u0432\u0441\u0435', 'Run all', '\u0412\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c \u0432\u0441\u0451', 'Run all cells'];
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  let hit = null;
  const walk = root => {
    if (hit) return;
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) walk(el.shadowRoot);
      if (hit) return;
      if (!el.tagName.endsWith('BUTTON')) continue;
      if (!labels.includes(norm(el.innerText || el.textContent))) continue;
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      hit = el;
      return;
    }
  };
  walk(document);
  if (!hit) return null;
  hit.click();
  return 'run all';
}"""


def run_all(page):
    for frame in page.frames:
        try:
            hit = frame.evaluate(RUN_ALL_JS)
        except Exception:
            continue
        if hit:
            return True
    return False


def accept_run(page, wait=20.0):
    end = time.time() + wait
    while time.time() < end:
        for frame in page.frames:
            try:
                hit = frame.evaluate(CLICK_LABEL_JS)
            except Exception:
                continue
            if hit:
                log("accepted " + hit)
                page.wait_for_timeout(2000)
                return True
        for frame in page.frames:
            try:
                if not frame.evaluate(HAS_DLG_JS):
                    continue
                cands = frame.evaluate(PICK_BTNS_JS)
            except Exception:
                continue
            if not cands:
                continue
            hit = cands[-1]
            try:
                page.mouse.click(hit["x"], hit["y"])
            except Exception:
                continue
            page.wait_for_timeout(2000)
            try:
                if not frame.evaluate(HAS_DLG_JS):
                    log("confirmed dialog")
                    return True
            except Exception:
                log("confirmed dialog")
                return True
        page.wait_for_timeout(500)
    return False


def ensure_connected(page, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if frame_call(page, CONNECT_JS):
            log("connect requested")
            page.wait_for_timeout(5000)
            continue
        try:
            if page.locator(".cell.running").count() > 0:
                return True
            if page.locator("#connect-button").count() == 0:
                return True
        except Exception:
            pass
        page.wait_for_timeout(2000)
    return False


OUTPUT_JS = """() => {
  const skip = '.CodeMirror, .cm-editor, .inputarea, .markdown, .editor, .monaco-editor';
  const out = [];
  document.querySelectorAll('pre').forEach(p => {
    if (p.closest(skip)) return;
    const st = window.getComputedStyle(p);
    if (st.display === 'none' || st.visibility === 'hidden') return;
    const t = p.innerText || p.textContent || '';
    if (t.trim()) out.push(t);
  });
  return out.join('\\n');
}"""


def frames_text(page):
    parts = []
    for frame in page.frames:
        try:
            t = frame.evaluate(OUTPUT_JS)
        except Exception:
            continue
        if t:
            parts.append(t)
    return "\n".join(parts)


def body_text(page):
    try:
        return page.evaluate(
            "() => (document.body && document.body.innerText || '').slice(0, 6000)"
        )
    except Exception:
        return ""


def editor_text(page):
    vals = frame_call(page, CELLS_JS) or []
    if vals:
        return "\n".join(vals)
    return "\n".join(frame_call(page, CELLS_TA_JS) or [])


def run_started(page):
    return (
        running(page)
        or bool(editor_text(page).strip())
        or "setup done" in frames_text(page)
    )


def wait_ready(page, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if (
                page.locator("textarea.inputarea").count() > 0
                and page.locator(".cell").count() > 0
            ):
                return True
        except Exception:
            pass
        page.wait_for_timeout(1500)
    return False


def dismiss_warnings(page, rounds=8):
    for _ in range(rounds):
        found = False
        for frame in page.frames:
            try:
                hit = frame.evaluate(CLICK_LABEL_JS)
            except Exception:
                continue
            if hit:
                log("dismissed " + hit)
                found = True
                page.wait_for_timeout(1500)
                break
        if not found:
            return True
    return False


GPU_BLOCK = (
    "due to usage limits",
    "из-за лимитов",
    "limits on usage",
    "Connect without GPU",
    "Подключиться без ускорителя",
    "No GPU backend",
)


def stream(page, ctx, stop_when, limit, dump):
    end = time.time() + limit
    prev = ""
    marker_at = None
    err_at = None
    quota_at = None
    started_at = time.time()
    while time.time() < end:
        if not alive(ctx):
            return False
        if not prev and time.time() - started_at > 600:
            log("no runtime output, assume gpu quota")
            return "quota"
        try:
            if quota_at is None:
                body = body_text(page)
                if any(m in body for m in GPU_BLOCK):
                    quota_at = time.time()
                    log("gpu quota dialog")
            if quota_at and time.time() - quota_at > 20:
                return "quota"
            text = frames_text(page)
            if text != prev:
                common = 0
                for a, b in zip(prev, text):
                    if a != b:
                        break
                    common += 1
                for line in text[common:].splitlines():
                    if not line.startswith("@@") and len(line) < 400:
                        log(line)
                prev = text
                dump.parent.mkdir(parents=True, exist_ok=True)
                dump.write_text(text, encoding="utf-8")
            dismiss_warnings(page)
            if stop_when in text and marker_at is None:
                marker_at = time.time()
                log(f"marker: {stop_when}")
            try:
                running = page.locator(".cell.running").count()
            except Exception:
                running = 1
            if marker_at and running == 0 and time.time() - marker_at > 8:
                return True
            if "Traceback (most recent call last)" in text and running == 0:
                if err_at is None:
                    err_at = time.time()
                elif time.time() - err_at > 30:
                    log("error marker in notebook output")
                    return False
            else:
                err_at = None
            page.wait_for_timeout(3000)
        except Exception as e:
            if not alive(ctx):
                return False
            log(f"stream: {type(e).__name__}")
            page.wait_for_timeout(2000)
    return stop_when in prev


def extract(text, out):
    saved = []
    for m in re.finditer(r"@@BEGIN (\S+) (\d+)\n(.*?)@@END \1", text, re.S):
        name, size, payload = m.group(1), int(m.group(2)), m.group(3)
        b64 = "".join(payload.split())
        if len(b64) < size:
            log(f"artifact {name}: truncated {len(b64)}/{size}")
            continue
        parts = Path(name).parts
        rel = Path(parts[-2]) / parts[-1] if len(parts) >= 2 else Path(parts[-1])
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(b64[:size]))
        saved.append((str(dest), len(b64) * 3 // 4))
    return saved


FIND_TEXT_JS = r"""(labels) => {
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  const res = [];
  const walk = root => {
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) walk(el.shadowRoot);
      if (el.childElementCount) continue;
      if (!labels.includes(norm(el.textContent))) continue;
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      res.push({x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2)});
    }
  };
  walk(document);
  return res;
}"""


def _find_chrome():
    if sys.platform != "win32":
        env = os.environ.get("PLAYWRIGHT_CHROMIUM")
        if env and Path(env).exists():
            return env
        for fixed in ("/usr/bin/google-chrome", "/opt/google/chrome/chrome"):
            if Path(fixed).exists():
                return fixed
        for base in (Path.home() / ".cache" / "ms-playwright").glob(
            "chromium*/chrome-linux*/chrome"
        ):
            return str(base)
        return None
    roots = [
        os.environ.get("PROGRAMFILES", ""),
        os.environ.get("PROGRAMFILES(X86)", ""),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    for root in roots:
        if not root:
            continue
        exe = Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
        if exe.exists():
            return str(exe)
    return None


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pids(profile):
    path = str(profile).replace("'", "''")
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
        "Where-Object { $_.CommandLine -and $_.CommandLine.Contains('"
        + path
        + "') } | "
        "ForEach-Object { $_.ProcessId }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except Exception:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def kill_profile(profile):
    for pid in _pids(profile):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False
        )


def _spawn(profile, headed, port):
    exe = _find_chrome()
    if not exe:
        return None, None
    kill_profile(profile)
    cmd = [
        exe,
        "--user-data-dir=" + str(profile),
        "--remote-debugging-port=" + str(port),
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-session-crashed-bubble",
        "--hide-crash-restore-bubble",
        "--disable-blink-features=AutomationControlled",
        "--disable-features=BlockThirdPartyCookies,TrackingProtection3pcd,ThirdPartyStoragePartitioning",
        "--window-size=1440,960",
    ]
    if not headed:
        cmd.append("--headless=new" if sys.platform == "win32" else "--headless")
    cmd.append("--no-sandbox" if sys.platform != "win32" else "about:blank")
    if sys.platform != "win32":
        cmd.append("about:blank")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    endpoint = "http://127.0.0.1:%d" % port
    for _ in range(60):
        try:
            urllib.request.urlopen(endpoint + "/json/version", timeout=2).read()
            return endpoint, proc
        except Exception:
            if proc.poll() is not None:
                return None, None
            time.sleep(1)
    return None, proc


def connect(p, profile, headed):
    external = os.environ.get("COLAB_CDP_ENDPOINT")
    if external:
        try:
            browser = p.chromium.connect_over_cdp(external)
            return browser.contexts[0] if browser.contexts else browser.new_context(), None
        except Exception:
            port = int(external.rsplit(":", 1)[-1])
            endpoint, proc = _spawn(profile, headed, port)
            if endpoint:
                browser = p.chromium.connect_over_cdp(endpoint)
                return (
                    browser.contexts[0] if browser.contexts else browser.new_context(),
                    proc,
                )
            if proc:
                proc.terminate()
            raise RuntimeError("cannot start chrome with profile " + str(profile))
    endpoint, proc = _spawn(profile, headed, _free_port())
    if endpoint:
        browser = p.chromium.connect_over_cdp(endpoint)
        return browser.contexts[0] if browser.contexts else browser.new_context(), proc
    if proc:
        proc.terminate()
    return p.chromium.launch_persistent_context(
        user_data_dir=str(profile),
        headless=not headed,
        viewport={"width": 1440, "height": 960},
        args=CHROME_ARGS,
    ), None


def teardown(ctx, proc, profile=None):
    try:
        ctx.close()
    except Exception:
        pass
    if profile:
        kill_profile(profile)
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        except Exception:
            pass


def find_text(page, labels):
    for frame in page.frames:
        try:
            hits = frame.evaluate(FIND_TEXT_JS, labels)
        except Exception:
            continue
        if hits:
            return hits[0]
    return None


def running(page):
    try:
        return page.locator(".cell.running").count() > 0
    except Exception:
        return False


def start_run(page, first_cell=""):
    if run_all(page):
        page.wait_for_timeout(3000)
        if accept_run(page, wait=10.0):
            log("run all clicked")
            return True
        if not first_cell and running(page):
            log("run all clicked")
            return True
    page.keyboard.press("Control+F9")
    page.wait_for_timeout(2500)
    if accept_run(page, wait=8.0):
        return True
    text = editor_text(page)
    if (
        first_cell
        and first_cell.strip()
        and first_cell.strip()[:60] not in text
        and not running(page)
    ):
        return False
    if running(page):
        return True
    menu = find_text(page, ["Runtime", "Среда выполнения"])
    if menu:
        page.mouse.click(menu["x"], menu["y"])
        page.wait_for_timeout(1800)
        item = find_text(page, ["Run all", "Выполнить все"])
        if item:
            page.mouse.click(item["x"], item["y"])
            page.wait_for_timeout(2500)
            if accept_run(page, wait=15.0):
                return True
    if accept_run(page, wait=10.0):
        return True
    return run_started(page)


def make_data_bundle(data_dir, out_path):
    files = ["train.bin", "val.bin", "tokenizer.json", "meta.json", "sft.jsonl"]
    with out_path.open("w", encoding="utf-8") as f:
        for name in files:
            p = Path(data_dir) / name
            if not p.exists():
                log(f"bundle: missing {name}")
                continue
            b = base64.b64encode(p.read_bytes()).decode()
            f.write(f"@@FILE {name} {len(b)}\n")
            for i in range(0, len(b), 76):
                f.write(b[i : i + 76] + "\n")
            f.write("@@END " + name + "\n")
    log(f"bundle {out_path} {out_path.stat().st_size} bytes")


def gist_parts(gist_url):
    m = re.search(r"gist\.github\.com/([^/]+)/([0-9a-f]+)", gist_url)
    if not m:
        raise RuntimeError(f"cannot parse gist url: {gist_url}")
    return m.group(1), m.group(2)


def blocked_sessions(page):
    try:
        if page.get_by_role("button", name="Manage sessions").count():
            return True
    except Exception:
        pass
    js = r"""() => {
      const out = [];
      const walk = (root) => {
        if (!root) return;
        out.push(root.innerText || '');
        const els = root.querySelectorAll ? root.querySelectorAll('*') : [];
        for (const el of els) {
          if (el.shadowRoot) walk(el.shadowRoot);
        }
      };
      walk(document.body);
      return out.join('\n').toLowerCase();
    }"""
    for frame in page.frames:
        try:
            t = frame.evaluate(js)
        except Exception:
            continue
        if "too many" in t or "maximum number of sessions" in t:
            return True
    return False


def close_dialogs(page):
    js = r"""() => {
      const labels = ['Отмена', 'Cancel', 'Закрыть', 'Close'];
      for (const b of document.querySelectorAll('button, [role=button]')) {
        const t = (b.innerText || b.textContent || '').trim();
        if (!labels.includes(t)) continue;
        const r = b.getBoundingClientRect();
        if (r.width && r.height) { b.click(); return t; }
      }
      return null;
    }"""
    for frame in page.frames:
        try:
            hit = frame.evaluate(js)
        except Exception:
            continue
        if hit:
            return hit
    return None


def wait_for_slot(page, timeout, check_every=60):
    end = time.time() + timeout
    attempt = 0
    while time.time() < end:
        attempt += 1
        if not blocked_sessions(page):
            return True
        log(f"session limit hit, closing dialog and retrying ({attempt})")
        close_dialogs(page)
        page.wait_for_timeout(3000)
        frame_call(page, CONNECT_JS)
        page.wait_for_timeout(check_every * 1000)
    return not blocked_sessions(page)


def free_sessions(page):
    js = r"""() => {
      const labels = ['Manage sessions', 'Управление сеансами', 'Управление сессиями'];
      for (const b of document.querySelectorAll('button, [role=button], a')) {
        const t = (b.innerText || b.textContent || '').trim();
        if (!labels.includes(t)) continue;
        const r = b.getBoundingClientRect();
        if (r.width && r.height) { b.click(); return t; }
      }
      return null;
    }"""
    opened = None
    for frame in page.frames:
        try:
            opened = frame.evaluate(js)
        except Exception:
            continue
        if opened:
            break
    if not opened:
        return False
    page.wait_for_timeout(4000)
    killed = 0
    for _ in range(10):
        loc = page.locator('button[aria-label*="Terminate"]')
        n = loc.count()
        if n == 0:
            break
        try:
            loc.first.click()
            killed += 1
        except Exception:
            break
        page.wait_for_timeout(4000)
    close = page.get_by_role("button", name="Close")
    if close.count():
        try:
            close.last.click()
        except Exception:
            pass
    page.wait_for_timeout(2000)
    if killed:
        log("terminated %d colab sessions" % killed)
    return killed > 0


def start_with_slot(page, first_cell, timeout, check_every=180):
    end = time.time() + timeout
    attempt = 0
    while time.time() < end:
        attempt += 1
        if blocked_sessions(page):
            if attempt % 5 == 1:
                log("session limit, freeing colab sessions")
            if free_sessions(page):
                page.wait_for_timeout(8000)
                frame_call(page, CONNECT_JS)
                continue
            close_dialogs(page)
            page.wait_for_timeout(check_every * 1000)
            frame_call(page, CONNECT_JS)
            continue
        if start_run(page, first_cell):
            page.wait_for_timeout(8000)
            if blocked_sessions(page):
                close_dialogs(page)
                page.wait_for_timeout(check_every * 1000)
                continue
            return True
        log("run all retry %d" % attempt)
        page.wait_for_timeout(5000)
    return False


def repo_slug():
    r = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True
    )
    url = r.stdout.strip()
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    if not m:
        raise RuntimeError("cannot parse origin url: " + url)
    return m.group(1)


def upload_data_release(tag, data_dir, names):
    paths = [str(Path(data_dir) / n) for n in names if (Path(data_dir) / n).exists()]
    if not paths:
        raise RuntimeError("no data files to upload")
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            tag,
            "--title",
            tag,
            "--notes",
            "danyai corpus bundle",
            "--latest=false",
        ],
        capture_output=True,
        check=False,
    )
    r = subprocess.run(
        ["gh", "release", "upload", tag] + paths + ["--clobber"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "release upload failed")
    log(f"uploaded {len(paths)} files to release {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--notebook", default="colab/train_danyai.ipynb")
    ap.add_argument(
        "--profile",
        default=str(
            Path(os.environ.get("LOCALAPPDATA", ".")) / "DanyAI" / "colab-profile"
        ),
    )
    ap.add_argument("--out", default="runs/mini")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--data", default="data")
    ap.add_argument("--config", default="configs/modern.json")
    ap.add_argument("--runs", default="/content/danyai_runs")
    ap.add_argument("--data-release", default="")
    ap.add_argument(
        "--trainer", default="autograd", choices=["autograd", "torch", "lora"]
    )
    ap.add_argument("--timeout", type=int, default=21600)
    ap.add_argument("--connect-wait", type=int, default=5400)
    ap.add_argument("--ready-wait", type=int, default=180)
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--dump", default="runs/colab_output.txt")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--lora-limit", type=int, default=0)
    ap.add_argument("--resume-url", default="")
    ap.add_argument("--channel", default="chrome")
    ap.add_argument("--login", action="store_true")
    args = ap.parse_args()

    if args.login:
        log("opening profile for google sign-in")
        with sync_playwright() as p:
            ctx, proc = connect(p, args.profile, True)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(
                "https://accounts.google.com",
                wait_until="domcontentloaded",
                timeout=60000,
            )
            log("sign in then close browser")
            try:
                while alive(ctx):
                    page.wait_for_timeout(5000)
            except Exception:
                pass
            teardown(ctx, proc, args.profile)
        return 0

    out = Path(args.out)
    nb_path = Path(args.notebook)
    out.mkdir(parents=True, exist_ok=True)

    if args.url:
        url = args.url
    else:
        names = ["train.bin", "val.bin", "tokenizer.json", "meta.json", "sft.jsonl"]
        if args.trainer == "lora":
            names.append("tools.jsonl")
        if args.data_release:
            repo = repo_slug()
            base = f"https://github.com/{repo}/releases/download/{args.data_release}/"
            upload_data_release(args.data_release, args.data, names)
            log(f"data release {base}")
            bundle_cell = BUNDLE_RELEASE.replace("__BASE__", base)
        else:
            bundle_path = out / "danyai_data.bundle"
            make_data_bundle(args.data, bundle_path)
            gist_data = publish(bundle_path, "danyai data bundle")
            user, gid = gist_parts(gist_data)
            data_raw = f"https://gist.githubusercontent.com/{user}/{gid}/raw/danyai_data.bundle"
            log(f"data raw {data_raw}")
            bundle_cell = BUNDLE.replace("__DATA_RAW__", data_raw)
        if args.trainer == "lora":
            resume = ""
            if args.resume_url:
                resume = "/content/resume"
            train = (
                TRAIN_LORA.replace("__MODEL__", args.model)
                .replace("__RUNS__", args.runs)
                .replace("__LIMIT__", str(args.lora_limit))
                .replace(
                    "__RESUME__",
                    f"--resume-adapter {resume}" if resume else "",
                )
            )
            cells = [SETUP_LORA, bundle_cell]
            key_path = Path("/home/lipton/.ssh/colab_push")
            if key_path.exists():
                cells.append(PUSH_CELL.replace("__KEY__", key_path.read_text().strip()))
            if args.resume_url:
                cells.append(RESUME_CELL.replace("__RESUME__", args.resume_url))
            cells.append(train)
            if not args.no_artifacts:
                cells.append(PACK_LORA.replace("__RUNS__", args.runs))
        else:
            tpl = TRAIN_TORCH if args.trainer == "torch" else TRAIN
            sft_tpl = SFT_TORCH if args.trainer == "torch" else SFT
            train = (
                tpl.replace("__CONFIG__", args.config)
                .replace("__RUNS__", args.runs)
                .replace("__BLOCK__", str(args.block))
                .replace("__BATCH__", str(args.batch))
                .replace("__STEPS__", str(args.steps))
                .replace("__WARMUP__", str(args.warmup))
            )
            sft_cell = sft_tpl.replace("__RUNS__", args.runs)
            setup_cell = SETUP_TORCH if args.trainer == "torch" else SETUP
            cells = [setup_cell, bundle_cell, train, sft_cell]
            if not args.no_artifacts:
                cells.append(PACK.replace("__RUNS__", args.runs))
                cells.append(PACK_SFT.replace("__RUNS__", args.runs))
        nb = notebook(cells)
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        nb_path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        gist = publish(nb_path, "danyai colab run")
        log(f"gist {gist}")
        url = gist_colab_url(gist)

    log(f"open {url}")

    with sync_playwright() as p:
        prepare_profile(args.profile)
        ctx, proc = connect(p, args.profile, args.headed)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        opened = False
        for attempt in range(4):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=120000)
                opened = True
                break
            except Exception as e:
                log(f"open retry {attempt + 1}: {type(e).__name__}")
                page.wait_for_timeout(15000)
        if not opened:
            log("error: cannot open notebook")
            teardown(ctx, proc, args.profile)
            return 1
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
            teardown(ctx, proc, args.profile)
            return 1
        log("notebook ready")
        if signed_out(page) or login_dialog(page):
            log("error: google sign-in required in profile " + args.profile)
            if not args.headed:
                teardown(ctx, proc, args.profile)
                return 2
            log("browser open, sign in then press Ctrl+C")
            try:
                while alive(ctx):
                    page.wait_for_timeout(5000)
            except KeyboardInterrupt:
                pass
            teardown(ctx, proc, args.profile)
            return 2
        page.wait_for_timeout(1500)
        try:
            page.locator(".cell").first.click(timeout=5000)
            page.keyboard.press("Escape")
        except Exception:
            pass
        page.wait_for_timeout(500)
        if blocked_sessions(page):
            log(f"colab session limit, waiting up to {args.connect_wait}s")
            close_dialogs(page)
            if not wait_for_slot(page, args.connect_wait):
                log("error: could not free a colab session slot")
                teardown(ctx, proc, args.profile)
                return 4
            log("session slot free")
            page.reload(wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(6000)
        ensure_connected(page, 30)
        first_cell = editor_text(page).splitlines()[:1]
        first_cell = first_cell[0].strip() if first_cell else ""
        confirmed = start_with_slot(page, first_cell, args.connect_wait)
        log("run all started" if confirmed else "warn: run all not confirmed")
        if not confirmed:
            try:
                page.screenshot(path=str(Path(args.dump).with_suffix(".png")))
            except Exception:
                pass

        stop = "pack done" if not args.no_artifacts else "train done"
        ok = stream(page, ctx, stop, args.timeout, Path(args.dump))
        if ok == "quota":
            log("gpu quota exhausted")
            teardown(ctx, proc, args.profile)
            return 6
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
        teardown(ctx, proc, args.profile)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
