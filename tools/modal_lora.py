import json
import pathlib
import subprocess
import urllib.request
import zipfile

import modal

REPO = "https://github.com/DanyaVoredom/DanyAI.git"
BRANCH = "dev"
DATA_RELEASE = "https://github.com/DanyaVoredom/DanyAI/releases/download/data-v2/"
DATA_FILES = ["sft.jsonl", "tools.jsonl"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("numpy", "torch", "transformers", "peft", "accelerate")
    .run_commands(
        f"git clone -q --branch {BRANCH} {REPO} /root/DanyAI",
        "cd /root/DanyAI && pip install -q -e .",
    )
    .workdir("/root/DanyAI")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("danyai-lora")
data_vol = modal.Volume.from_name("danyai-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("danyai-runs", create_if_missing=True)
hf_vol = modal.Volume.from_name("danyai-hf", create_if_missing=True)


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def fetch_data(root="/data"):
    base = pathlib.Path(root)
    base.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        dst = base / name
        if dst.exists() and dst.stat().st_size > 0:
            print("cached", name, dst.stat().st_size, flush=True)
            continue
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(DATA_RELEASE + name, tmp)
        tmp.rename(dst)
        print("got", name, dst.stat().st_size, flush=True)


def build_data(root="/data"):
    out = pathlib.Path(root) / "lora_train.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as w:
        for name in DATA_FILES:
            with open(pathlib.Path(root) / name, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except Exception:
                        continue
                    w.write(line + "\n")
                    n += 1
    print("rows", n, flush=True)
    return out


@app.function(image=image, gpu="A10G", timeout=21600,
              volumes={"/data": data_vol, "/runs": runs_vol, "/hf": hf_vol})
def lora(model: str = "huihui-ai/Huihui-Qwen3.5-2B-abliterated",
         max_len: int = 512, batch: int = 2, grad_accum: int = 8,
         epochs: float = 1.0, limit: int = 0, lr: float = 1e-4,
         out: str = "lora", sft_only: bool = False):
    fetch_data()
    data_vol.commit()
    if sft_only:
        path = pathlib.Path("/data/sft.jsonl")
    else:
        path = build_data()

    outdir = pathlib.Path("/runs") / out
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", "-m", "tools.finetune_lora", "--model", model,
           "--data", str(path), "--out", str(outdir),
           "--max-len", str(max_len), "--batch", str(batch),
           "--grad-accum", str(grad_accum), "--epochs", str(epochs),
           "--lr", str(lr), "--lora-r", "16", "--lora-alpha", "32"]
    if limit:
        cmd += ["--limit", str(limit)]
    run(cmd)
    runs_vol.commit()
    hf_vol.commit()

    archive = pathlib.Path("/runs") / f"{out}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(outdir))
    runs_vol.commit()
    print("archive", archive, archive.stat().st_size, flush=True)
    return {"out": str(outdir), "zip": str(archive), "bytes": archive.stat().st_size}


@app.local_entrypoint()
def main(out: str = "lora", epochs: float = 1.0, limit: int = 0,
         max_len: int = 512, batch: int = 2, sft_only: bool = False):
    print(lora.remote(out=out, epochs=epochs, limit=limit, max_len=max_len,
                      batch=batch, sft_only=sft_only))
