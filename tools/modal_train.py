import json
import os
import pathlib
import subprocess
import urllib.request

import modal

REPO = "https://github.com/DanyaVoredom/DanyAI.git"
BRANCH = "dev"
DATA_RELEASE = "https://github.com/DanyaVoredom/DanyAI/releases/download/data-v2/"
DATA_FILES = ["train.bin", "val.bin", "tokenizer.json", "meta.json", "sft.jsonl", "tools.jsonl"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("numpy", "torch")
    .run_commands(
        f"git clone -q --branch {BRANCH} {REPO} /root/DanyAI",
        "cd /root/DanyAI && pip install -q -e .",
    )
    .workdir("/root/DanyAI")
)

app = modal.App("danyai-train")
data_vol = modal.Volume.from_name("danyai-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("danyai-runs", create_if_missing=True)


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


@app.function(image=image, gpu="T4", timeout=14400, volumes={"/data": data_vol, "/runs": runs_vol})
def train(block: int = 256, batch: int = 32, steps: int = 6000, warmup: int = 300,
          sft_steps: int = 1500, sft_batch: int = 8, sft_block: int = 384,
          lr: float = 0.0, out: str = "big2", resume: bool = True):
    fetch_data()
    data_vol.commit()

    cfg = json.load(open("configs/big.json"))
    cfg.update(block_size=block, batch_size=batch, steps=steps, warmup=warmup, log_every=10)
    json.dump(cfg, open("configs/modal.json", "w"), indent=2)

    outdir = pathlib.Path("/runs") / out
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", "-m", "danyai.torch_train", "--config", "configs/modal.json",
           "--data", "/data", "--out", str(outdir)]
    if lr:
        cmd += ["--lr", str(lr)]
    ckpt = outdir / "ckpt.npz"
    if resume and ckpt.exists():
        cmd += ["--resume", str(ckpt)]
    run(cmd)
    runs_vol.commit()

    run(["python", "-m", "danyai.torch_sft", "--ckpt", str(outdir / "best.npz"),
         "--data", "/data/sft.jsonl", "--tok", "/data/tokenizer.json",
         "--out", str(outdir / "sft"), "--steps", str(sft_steps),
         "--batch", str(sft_batch), "--block", str(sft_block)])
    runs_vol.commit()

    run(["python", "-m", "danyai.sample", "--ckpt", str(outdir / "sft" / "best.npz"),
         "--tok", "/data/tokenizer.json", "--prompt", "<|user|>Кто ты?<|assistant|>",
         "--tokens", "80", "--temp", "0.7"])
    runs_vol.commit()

    best = outdir / "best.npz"
    sft = outdir / "sft" / "best.npz"
    return {
        "out": str(outdir),
        "best": best.stat().st_size if best.exists() else 0,
        "sft": sft.stat().st_size if sft.exists() else 0,
    }


@app.function(image=image, timeout=1800, volumes={"/runs": runs_vol, "/data": data_vol})
def evaluate(out: str = "big2", tokens: int = 80):
    ckpt = pathlib.Path("/runs") / out / "sft" / "best.npz"
    run(["python", "-m", "tools.check_chat", "--ckpt", str(ckpt), "--tok", "/data/tokenizer.json",
         "--tokens", str(tokens)])


@app.function(image=image, timeout=3600, volumes={"/runs": runs_vol, "/data": data_vol})
def export_onnx(out: str = "big2"):
    run(["python", "-m", "tools.export_onnx", "--ckpt", f"/runs/{out}/sft/best.npz",
         "--tok", "/data/tokenizer.json", "--out", f"/runs/{out}/onnx"])
    runs_vol.commit()


@app.local_entrypoint()
def main(gpu_job: str = "train", out: str = "big2", steps: int = 0, sft_steps: int = 1500):
    if gpu_job == "eval":
        evaluate.remote(out=out)
        return
    kw = {"out": out, "sft_steps": sft_steps}
    if steps:
        kw["steps"] = steps
    print(train.remote(**kw))
