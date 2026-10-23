import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "runs", "smoke")
RAW = os.path.join(WORK, "raw")
DATA = os.path.join(WORK, "data")


def run(args):
    r = subprocess.run([sys.executable, "-m"] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("failed: " + " ".join(args))
    return r.stdout


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(RAW)
    words = ["dany", "local", "model", "train", "data", "text", "small", "fast", "cpu", "python"]
    lines = []
    for i in range(4000):
        lines.append(" ".join(words[(i + j * 3) % len(words)] for j in range(12)) + "\n")
    with open(os.path.join(RAW, "corpus.txt"), "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(run(["danyai.prepare", "--input", RAW, "--out", DATA, "--vocab", "512"]))
    cfg = json.load(open(os.path.join(ROOT, "configs", "nano.json"), encoding="utf-8"))
    cfg["steps"] = 40
    cfg["batch_size"] = 8
    cfg["log_every"] = 10
    cfg["eval_every"] = 20
    cfg["save_every"] = 40
    cfg_path = os.path.join(WORK, "cfg.json")
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"))
    print(run(["danyai.train", "--config", cfg_path, "--data", DATA, "--out", os.path.join(WORK, "run")]))
    print(run(["danyai.sample", "--ckpt", os.path.join(WORK, "run", "best.npz"),
               "--tok", os.path.join(DATA, "tokenizer.json"), "--prompt", "dany", "--tokens", "40"]))
    print("smoke ok")


if __name__ == "__main__":
    main()