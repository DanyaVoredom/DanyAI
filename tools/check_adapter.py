import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--expect-r", type=int, default=16)
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as z:
        names = z.namelist()
        cfg_name = "adapter_config.json"
        if cfg_name not in names:
            raise SystemExit("adapter_config.json missing")
        cfg = json.loads(z.read(cfg_name).decode("utf-8"))
        need = ["r", "lora_alpha", "target_modules", "base_model_name_or_path"]
        for k in need:
            if k not in cfg:
                raise SystemExit(f"missing key {k}")
        if int(cfg["r"]) != args.expect_r:
            raise SystemExit(f"unexpected r {cfg['r']}")
        weight = [n for n in names if n.endswith(".safetensors")]
        if not weight:
            raise SystemExit("no safetensors weights")
        data = z.read(weight[0])
        header_len = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8 + header_len].decode("utf-8"))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        lora = [k for k in tensors if "lora_" in k]
        if not lora:
            raise SystemExit("no lora tensors")
        modules = sorted({k.split(".lora_")[0].rsplit(".", 1)[-1] for k in lora})
        shapes = {tensors[k]["shape"] for k in lora}
        print(f"base {cfg['base_model_name_or_path']}")
        print(f"r {cfg['r']} alpha {cfg['lora_alpha']} modules {cfg['target_modules']}")
        print(f"tensors {len(lora)} files {names}")
        print(f"shapes {sorted(shapes)}")
        print(f"bytes {len(data)}")
        dense = sum(1 for s in shapes if len(s) == 2)
        if dense != len(shapes):
            raise SystemExit("unexpected lora tensor ranks")
        modules_actual = set(modules)
        missing = set(cfg["target_modules"]) - modules_actual
        if missing:
            raise SystemExit(f"modules not present in weights: {sorted(missing)}")
        print("adapter ok")


if __name__ == "__main__":
    main()
