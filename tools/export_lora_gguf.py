import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True, help="путь к клону llama.cpp")
    ap.add_argument("--adapter", required=True, help="каталог с LoRA (adapter_config.json, adapter_model.safetensors)")
    ap.add_argument("--out", required=True, help="куда сохранить .gguf")
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32", "bf16", "q8_0"])
    args = ap.parse_args()

    conv = os.path.join(args.llama, "convert_lora_to_gguf.py")
    if not os.path.isfile(conv):
        raise SystemExit(f"нет {conv}")
    cmd = [sys.executable, conv, args.adapter, "--outfile", args.out, "--outtype", args.outtype]
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    print(f"готово: {args.out} {os.path.getsize(args.out)} байт")


if __name__ == "__main__":
    main()
